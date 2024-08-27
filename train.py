import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from PIL import Image
import numpy as np
import traceback
import threading
import warnings
import curses
import random
import shutil
import torch
import time
import cv2
import sys

# Remove warnings about CUDA
warnings.filterwarnings("ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")

# Constants
PATH = os.path.dirname(__file__)
DATA_PATH = PATH + "\\dataset"
MODEL_PATH = PATH + "\\models"
VOCAB_FILE = PATH + "\\vocab.txt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_TYPE = "language"
if TRAIN_TYPE == "language":
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    CLASSES = 3
    LEARNING_RATE = 0.003
    MAX_LEARNING_RATE = 0.003
    TRAIN_VAL_RATIO = 0.90
    NUM_WORKERS = 0
    DROPOUT = 0.475
    PATIENCE = 25
    SHUFFLE_TRAIN = True
    SHUFFLE_VAL = True
    SHUFFLE_EACH_EPOCH = True
    PIN_MEMORY = False
    DROP_LAST = True
    CACHE = True
    EMBEDDING_DIM = 110
    HIDDEN_DIM =  350
TRAIN_ATTEMPTS = 2
MAX_TRAIN_THREADS = 1
if TRAIN_ATTEMPTS > 49:
    TRAIN_ATTEMPTS = 49
    print("WARNING: TRAIN_ATTEMPTS > 49, setting to 49. (It is impossible to train with more than 49 threads)")
elif TRAIN_ATTEMPTS < 1:
    TRAIN_ATTEMPTS = 1
    print("WARNING: TRAIN_ATTEMPTS < 1, setting to 1. (It is impossible to train with less than 1 thread)")
elif MAX_TRAIN_THREADS > 10:
     print("WARNING: MAX_TRAIN_THREADS > 10, It is not recommended to train with more than 10 threads")

FINAL_TRAIN_DATA = [{"best_model": None, 
               "training_dataset_accuracy": 0, 
               "val_dataset_accuracy": 0,
               "best_epoch": 0,
               "end_epoch": 0,
               "best_training_loss": 0,
               "best_val_loss": 0,
               "training_time": "",
               "training_date": ""}] * TRAIN_ATTEMPTS
VARIABLE_TRAIN_DATA = [{"status": "queued",
                        "exception": None,
                        "epoch": 0, 
                        "epoch_start_time": 0,
                        "training_loss": 0, 
                        "val_loss": 0, 
                        "elapsed": 0, 
                        "time_per_epoch": 0,
                        "summary_writer": None}] * TRAIN_ATTEMPTS

DATAPOINTS = 0
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        DATAPOINTS += 1
if DATAPOINTS == 0:
    print("No data found, exiting...")
    exit()

RED = "\033[91m"
GREEN = "\033[92m"
DARK_GREY = "\033[90m"
NORMAL = "\033[0m"

def timestamp():
    return DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + NORMAL

# Custom dataset class
def load_data(data_folder):
    data = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), 'r') as file:
                lines = file.readlines()
                message = lines[0].strip()
                channel = lines[1].strip()
                output = int(lines[2].strip())
                data.append((message, channel, output))
    return data
        
class CustomDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length
        self.vocab = self.build_vocab()

    def build_vocab(self):
        vocab = set()
        for message, channel, _ in self.data:
            vocab.update(message)
            vocab.update(channel)
        return {char: idx + 1 for idx, char in enumerate(vocab)}

    def encode(self, text):
        encoded = [self.vocab.get(char, 0) for char in text]
        return encoded + [0] * (self.max_length - len(encoded))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        message, channel, label = self.data[idx]
        message_encoded = self.encode(message)[:self.max_length]
        channel_encoded = self.encode(channel)[:self.max_length]
        return torch.tensor(message_encoded + channel_encoded), torch.tensor(label)

# Define the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, max_length):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(2 * max_length * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text).view(text.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(embedded)))
        return self.softmax(self.fc2(x))
    
def SplitDataset(data, train_val_ratio, max_length):
    datapoints = len(data)
    train_amount = int(train_val_ratio * datapoints)
    val_amount = datapoints - train_amount

    train_indices = random.sample(range(datapoints), train_amount)
    val_indices = [i for i in range(datapoints) if i not in train_indices]

    train_data = []
    val_data = []

    for i in train_indices:
        train_data.append(data[i])
    for i in val_indices:
        val_data.append(data[i])

    train_dataset = CustomDataset(train_data, max_length=max_length)
    val_dataset = CustomDataset(val_data, max_length=max_length)

    return train_dataset, val_dataset, train_amount, val_amount

def GetMaxLength():
    data = load_data(DATA_PATH)
    max_length = 0
    for message, _, _ in data:
        if len(message) > max_length:
            max_length = len(message)
    return max_length

def TrainingLoop(i: int, train_dataset: CustomDataset, val_dataset: CustomDataset, vocab_size: int, max_length: int, model_variable_data: dict, model_final_data: dict):
    try:
        model_variable_data["status"] = "initializing"
        total_start_time = time.time()

        def update_elapsed():
            while model_variable_data["status"] == "training":
                model_variable_data["elapsed"] = time.time() - total_start_time
                time.sleep(0.2)

        # Initialize training assets
        summary_writer = model_variable_data["summary_writer"]
        model = ConvolutionalNeuralNetwork(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, CLASSES, DROPOUT, max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_VAL)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)
        
        # Early stopping variables
        best_validation_loss = float('inf')
        best_model = None
        best_model_epoch = None
        best_model_training_loss = None
        best_model_validation_loss = None
        wait = 0

        training_start_time = time.time()
        epoch_total_time = 0
        training_loss = 0
        validation_loss = 0

        model_variable_data["status"] = "training"
        threading.Thread(target=update_elapsed).start()
        for epoch, _ in enumerate(range(NUM_EPOCHS), 1):
            model_variable_data["epoch"] = epoch
            epoch_start_time = time.time()
            model_variable_data["epoch_start_time"] = epoch_start_time
            epoch_training_start_time = time.time()

            if SHUFFLE_EACH_EPOCH: 
                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_VAL)

            # Training phase
            model.train()
            running_training_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_training_loss += loss.item()
            running_training_loss /= len(train_dataloader)
            training_loss = running_training_loss
            model_variable_data["training_loss"] = training_loss

            epoch_training_time = time.time() - epoch_training_start_time
            epoch_validation_start_time = time.time()

            # Validation phase
            model.eval()
            running_validation_loss = 0.0
            with torch.no_grad(), autocast():
                for i, data in enumerate(val_dataloader, 0):
                    inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_validation_loss += loss.item()
            running_validation_loss /= len(val_dataloader)
            validation_loss = running_validation_loss
            model_variable_data["val_loss"] = validation_loss

            epoch_validation_time = time.time() - epoch_validation_start_time
            total_epoch_time = time.time() - epoch_start_time
            model_variable_data["time_per_epoch"] = total_epoch_time

            torch.cuda.empty_cache()

            # Early stopping
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model = model
                best_model_epoch = epoch
                best_model_training_loss = training_loss
                best_model_validation_loss = validation_loss
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE and PATIENCE > 0:
                    # Log values to Tensorboard
                    summary_writer.add_scalars(f'Stats', {
                        'train_loss': training_loss,
                        'validation_loss': validation_loss,
                        'epoch_total_time': total_epoch_time,
                        'epoch_training_time': epoch_training_time,
                        'epoch_validation_time': epoch_validation_time
                    }, epoch)
                    break

            # Log values to Tensorboard
            summary_writer.add_scalars(f'Stats', {
                'train_loss': training_loss,
                'validation_loss': validation_loss,
                'epoch_total_time': total_epoch_time,
                'epoch_training_time': epoch_training_time,
                'epoch_validation_time': epoch_validation_time
            }, epoch)
        
        total_time = time.time() - total_start_time
        training_complete_time = time.time()

        torch.cuda.empty_cache()

        best_model.eval()
        total_train = 0
        correct_train = 0
        with torch.no_grad():
            for data in train_dataloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = best_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        training_dataset_accuracy = str(round(100 * (correct_train / total_train), 2)) + "%"

        torch.cuda.empty_cache()

        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for data in val_dataloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = best_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        validation_dataset_accuracy = str(round(100 * (correct_val / total_val), 2)) + "%"

        torch.cuda.empty_cache()

        model_final_data["best_model"] = best_model
        model_final_data["training_dataset_accuracy"] = training_dataset_accuracy
        model_final_data["val_dataset_accuracy"] = validation_dataset_accuracy
        model_final_data["best_epoch"] = best_model_epoch
        model_final_data["end_epoch"] = epoch
        model_final_data["bext_train_loss"] = best_model_training_loss
        model_final_data["best_val_loss"] = best_model_validation_loss
        model_final_data["total_time"] = total_time
        model_final_data["training_time"] = training_complete_time
        model_final_data["training_date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        model_variable_data["status"] = "finished"
        return
    except:
        model_variable_data["status"] = "error"
        model_variable_data["exception"] = traceback.format_exc()
        
def main():
    models_trained = 0
    active_threads = 0
    threads = []
    datapoints = 0

    data = load_data(DATA_PATH)
    MAX_LENGTH = GetMaxLength()
    
    train_dataset, val_dataset, train_size, val_size = SplitDataset(data, TRAIN_VAL_RATIO, MAX_LENGTH)

    # Initialize model
    VOCAB_SIZE = len(train_dataset.vocab) + 1

    def get_model_size_mb(model):
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.size())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        bytes_per_param = next(model.parameters()).element_size()
        model_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
        return total_params, trainable_params, non_trainable_params, model_size_mb

    example_model = ConvolutionalNeuralNetwork(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, CLASSES, DROPOUT, MAX_LENGTH)
    total_params, trainable_params, non_trainable_params, model_size_mb = get_model_size_mb(example_model)
    del example_model

    print("\n----------------------------------------------\n")
    print(timestamp() + f"Using {str(DEVICE).upper()} for training")
    print(timestamp() + 'Number of CPU cores:', multiprocessing.cpu_count())
    print()
    print(timestamp() + "Model properties:")
    print(timestamp() + f"> Total parameters: {total_params}")
    print(timestamp() + f"> Trainable parameters: {trainable_params}")
    print(timestamp() + f"> Non-trainable parameters: {non_trainable_params}")
    print(timestamp() + f"> Predicted model size: {model_size_mb:.2f}MB")
    print()
    print(timestamp() + "Training settings:")
    print(timestamp() + "> Epochs:", NUM_EPOCHS)
    print(timestamp() + "> Batch size:", BATCH_SIZE)
    print(timestamp() + "> Classes:", CLASSES)
    print(timestamp() + "> Data points:", DATAPOINTS)
    print(timestamp() + "> Learning rate:", LEARNING_RATE)
    print(timestamp() + "> Max learning rate:", MAX_LEARNING_RATE)
    print(timestamp() + "> Dataset split:", TRAIN_VAL_RATIO)
    print(timestamp() + "> Number of workers:", NUM_WORKERS)
    print(timestamp() + "> Dropout:", DROPOUT)
    print(timestamp() + "> Patience:", PATIENCE)
    print(timestamp() + "> Shuffle Train:", SHUFFLE_TRAIN)
    print(timestamp() + "> Shuffle Val:", SHUFFLE_VAL)
    print(timestamp() + "> Pin memory:", PIN_MEMORY)
    print(timestamp() + "> Drop last:", DROP_LAST)
    print(timestamp() + "> Cache:", CACHE)
    print(timestamp() + "> Embedding dim:", EMBEDDING_DIM)
    print(timestamp() + "> Hidden dim:", HIDDEN_DIM)
    print(timestamp() + f"> Vocabulary size: {VOCAB_SIZE}")
    print(timestamp() + f"> Max length: {MAX_LENGTH}")

    print("\n----------------------------------------------\n")

    print(timestamp() + "Loading...")

    # Create tensorboard logs folder if it doesn't exist
    if not os.path.exists(f"{PATH}/logs"):
        os.makedirs(f"{PATH}/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/logs"):
        try:
            shutil.rmtree(f"{PATH}/logs/{obj}")
        except:
            os.remove(f"{PATH}/logs/{obj}")

    # Create Tensorboard writer for each training attempt
    for i in range(TRAIN_ATTEMPTS):
        VARIABLE_TRAIN_DATA[i]["summary_writer"] = SummaryWriter(f"{PATH}/Training/Classification/logs/{i + 1}", comment=f"Classification-Training-{i}", flush_secs=20)

    for i in range(TRAIN_ATTEMPTS):
        print(f"Assigned variable list {VARIABLE_TRAIN_DATA[i]} to thread {i}")
        threads.append(threading.Thread(target=TrainingLoop, args=(i, train_dataset, val_dataset, VOCAB_SIZE, MAX_LENGTH, VARIABLE_TRAIN_DATA[i], FINAL_TRAIN_DATA[i])))

    def clear_lines(screen, num_lines):
        """Clear num_lines from the screen by moving up and clearing."""
        current_line = screen.getyx()[0]
        for _ in range(num_lines):
            if current_line > 0:
                screen.move(current_line - 1, 0)
                screen.clrtoeol()
                current_line -= 1

    def run_training(screen):
        screen.addstr("\n----------------------------------------------\n")
        screen.addstr(f"{timestamp()} Training...\n")

        models_trained = 0
        active_threads = 0
        thread_index = 0
        first_time = True
        debug_mode = True

        training_start = time.time()
        while True:
            for i, thread in enumerate(threads):
                if not thread.is_alive() and VARIABLE_TRAIN_DATA[i]["status"] == "finished":
                    active_threads -= 1
                    models_trained += 1

            if models_trained == TRAIN_ATTEMPTS:
                break

            if active_threads < MAX_TRAIN_THREADS:
                active_threads += 1
                threads[thread_index].start()
                thread_index += 1

            error = False
            for i in range(TRAIN_ATTEMPTS):
                if VARIABLE_TRAIN_DATA[i]["status"] == "error":
                    error = True
            if error:
                break

            screen.addstr(0, 0, f"Training models... Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - training_start))}, ETA: Coming Soon, ({models_trained}/{TRAIN_ATTEMPTS})\n")
            if debug_mode:
                screen.addstr(1, 0, f"{VARIABLE_TRAIN_DATA}\n")
                progress_gap = 6
            else:
                progress_gap = 1
            for i in range(TRAIN_ATTEMPTS):
                model_data = VARIABLE_TRAIN_DATA[i]
                final_model_data = FINAL_TRAIN_DATA[i]
                if model_data["status"] == "queued":
                    screen.addstr(i + progress_gap, 0, f"Model {i + 1} Status: Queued\n")
                elif model_data["status"] == "initializing":
                    screen.addstr(i + progress_gap, 0, f"Model {i + 1} Status: Initializing...\n")
                elif model_data["status"] == "training":
                    epoch = model_data["epoch"]
                    if epoch > 1:
                        epoch_start_time = model_data["epoch_start_time"]
                        train_loss = model_data["training_loss"]
                        val_loss = model_data["val_loss"]
                        time_per_epoch = model_data["time_per_epoch"]

                        epochs_left = NUM_EPOCHS - (epoch - 1)
                        eta = time.strftime("%H:%M:%S", time.gmtime(epochs_left * time_per_epoch))

                        # Create progress bar
                        progress_incriment = time_per_epoch / 20
                        epoch_elapsed = time.time() - epoch_start_time
                        progress = 0
                        bars = 0
                        while progress < epoch_elapsed:
                            progress += progress_incriment
                            bars += 1
                            if bars == 20:
                                break
                        progress_string = ""
                        for _ in range(bars):
                            progress_string += "█"
                        while len(progress_string) < 20:
                            progress_string += "░"

                        screen.addstr(i + progress_gap, 0, f"Model {i + 1} Status: Training | {progress_string} Epoch {epoch}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, {time_per_epoch:.2f}s/Epoch, ETA: {eta}\n")
                    else:
                        screen.addstr(i + progress_gap, 0, f"Model {i + 1} Status: Training | Epoch 1 (Data transmission will start next epoch)\n")
                elif model_data["status"] == "finished":
                    total_time = final_model_data["total_time"]
                    train_loss = final_model_data["best_training_loss"]
                    val_loss = final_model_data["best_val_loss"]
                    epoch = final_model_data["end_epoch"]
                    best_epoch = final_model_data["best_epoch"]
                    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
                    training_dataset_accuracy = final_model_data["training_dataset_accuracy"]
                    val_dataset_accuracy = final_model_data["val_dataset_accuracy"]

                    screen.addstr(i + progress_gap, 0, f"Model {i + 1} Status: Finished | Epoch {epoch} (Best Epoch: {best_epoch}), Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Total Time: {formatted_time}, Accuracies: {training_dataset_accuracy:.1f}%, {val_dataset_accuracy:.1f}%\n")

            screen.refresh()

    input("Are you ready to start training? This will clear the screen. Press Enter to continue...")
    os.system('cls' if os.name=='nt' else 'clear')

    # Run the training in curses wrapper
    training_start = time.time()
    curses.wrapper(run_training)
    training_end = time.time()

    os.system('cls' if os.name=='nt' else 'clear')

    for i in range(TRAIN_ATTEMPTS):
        if VARIABLE_TRAIN_DATA[i]["status"] == "error":
            print(f"\n{timestamp()} Training Failed | Model {i + 1} | Error:\n{VARIABLE_TRAIN_DATA[i]['exception']}")
            input("Press Enter to exit...")
            exit()

    print(f"\n----------------------------------------------\n")
    print(f"{timestamp()} Training Complete | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(training_end - training_start))}")

    best_model_val_loss = float('inf')
    best_model_indice = 0
    for i in range(TRAIN_ATTEMPTS):
        if FINAL_TRAIN_DATA[i]["best_val_loss"] < best_model_val_loss:
            best_model_val_loss = FINAL_TRAIN_DATA[i]["best_val_loss"]
            best_model_indice = i
        
    best_model_data = FINAL_TRAIN_DATA[best_model_indice]
    bext_model = best_model_data["best_model"]
    training_dataset_accuracy = best_model_data["training_dataset_accuracy"]
    val_dataset_accuracy = best_model_data["val_dataset_accuracy"]
    best_epoch = best_model_data["best_epoch"]
    end_epoch = best_model_data["end_epoch"]
    best_training_loss = best_model_data["best_training_loss"]
    best_val_loss = best_model_data["best_val_loss"]
    total_time = best_model_data["total_time"]
    training_time = best_model_data["training_time"]
    training_date = best_model_data["training_date"]


    print(f"{timestamp()} Best Model Data:")
    print(f"Training Dataset Accuracy: {training_dataset_accuracy:.1f}%")
    print(f"Validation Dataset Accuracy: {val_dataset_accuracy:.1f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"End Epoch: {end_epoch}")
    print(f"Best Training Loss: {best_training_loss:.8f}")
    print(f"Best Validation Loss: {best_val_loss:.8f}")
    print(f"Total Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Training Time: {training_time}")
    print(f"Training Date: {training_date}")

    print(f"\n----------------------------------------------\n")

    print(f"{timestamp()} Saving best model...")

    metadata_model = str(best_model).replace('\n', '')
    metadata = (f"epochs#{end_epoch}",
                f"best_epoch#{best_epoch}",
                f"batch#{BATCH_SIZE}",
                f"classes#{CLASSES}",
                f"datapoints#{DATAPOINTS}",
                f"learning_rate#{LEARNING_RATE}",
                f"max_learning_rate#{MAX_LEARNING_RATE}",
                f"dataset_split#{TRAIN_VAL_RATIO}",
                f"number_of_workers#{NUM_WORKERS}",
                f"dropout#{DROPOUT}",
                f"patience#{PATIENCE}",
                f"train_shuffle#{SHUFFLE_TRAIN}",
                f"val_shuffle#{SHUFFLE_VAL}",
                f"pin_memory#{PIN_MEMORY}",
                f"hidden_dim#{HIDDEN_DIM}",
                f"embedding_dim#{EMBEDDING_DIM}",
                f"max_length#{MAX_LENGTH}",
                f"vocab_size#{VOCAB_SIZE}",
                f"training_time#{training_time}",
                f"training_date#{training_date}",
                f"training_device#{DEVICE}",
                f"training_os#{os.name}",
                f"architecture#{metadata_model}",
                f"torch_version#{torch.__version__}",
                f"numpy_version#{np.__version__}",
                f"pil_version#{Image.__version__}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{best_training_loss}",
                f"validation_loss#{best_val_loss}",
                f"training_dataset_accuracy#{training_dataset_accuracy}",
                f"validation_dataset_accuracy#{val_dataset_accuracy}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    best_model_saved = False
    for i in range(5):
        try:
            best_model = torch.jit.script(best_model)
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"LanguageClassification-BEST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")
        
    with open(VOCAB_FILE, 'w') as file:
        for char, idx in train_dataset.vocab.items():
            file.write(f"{char} {idx}\n")
    print(timestamp() + "Saved vocabulary successfully.")
    print("Language classification model training complete!")

    print("\n----------------------------------------------\n")
    input("Press Enter to exit...")
    exit()

if __name__ == '__main__':
    main()