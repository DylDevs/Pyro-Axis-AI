from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn as nn
import traceback
import random
import torch
import time
import os

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
    
class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, max_length):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(2 * max_length * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text).view(text.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(embedded)))
        return self.softmax(self.fc2(x))
    
def GetMaxLength(data):
    max_length = 0
    for message, _, _ in data:
        if len(message) > max_length:
            max_length = len(message)
    return max_length
    
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

# If the hyperparameters do not have these keys, it will raise an error
hyperparameters_keys = ["data_path", "num_epochs", "batch_size", "classes", "learning_rate", "max_learning_rate",
    "train_val_ratio", "num_workers", "dropout", "patience", "shuffle_train", "shuffle_val",
    "shuffle_each_epoch", "pin_memory", "drop_last", "cache", "embedding_dim", "hidden_dim"]

class MissingHyperparametersException(Exception):
    pass

def Setup(hyperparameters, device : torch.device):
    for key in hyperparameters_keys:
        if key not in hyperparameters:
            return MissingHyperparametersException(f"Missing key '{key}' in hyperparameters"), None
    
    root_path = os.path.dirname(os.path.dirname(__file__)).replace("/", "\\")
    hyperparameters["data_path"] = root_path + "\\" + hyperparameters["data_path"].replace("/", "\\").replace("/", "\\\"")

    try:
        try:
            data = load_data(hyperparameters["data_path"])
        except FileNotFoundError:
            return "Data path not found", None, FileNotFoundError(f"Could not find data at {hyperparameters['data_path']}")
        hyperparameters["max_length"] = GetMaxLength(data)
        hyperparameters["train_dataset"], hyperparameters["val_dataset"], train_size, val_size = SplitDataset(data, hyperparameters["train_val_ratio"], hyperparameters["max_length"])
        hyperparameters["vocab_size"] = len(hyperparameters["train_dataset"].vocab) + 1

        model = NeuralNetwork(hyperparameters["vocab_size"], hyperparameters["embedding_dim"], hyperparameters["hidden_dim"], hyperparameters["classes"], hyperparameters["dropout"], hyperparameters["max_length"])
        train_dataloader = DataLoader(hyperparameters["train_dataset"], batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle_train"], num_workers=hyperparameters["num_workers"], pin_memory=hyperparameters["pin_memory"], drop_last=hyperparameters["drop_last"])
        if hyperparameters["val_dataset"] is not None:
            val_dataloader = DataLoader(hyperparameters["val_dataset"], batch_size=hyperparameters["batch_size"], shuffle=hyperparameters["shuffle_val"], num_workers=hyperparameters["num_workers"], pin_memory=hyperparameters["pin_memory"], drop_last=hyperparameters["drop_last"])
        else:
            val_dataloader = None
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=hyperparameters["max_learning_rate"], steps_per_epoch=len(train_dataloader), epochs=hyperparameters["num_epochs"])
    except Exception as e:
        return e, None, traceback.format_exc()

    # Your train loop should accept this data as an argument
    model_data = {
        "model": model,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "train_size": train_size,
        "val_size": val_size,
        "device": device,
        "hyperparameters": hyperparameters
    }

    return "ok", model_data, None

class MissingDataException(Exception):
    pass

def Train(model_data: dict):
    try:
        model = model_data["model"]
        train_dataloader = model_data["train_dataloader"]
        val_dataloader = model_data.get("val_dataloader", None)  # Use .get to handle missing val_dataloader
        criterion = model_data["criterion"]
        optimizer = model_data["optimizer"]
        scaler = model_data["scaler"]
        scheduler = model_data["scheduler"]
        train_size = model_data["train_size"]
        val_size = model_data["val_size"]
        hyperparameters = model_data["hyperparameters"]
    except Exception as e:
        return MissingDataException(e)

    epoch_training_start_time = time.time()
    if hyperparameters["shuffle_each_epoch"]:
        train_dataloader = DataLoader(
            hyperparameters["train_dataset"],
            batch_size=hyperparameters["batch_size"],
            shuffle=hyperparameters["shuffle_train"],
            num_workers=hyperparameters["num_workers"],
            pin_memory=hyperparameters["pin_memory"],
            drop_last=hyperparameters["drop_last"]
        )
        if hyperparameters["val_dataset"] is not None:
            val_dataloader = DataLoader(
                hyperparameters["val_dataset"],
                batch_size=hyperparameters["batch_size"],
                shuffle=hyperparameters["shuffle_val"],
                num_workers=hyperparameters["num_workers"],
                pin_memory=hyperparameters["pin_memory"],
                drop_last=hyperparameters["drop_last"]
            )

    # Training phase
    model.train()
    running_training_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(model_data["device"], non_blocking=True), data[1].to(model_data["device"], non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type=str(model_data["device"])):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_training_loss += loss.item()
    running_training_loss /= len(train_dataloader)
    training_loss = running_training_loss
    model_data["training_loss"] = training_loss

    # Validation phase (if validation data is available)
    if val_dataloader is not None and len(val_dataloader) > 0:
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad(), autocast(device_type=str(model_data["device"])):
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data[0].to(model_data["device"], non_blocking=True), data[1].to(model_data["device"], non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_validation_loss += loss.item()
        running_validation_loss /= len(val_dataloader)
        validation_loss = running_validation_loss
        model_data["val_loss"] = validation_loss
    else:
        model_data["val_loss"] = None # Indicate no validation loss available

    torch.cuda.empty_cache()
    return model_data