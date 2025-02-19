from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import random
import torch
import json
import os
import re

# Resources to create models for the Pyro Axis AI Training Hub
from utils.modules import *

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, max_length=128, num_layers=2, num_heads=2, hidden_dim=128, dropout_rate=0.1):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate
        )
        # Reduce the number of transformer layers
        self.transformer_blocks = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()
        position_ids = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_seq) + self.position_embedding(position_ids)
        x = self.layer_norm(x)
        
        # Use a simplified target mask
        target_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_seq.device), diagonal=1).bool()
        
        memory = torch.zeros_like(x).transpose(0, 1)  # Placeholder memory
        x = self.transformer_blocks(x.transpose(0, 1), memory=memory, tgt_mask=target_mask)
        x = self.fc_out(x.transpose(0, 1))
        
        return x

class Model(ModelTemplate):
    def __init__(self, json_data: dict) -> None:
        super().__init__(json_data)

        # Set device
        hyp_device = self.GetHyp("Device")
        if hyp_device == "Auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif hyp_device == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hyp_device == "CPU":
            self.device = torch.device("cpu")
        else:
            print(f"GPU was selected, but CUDA is not available. Using CPU instead.", color=Colors.YELLOW)
            self.device = torch.device("cpu")

        # Variables for training UI
        self.epochs_until_patience = 0
        self.train_size = 0
        self.val_size = 0
        self.max_length = 0
        self.vocab_size = 0
        self.total_parameters = 0
        self.trainable_parameters = 0
        self.non_trainable_parameters = 0
        self.model_size = 0
        self.optimizer_str = ""
        self.criterion_str = ""
        self.scaler_str = ""
        self.warmup_scheduler_str = ""
        self.scheduler_str = ""
        self.learning_rates = []

    def CreateTrainDataLoader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.GetHyp("Batch Size"), shuffle=self.GetHyp("Shuffle Train"), num_workers=self.GetHyp("Num Workers"), pin_memory=self.GetHyp("Pin Memory"), drop_last=self.GetHyp("Drop Last"), collate_fn=self.collate_fn)
    
    def CreateValDataLoader(self):
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.GetHyp("Batch Size"), shuffle=self.GetHyp("Shuffle Val"), num_workers=self.GetHyp("Num Workers"), pin_memory=self.GetHyp("Pin Memory"), drop_last=self.GetHyp("Drop Last"), collate_fn=self.collate_fn)

    def load_data(self):
        path = f"{self.data_path}/{self.GetHyp('Data JSON')}"
        if not os.path.exists(path): raise FileNotFoundError(f"Data JSON {path} does not exist")
        with open(path, "r") as f:
            dataset = json.load(f)
        qa = dataset["questions_and_answers"]
        if not qa: raise Exception("No questions and answers found in JSON data")
        random.shuffle(qa) # Shuffle for randomness
        return qa # [{"question": "q", "answer": "a"}, ...]

    class Vocab:
        def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
            self.pad_token = pad_token
            self.unk_token = unk_token
            self.word2idx = {self.pad_token: 0, self.unk_token: 1}
            self.idx2word = {0: self.pad_token, 1: self.unk_token}
            self.freeze_vocab = False

        def build_vocab(self, data):
            for item in data:
                for token in self.tokenize(item["question"]) + self.tokenize(item["answer"]):
                    if token not in self.word2idx and not self.freeze_vocab:
                        idx = len(self.word2idx)
                        self.word2idx[token] = idx
                        self.idx2word[idx] = token

        @staticmethod
        def tokenize(text: str):
            return re.findall(r'\w+', text.lower())

        def text_to_sequence(self, text):
            return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in self.tokenize(text)]

        def freeze(self):
            self.freeze_vocab = True  # Stop adding new tokens

        def __len__(self):
            return len(self.word2idx)
    
    class QADataset(Dataset):
        def __init__(self, data, vocab): # vocab is a Vocab class (can't be defined in the type signature)
            self.data = data
            self.vocab = vocab

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question = torch.tensor(self.vocab.text_to_sequence(self.data[idx]["question"]))
            answer = torch.tensor(self.vocab.text_to_sequence(self.data[idx]["answer"]))
            return question, answer
    
    def collate_fn(self, batch):
        questions, answers = zip(*batch)
        questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
        answers_padded = pad_sequence(answers, batch_first=True, padding_value=0)
        return questions_padded, answers_padded
    
    def SplitDataset(self, data, vocab, train_val_ratio) -> tuple[QADataset, QADataset, int, int]:
        """
        Split the data into train and validation sets.

        Args:
            data (list): The data to split.
            vocab (Vocab): The vocabulary used to tokenize the data.
            train_val_ratio (float): The ratio of train data to validation data.

        Returns:
            tuple: A tuple containing the train and validation datasets along with the number of train and validation data points.
        """
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

        train_dataset = self.QADataset(train_data, vocab)
        val_dataset = self.QADataset(val_data, vocab)

        return train_dataset, val_dataset, train_amount, val_amount
    
    def GetMaxLength(self, data):
        """
        Find the maximum length of a message in the data.

        Args:
            data (list): The data to find the maximum length from.

        Returns:
            int: The maximum length of a message in the data.
        """
        max_length = 0
        for pair in data:
            q = pair["question"]
            if len(q) > max_length:
                max_length = len(q)
        return max_length

    def GetModelSize(self, model : nn.Module):
        """Get the estimated size of a model in MB."""
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.size())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        bytes_per_param = next(model.parameters()).element_size()
        model_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
        return total_params, trainable_params, non_trainable_params, model_size_mb
    
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            # Linearly increase learning rate
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0

    def Initialize(self):
        """This function sets up the model and data for training. (Called by the training controller)"""
        # Load data
        data = self.load_data()
        self.max_length = self.GetMaxLength(data)

        # Calculate total number of steps and warmup steps
        self.total_steps = (len(data) * self.GetHyp("Epochs")) // self.GetHyp("Batch Size")
        self.warmup_steps = int(self.total_steps * self.GetHyp("Warmup Percentage"))

        # Create vocabulary and dataset
        vocab = self.Vocab()
        vocab.build_vocab(data)

        # Split the dataset into train and validation sets
        self.train_dataset, self.val_dataset, self.train_size, self.val_size = self.SplitDataset(data, vocab, self.GetHyp("Train Val Ratio"))
        self.vocab_size = len(self.train_dataset.vocab) + 1

        # Create data loaders
        self.CreateTrainDataLoader()
        self.CreateValDataLoader()

        # Create model
        self.model = GPT(len(vocab), self.GetHyp("Embedding Dim"), self.max_length, self.GetHyp("Num Layers"), self.GetHyp("Num Heads"), self.GetHyp("Hidden Dim"), self.GetHyp("Dropout")).to(self.device)
        self.total_parameters, self.trainable_parameters, self.non_trainable_parameters, self.model_size = self.GetModelSize(self.model)

        # Initialize optimizer and loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.GetHyp("Learning Rate"), weight_decay=self.GetHyp("Weight Decay"))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scaler = GradScaler()

        # Initialize schedulers
        self.warmup_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.GetHyp("Scheduler Factor"), patience=self.GetHyp("Scheduler Patience"))

        # Create string representation of utilities
        self.optimizer_str = type(self.optimizer).__name__
        self.criterion_str = type(self.criterion).__name__
        self.scaler_str = type(self.scaler).__name__
        self.warmup_scheduler_str = type(self.warmup_scheduler).__name__
        self.scheduler_str = type(self.scheduler).__name__

    def Train(self):
        """This function trains the model. (Called by the training controller every epoch)"""
        if self.GetHyp("Shuffle Each Epoch"):
            self.CreateTrainDataLoader() if self.GetHyp("Shuffle Train") else None
            self.CreateValDataLoader() if self.GetHyp("Shuffle Val") else None

        # Training Phase
        self.model.train()
        running_training_loss = 0.0
        with autocast(device_type=str(self.device)):
            for questions, answers in self.train_dataloader:
                # Prepare inputs and targets for training
                inputs = answers[:, :-1]
                targets = answers[:, 1:].contiguous().view(-1)
                questions, inputs, targets = questions.to(self.device), inputs.to(self.device), targets.to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                outputs = outputs.view(-1, len(self.train_dataset.vocab))  # Reshape outputs to match target size
                loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.GetHyp("Max Gradient Norm"))

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.warmup_scheduler.step()

                # Accumulate training loss
                running_training_loss += loss.item()

        running_training_loss /= len(self.train_dataloader)
        training_loss = running_training_loss

        # Validation Phase
        self.model.eval()
        running_validation_loss = 0.0
        with torch.no_grad(), autocast(device_type=str(self.device)):
            for val_questions, val_answers in self.val_dataloader:
                # Prepare inputs and targets for validation
                val_inputs = val_answers[:, :-1]
                val_targets = val_answers[:, 1:].contiguous().view(-1)
                val_questions, val_inputs, val_targets = val_questions.to(self.device), val_inputs.to(self.device), val_targets.to(self.device)

                # Forward pass for validation
                val_outputs = self.model(val_inputs)
                val_outputs = val_outputs.view(-1, len(self.train_dataset.vocab))  # Reshape to match targets
                val_loss = self.criterion(val_outputs, val_targets)
                running_validation_loss += val_loss.item()

        running_validation_loss /= len(self.val_dataloader)
        validation_loss = running_validation_loss

        # Calculate weighted average of training and validation loss for scheduler step
        combined_loss = (self.GetHyp("Combined Loss Ratio") * training_loss) + ((1 - self.GetHyp("Combined Loss Ratio")) * validation_loss)
        self.scheduler.step(combined_loss)

        self.train_losses.append(training_loss)
        self.val_losses.append(validation_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
    def AfterTrain(self, controller : TrainingController):
        torch.cuda.empty_cache()

        # Update epochs until patience
        if controller.best_epoch == controller.epoch:
            self.epochs_until_patience = 0
        else:
            self.epochs_until_patience += 1

    def Save(self, model : GPT):
        """This function saves the model. (Called by the training controller)"""
        torch.cuda.empty_cache()

        model_name = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"

        path = os.path.join(self.models_path, self.GetHyp("Model Save Path"))
        if not os.path.exists(path):
            os.makedirs(path)

        torch.jit.save(torch.jit.script(model), os.path.join(path, model_name))