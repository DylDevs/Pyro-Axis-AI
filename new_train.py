import datetime
import sys

class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    DARK_GREY = "\033[90m"
    NORMAL = "\033[0m"

def timestamp():
    return Colors.DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + Colors.NORMAL

last_print_type = "normal"
def print(message: str, color: Colors = Colors.NORMAL, end: str = "\n", reprint : bool = False, show_timestamp: bool = True):
    global last_print_type
    if show_timestamp:
        start = timestamp()
    else:
        start = ""
    if not reprint:
        if last_print_type == "reprint":
            sys.stdout.write(f"\n{start}{color}{message}{Colors.NORMAL}{end}")
        else:
            sys.stdout.write(f"{start}{color}{message}{Colors.NORMAL}{end}")
        last_print_type = "normal"
    else:
        sys.stdout.write(f"\r{start}{color}{message}{Colors.NORMAL}                                       ")
        last_print_type = "reprint"

def empty_line():
    print("", show_timestamp=False)
    
def reset_reprint():
    # Allows for reprints in a row
    print("", end="", show_timestamp=False)

print("Importing libraries...", color=Colors.BLUE, reprint=True)
# Some of these modules aren't used in this file, importing them now will allow for faster load times later
from torch.utils.tensorboard import SummaryWriter
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
from pydantic import BaseModel
import torch.optim as optim
from fastapi import FastAPI
import torch.nn as nn
import subprocess
import traceback
import importlib
import threading
import logging
import uvicorn
import random
import socket
import torch
import time
import os

# Import the webserver
import webserver

print("Imported libraries", color=Colors.GREEN, reprint=True)
reset_reprint()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU detected", color=Colors.GREEN)
else:
    device = torch.device("cpu")
    print("GPU not detected, using CPU", color=Colors.YELLOW)

# Path vars
PATH = os.path.dirname(__file__)
MODEL_TYPES_PATH = PATH + "/modelTypes"

# Changable vars
start_frontend = True
start_webserver = True
debug = False

# Constants

model_types = {}
files = os.listdir(MODEL_TYPES_PATH)
for filename in files:
    if not filename.endswith(".py"):
        files.remove(filename)
print(f"Loading model architectures (0/{len(files)})", color=Colors.BLUE, reprint=True)

for i, filename in enumerate(files):
    print(f"Loading model architectures ({i + 1}/{len(files)})", color=Colors.BLUE, reprint=True)
    model_import = importlib.import_module(f"modelTypes.{filename[:-3]}")
    model_types[filename[:-3]] = model_import

print(f"Loaded {len(model_types)} model architectures", color=Colors.GREEN, reprint=True)
empty_line()

class TrainingSession:
    class Exceptions:
        class InvalidModelType(Exception): pass
        class MissingHyperparameters(Exception): pass
        class ModelNotSetup(Exception): pass

    def __init__(self, model_type: str, hyperparameters: dict):
        self.model_type = model_type
        self.model_display_name = model_type.replace("_", " ").title()
        self.hyperparameters = hyperparameters

        self.model_data = None
        self.patience_epochs = 0

        self.status = "Queued"
        self.epoch = 0
        self.epoch_start_time = 0
        self.training_loss = 0
        self.val_loss = 0
        self.elapsed = 0
        self.time_per_epoch = 0
        
        self.stop_type = None
        self.best_epoch = 0
        self.best_training_loss = 0
        self.best_val_loss = 0
        self.training_start_time = 0
        self.training_end_time = 0
        self.training_dataset_accuracy = 0
        self.val_dataset_accuracy = 0
        self.best_model = None
        
    def setup(self):
        # When an exception is returned, the webserver will decode it
        if self.model_type not in model_types:
            return self.Exceptions.InvalidModelType
        if not self.hyperparameters:
            return self.Exceptions.MissingHyperparameters
        
        # Set up the model (model_data is alot of things, doesn't need to be unpacked since it will be passed to other functions)
        model_setup_status, self.model_data = model_types[self.model_type].Setup(self.hyperparameters, device)
        if model_setup_status != "ok":
            return model_setup_status # This will be an exception
        
        return "ok"
    
    def _elapsed(self):
        while self.status == "Training":
            self.elapsed = time.time() - self.training_start_time
            time.sleep(0.2)
    
    def _train(self):
        self.status = "Training"
        self.training_start_time = time.time()

        for epoch in range(self.hyperparameters["epochs"]):
            self.epoch = epoch
            self.epoch_start_time = time.time()

            self.model_data = model_types[self.model_type].Train(self.model_data)

            self.training_loss = self.model_data["training_loss"]
            self.val_loss = self.model_data["val_loss"]
            self.time_per_epoch = time.time() - self.epoch_start_time

            if self.model_data["val_loss"] < self.model_data["best_val_loss"]:
                self.best_val_loss = self.model_data["val_loss"]
                self.best_training_loss = self.model_data["training_loss"]
                self.best_model = self.model_data["model"]
                self.best_epoch = epoch
                self.patience_epochs -= 1
                if self.patience_epochs == 0:
                    self.status = "Finished"
                    self.stop_type = "Patience"
                    break
            else:
                self.patience_epochs = self.hyperparameters["patience"]

            if self.status != "Training":
                break

    def start_training(self):
        if self.model_data is None:
            return self.Exceptions.ModelNotSetup
        
        train_thread = threading.Thread(target=self._train)
        train_thread.start()
        elapsed_thread = threading.Thread(target=self._elapsed)
        elapsed_thread.start()

    def ManualStop(self):
        self.status = "Finished"
        self.stop_type = "Manual"
        self.training_end_time = time.time()

    def GetModelStatus(self):
        if self.status != "Finished":
            return {"status": self.status,
                "type": self.model_display_name,
                "epoch": self.epoch,
                "epoch_start_time": self.epoch_start_time,
                "training_loss": self.training_loss,
                "val_loss": self.val_loss,
                "elapsed": self.elapsed,
                "time_per_epoch": self.time_per_epoch}
        else:
            return {"status": self.status,
                "type": self.model_display_name,
                "stop_type": self.stop_type,
                "best_epoch": self.best_epoch,
                "epoch": self.epoch,
                "bext_training_loss": self.best_training_loss,
                "best_val_loss": self.best_val_loss,
                "training_start_time": self.training_start_time,
                "training_end_time": self.training_end_time,
                "elapsed": self.elapsed,
                "training_dataset_accuracy": self.training_dataset_accuracy,
                "val_dataset_accuracy": self.val_dataset_accuracy}

frontend_url, backend_url = webserver.run(frontend=start_frontend, backend=start_webserver, debug=debug)
if frontend_url:
    print(f"Frontend URL: {frontend_url} (localhost:3000)", color=Colors.GREEN)
if backend_url:
    print(f"Backend URL: {backend_url} (localhost:8000)", color=Colors.GREEN)

print("Awaiting client connection...", color=Colors.BLUE, reprint=True)
# Pass the training session class since we cant inpot it in the webserver (circular import)
client_ip = webserver.WaitForClient(TrainingSession)
print(f"Client connected at {client_ip}!", color=Colors.GREEN, reprint=True)