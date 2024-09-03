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
import os

# Import the webserver
import webserver

print("Imported libraries", color=Colors.GREEN, reprint=True)
reset_reprint()

# Path vars
PATH = os.path.dirname(__file__)
MODEL_TYPES_PATH = PATH + "/modelTypes"

# Changable vars
start_frontend = True
start_webserver = True
debug = False

# Constants


# Classes
class Model:
    def __init__(self, name: str, model_type: str):



# Functions

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
frontend_url, backend_url = webserver.run(frontend=start_frontend, backend=start_webserver, debug=debug)
if frontend_url:
    print(f"Frontend URL: {frontend_url} (localhost:3000)", color=Colors.GREEN)
if backend_url:
    print(f"Backend URL: {backend_url} (localhost:8000)", color=Colors.GREEN)
print("Awaiting client connection...", color=Colors.BLUE, reprint=True)
client_ip = webserver.WaitForClient()
print(f"Client connected at {client_ip}!", color=Colors.GREEN, reprint=True)