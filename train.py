import logging as logger
import datetime
import sys

debug_printing = True

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
import importlib
import threading
import torch
import time
import os

# Import the webserver and modules for models
from modelTypes import modules
import webserver

print("Imported libraries", color=Colors.GREEN, reprint=True)
reset_reprint()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for training (by default)", color=Colors.GREEN)
else:
    device = torch.device("cpu")
    print("GPU not detected, using CPU for training (by default)", color=Colors.YELLOW)

empty_line()

# Path vars
PATH = os.path.dirname(__file__)
MODEL_TYPES_PATH = PATH + "/modelTypes"

# Changable vars
start_frontend = False
start_webserver = True
debug = False

# Constants
model_types = {}
files = os.listdir(MODEL_TYPES_PATH)
for filename in files:
    if not filename.endswith(".py") or filename == "modules.py":
        files.remove(filename)

if len(files) == 0:
    print("No models found, exiting", color=Colors.RED)
    exit(1)
        
print(f"Loading {len(files)} model " + ("architecture" if len(files) == 1 else "architectures") + "...", color=Colors.BLUE, reprint=True)
for i, filename in enumerate(files):
    model_import = importlib.import_module(f"modelTypes.{filename[:-3]}")

    # Check for required variables
    if "name" not in model_import.__dict__:
        print(f"Model {filename[:-3]} is missing the name variable (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "description" not in model_import.__dict__:
        print(f"Model {filename[:-3]} is missing the description variable (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "hyperparameters" not in model_import.__dict__:
        print(f"Model {filename[:-3]} is missing the hyperparameters variable (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "data_type" not in model_import.__dict__:
        print(f"Model {filename[:-3]} is missing the data_type variable (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if not isinstance(model_import.hyperparameters, list):
        print(f"Model {filename[:-3]} hyperparameters is not a list (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue

    required_hyperparameters = ["epochs"]
    found = [False] * len(required_hyperparameters)
    exit_loop = False
    for hyperparameter in model_import.hyperparameters:
        if not isinstance(hyperparameter, modules.RequiredHyperparameter):
            print(f"Model {filename[:-3]} hyperparameter {hyperparameter.name} is not a RequiredHyperparameter object (Check the in-app docs for more info), skipping", color=Colors.RED)
            exit_loop = True
            break
        if hyperparameter.name in required_hyperparameters:
            found[required_hyperparameters.index(hyperparameter.name)] = True
    if exit_loop:
        continue
    if not all(found):
        print(f"Model {filename[:-3]} is missing the following required hyperparameters: {', '.join(required_hyperparameters[i] for i, x in enumerate(found) if not x)} (Check the in-app docs for more info), skipping", color=Colors.RED)

    if not isinstance(model_import.name, str):
        print(f"Model {filename[:-3]} name is not a string (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if not isinstance(model_import.description, str):
        print(f"Model {filename[:-3]} description is not a string (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if not isinstance(model_import.data_type, str):
        print(f"Model {filename[:-3]} data_type is not a string (Should be one of 'text', 'image', 'audio', 'other'), skipping", color=Colors.RED)
        continue
    if model_import.data_type not in ["text", "image", "audio", "other"]:
        print(f"Model {filename[:-3]} data_type is not a valid data type (Should be one of 'text', 'image', 'audio', 'other'), skipping", color=Colors.RED)
        continue

    # Check for required classes and functions
    if "Model" not in model_import.__dict__:
        print(f"Model {filename[:-3]} is missing the Model class (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue

    model = model_import.Model
    if not issubclass(model, modules.ModelTemplate):
        print(f"Model {filename[:-3]} Model class is not a subclass of ModelTemplate (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "Setup" not in model.__dict__:
        print(f"Model {filename[:-3]} is missing the Setup function (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "Train" not in model.__dict__:
        print(f"Model {filename[:-3]} is missing the Train function (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue
    if "Save" not in model.__dict__:
        print(f"Model {filename[:-3]} is missing the Save function (Check the in-app docs for more info), skipping", color=Colors.RED)
        continue

    model = modules.Model(
        name = filename[:-3],
        display_name = model_import.name,
        description = model_import.description,
        data_type = model_import.data_type,
        model_class = model,
        required_hyperparameters = model_import.hyperparameters,
        hyperparameters = [modules.Hyperparameter(hyperparameter.name, hyperparameter.default) for hyperparameter in model_import.hyperparameters]
    )
    
    model_types[filename[:-3]] = model

if len(model_types) == 0:
    print("No valid models found, exiting", color=Colors.RED)
    exit(1)

print(f"Loaded {len(model_types)} model " + ("architecture" if len(model_types) == 1 else "architectures"), color=Colors.GREEN, reprint=True)
empty_line()

frontend_url, backend_url = webserver.run(frontend=start_frontend, backend=start_webserver, debug=debug)
if frontend_url:
    print(f"Frontend URL: {frontend_url} (localhost:3000)", color=Colors.GREEN)
if backend_url:
    print(f"Backend URL: {backend_url} (localhost:8000)", color=Colors.GREEN)

print("Awaiting client connection...", color=Colors.BLUE, reprint=True)

# Pass the training session class since we cant inport it in the webserver (circular import)
client_ip = webserver.WaitForClient()
print(f"Client connected at {client_ip}!", color=Colors.GREEN, reprint=True)