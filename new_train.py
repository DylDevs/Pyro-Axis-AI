import importlib
import traceback
import datetime
import sys
import os

class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    DARK_GREY = "\033[90m"
    NORMAL = "\033[0m"

def timestamp():
    return Colors.DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + Colors.NORMAL

last_pint_type = "normal"
def print(message: str, color: Colors = Colors.NORMAL, end: str = "\n", reprint=False):
    if not reprint:
        if last_pint_type == "reprint":
            sys.stdout.write(f"\n{timestamp()}{color}{message}{Colors.NORMAL}{end}")
        else:
            sys.stdout.write(f"{timestamp()}{color}{message}{Colors.NORMAL}{end}")
        last_pint_type = "normal"
    else:
        sys.stdout.write(f"\r{timestamp()}{color}{message}{Colors.NORMAL}                                       ")
        last_pint_type = "reprint"

MODEL_TYPES_PATH = os.path.join(os.path.dirname(__file__), "modelTypes")

model_types = {}
files = os.listdir(MODEL_TYPES_PATH)
for filename in files:
    if not filename.endswith(".py"):
        files.remove(filename)
print(f"Loading model types (0/{len(files)})", color=Colors.BLUE, reprint=True)
for i, filename in enumerate(files):
    print(f"Loading model types ({i + 1}/{len(files)})", color=Colors.BLUE, reprint=True)
    model_import = importlib.import_module(f"modelTypes.{filename[:-3]}")
    model_types[filename[:-3]] = model_import

print(f"Loaded {len(model_types)} model types", color=Colors.GREEN, reprint=True)
print("another print")