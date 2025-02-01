from utils.modules import print, Colors # Edited print function with color and repri
from datetime import datetime # Used for working with dates
import multiprocessing
import http.server
import socketserver
import requests
import time
import os # Used for file management

RETYPE_MODULE = os.path.expanduser("~") + "/AppData/Roaming/npm/node_modules/retypeapp"
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
BUILD_DIR = os.path.join(DOCS_DIR, "output")
DOCS_AVAILABLE = True
DOCS_RUNNING = False
HTTPD = None
DOCS_HAVE_STARTED = False

def get_most_recent_edit():
    most_recent_time = datetime.min

    for root, _, files in os.walk(DOCS_DIR):
        if root.startswith(BUILD_DIR):
            continue
        
        for file in files:
            file_path = os.path.join(root, file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if mod_time > most_recent_time:
                most_recent_time = mod_time
    
    return most_recent_time

def check_retype():
    if not os.path.exists(RETYPE_MODULE):
        print("Retype is not installed. Installing...", color=Colors.BLUE, reprint=True)
        status = os.system("npm install -g retypeapp > /dev/null 2>&1")
        if status != 0:
            print("Failed to install Retype, in-app docs are unavailable.", color=Colors.RED, reprint=True)
            global DOCS_AVAILABLE
            DOCS_AVAILABLE = False
        else:
            print("Retype has been installed.", color=Colors.GREEN, reprint=True)

def build_docs():
    print("Building docs...", color=Colors.BLUE, reprint=True)
    check_retype()
    status = os.system(f"cd {DOCS_DIR} && retype build > /dev/null 2>&1")
    if status != 0:
        print("Failed to build docs, in-app docs are unavailable.", color=Colors.RED, reprint=True)
        global DOCS_AVAILABLE
        DOCS_AVAILABLE = False
    else:
        print("Docs have been built.", color=Colors.GREEN, reprint=True)

def run_docs():
    global HTTPD
    with socketserver.TCPServer(("", 5000), http.server.SimpleHTTPRequestHandler) as HTTPD:
        HTTPD.serve_forever()

def stop_docs():
    global HTTPD
    if HTTPD is not None:
        HTTPD.shutdown()
        HTTPD.server_close()
    HTTPD = None

def start_docs():
    if not os.path.exists(BUILD_DIR):
        build_docs()
    else:
        most_recent_edit = get_most_recent_edit()
        last_build_time = datetime.fromtimestamp(os.path.getmtime(BUILD_DIR))
        if last_build_time < most_recent_edit:
            build_docs()

    if DOCS_AVAILABLE and not DOCS_RUNNING:
        print("Starting docs...", color=Colors.BLUE, reprint=True)
        DOCS_RUNNING = True
        p = multiprocessing.Process(target=run_docs, daemon=True)
        p.start()

        # Wait until the server is ready
        RETRY_INTERVAL = 0.2
        while not DOCS_HAVE_STARTED:
            try:
                response = requests.get(
                    f'http://localhost:5000',
                    timeout=1
                )
                if response.ok:
                    DOCS_HAVE_STARTED = True
                    break
            except:
                pass  # Handle timeout
            
            time.sleep(RETRY_INTERVAL)

        print("Docs have been started.", color=Colors.GREEN, reprint=True)

def run():
    p = multiprocessing.Process(target=start_docs, daemon=True)
    p.start()

def stop():
    stop_docs()