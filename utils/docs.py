from utils.modules import print, Colors, empty_line
from datetime import datetime
import multiprocessing
import socketserver
import http.server
import requests
import shutil
import time
import os

RETYPE_MODULE = os.path.expanduser("~") + "/AppData/Roaming/npm/node_modules/retypeapp"
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
BUILD_DIR = os.path.join(DOCS_DIR, "output")
DOCS_RUNNING = False
HTTPD = None

def check_retype():
    if not os.path.exists(RETYPE_MODULE):
        print("Retype is not installed. Installing...", color=Colors.BLUE)
        status = os.system("npm install -g retypeapp > nul 2>&1")
        if status != 0:
            print("Failed to install Retype, in-app docs are unavailable.", color=Colors.RED, reprint=True)
        else:
            print("Retype has been installed.", color=Colors.GREEN, reprint=True)
    
        return status == 0
    return True

def build_docs():
    print("Building docs...", color=Colors.BLUE)
    if os.path.exists(BUILD_DIR): shutil.rmtree(BUILD_DIR)
    status = os.system(f"cd {DOCS_DIR} && retype build > nul 2>&1")
    if status != 0:
        print("Failed to build documentation, in-app docs are unavailable.", color=Colors.RED, reprint=True)
    else:
        print("Built documentation", color=Colors.GREEN, reprint=True)

    return status == 0

def run_docs():
    global HTTPD
    os.chdir(BUILD_DIR)
    with socketserver.TCPServer(("", 5000), http.server.SimpleHTTPRequestHandler) as HTTPD:
        HTTPD.serve_forever()

def stop_docs():
    global HTTPD
    if HTTPD is not None:
        HTTPD.shutdown()
        HTTPD.server_close()
    HTTPD = None

def run():
    global DOCS_RUNNING
    empty_line() # Print empty line for cleaner output
    if check_retype() == False: return False
    if build_docs() == False: return False

    if not DOCS_RUNNING:
        print("Starting docs...", color=Colors.BLUE, reprint=True)
        DOCS_RUNNING = True
        p = multiprocessing.Process(target=run_docs, daemon=True)
        p.start()

        # Wait until the server is ready
        RETRY_INTERVAL = 0.2
        while True: # Loop until documentation sever is ready
            try:
                response = requests.get(
                    f'http://localhost:5000',
                    timeout=1
                )
                if response.ok:
                    break
            except:
                pass  # Handle timeout
            
            time.sleep(RETRY_INTERVAL)

        print("Docs have been started.", color=Colors.GREEN, reprint=True)
        return True

def stop():
    stop_docs()