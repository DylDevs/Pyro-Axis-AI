from fastapi.middleware.cors import CORSMiddleware # FastAPI Middleware
from fastapi import FastAPI, Request, Body # FastAPI Framework and utilities
from typing import Any # Used for type hinting
import subprocess # Used for running commands in the terminal
import threading # Used for running functions asynchronously
import uvicorn # Used for running FastAPI
import socket # Used for getting local IP
import time # Used for timing
import json # Used for parsing JSON
import os # Used for file management

import utils.modules as modules # Modules and utilities for training models
from utils.modules import print # Edited print function with color and reprint
import utils.docs as docs # Start in-app docs

CACHE_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "cache.json")

training_controllers : list[modules.TrainingController] = []
model_loader : modules.ModelTypeLoader = None
def SetModelLoader(loader : modules.ModelTypeLoader):
    global model_loader
    model_loader = loader

def GetWebData():
    try:
        sockets = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.connect(("8.8.8.8", 80))
        IP = sockets.getsockname()[0]
        sockets.close()
    except:
        IP = "localhost"

    frontend_url = f"http://{IP}:3000"
    webserver_url = f"http://{IP}:8000"

    with open(CACHE_JSON, "r") as f:
        cache_content = json.load(f)
        cache_content["webserver_url"] = webserver_url
        f.close()

    with open(CACHE_JSON, "w") as f:
        json.dump(cache_content, f)
        f.close()

    return IP, frontend_url, webserver_url

app = FastAPI(title="Pyro Axis AI Training", 
    description="Webservers to handle connection between Torch train code and client",
    version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_connected = False
client_ip = None
training_started = False
@app.get("/")
async def root(request: Request):
    global client_connected, client_ip
    client_ip = request.client.host
    client_connected = True
    IP, _, webserver_url = GetWebData()
    return {"status": "ok", "url": webserver_url, "ip": IP}

@app.get("/models")
async def models():
    global model_loader
    model_types = model_loader.GetModelTypes()

    model_types_dict = []
    for model_type in model_types:
        model_types_dict.append(model_type.FrontendData())

    return {"status": "ok", "models": model_types_dict}

@app.post("/train")
def train(data: Any = Body(...)):
    global model_loader, training_controllers

    model_types = model_loader.GetModelTypes()
    parent_model : modules.Model = model_types[data["model_index"]]

    new_json_data = parent_model.json_data
    new_json_data["hyperparameters"] = data["hyperparameters"] # [{"name": "name", "value": "value"}, ...]
    new_model = modules.Model(new_json_data, parent_model.model_class)

    print(f"Request to train {new_model.json_data['name']} received", color=modules.Colors.BLUE)
    training_controller = modules.TrainingController(new_model)
    if training_controller.status == "Error": return {"status": "error", "error": training_controller.error_tb}

    training_controller.Train() # Starts training thread
    training_controllers.append(training_controller)

    return {"status": "ok"}

@app.get("/status")
def status():
    global training_controllers
    data = []
    for training_controller in training_controllers:
        frontend_data = training_controller.FrontendData()
        if frontend_data is None:
            training_controllers.remove(training_controller)
            return {"status": "error", "error": training_controller.error_str, "traceback": training_controller.error_tb}
        data.append(training_controller.FrontendData())
    
    return {"status": "ok", "data": data}

@app.get("/docs/start")
def start_docs():
    status = docs.run()
    if not status: return {"status": "error"}
    else: return {"status": "ok"}

@app.get("/docs/stop")
def stop_docs():
    docs.stop()
    return {"status": "ok"}

def start_backend(debug):
    if debug:
        log_level = "debug"
    else:
        log_level = "error"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=log_level)

def start_frontend():
    # Redirect both stdout and stderr to /dev/null
    subprocess.run("cd ui && npm run dev", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run(frontend = True, backend = True, debug = False):
    IP, frontend_url, webserver_url = GetWebData()
    if backend:
        threading.Thread(target=start_backend, args=(debug,)).start()
    else:
        webserver_url = None
    if frontend:
        threading.Thread(target=start_frontend).start()
    else:
        frontend_url = None
    return frontend_url, webserver_url

def WaitForClient():
    global client_connected
    while not client_connected:
        pass
    return client_ip