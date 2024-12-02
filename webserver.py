from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import threading
import uvicorn
import logging
import socket
import random
import os

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
    return IP, frontend_url, webserver_url

app = FastAPI(title="Torch AI Training", 
    description="Webservers to handle connection between Torch train code and client",
    version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    hyperparameters: dict

training_sessions = []

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

@app.post("/train/{model_type}")
async def train(model_type: str, request: TrainRequest):
    global training_started
    hyperparameters = request.hyperparameters
    training_started = True

    session = TrainingSession(model_type, hyperparameters)
    setup_status = session.setup()
    if setup_status != "ok":
        if isinstance(setup_status, Exception):
            setup_status = str(setup_status)

        return {"status": "error", "traceback": setup_status}
    
    session.start_training()
    training_sessions.append(session)
    return {"status": "ok"}

@app.get("/models")
async def get_models():
    # Return list of models and their status
    data = []
    for model in training_sessions:
        data.append(model.GetModelStatus())

    return {"status": "ok", "training_data": data}

@app.get("/models/{model_id}/stop_save")
async def stop_and_save_model(model_id):
    # Stop the training of a specific model and save it
    return {"status": "ok"}

@app.get("/models/{model_id}/stop")
async def stop_model(model_id):
    # Stop the training of a specific model and do not save it
    return {"status": "ok"}

@app.get("/models/stop_all")
async def stop_all_models():
    # Stop training of all models and do not save them
    return {"status": "ok"}

# TODO: Impliment saved_models endpoint on frontend

@app.get("/saved_models/")
async def get_saved_models():
    # Return list of saved models and their parameters
    return {"status": "ok"}

@app.get("/saved_models/{saved_model_id}/delete")
async def delete_model(saved_model_id):
    # Delete a saved model
    return {"status": "ok"}

@app.get("/saved_models/{saved_model_id}")
async def get_saved_model(saved_model_id):
    # Return status of a specific saved model
    return {"status": "ok"}

@app.get("/shutdown")
async def shutdown():
    # Shutdown the training server
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

# Need to get the training session class as an arg due to circular import issues
def WaitForClient(training_session):
    global client_connected, TrainingSession
    TrainingSession = training_session
    while not client_connected:
        pass
    return client_ip