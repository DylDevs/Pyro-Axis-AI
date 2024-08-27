from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import socket
import threading
import uvicorn
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

app = FastAPI(title="PyTorch AI Training", 
    description="Webservers to handle connection between PyTorch train code and client",
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

test_data = [{"status": "Training",
    "type": "Object Detection",
    "exception": None,
    "epoch": 1,
    "epoch_start_time": 0.12345,
    "training_loss": 0.12345678910,
    "val_loss": 0.10987654321,
    "elapsed": 1234.567,
    "time_per_epoch": 12.34}, 
    {"status": "Queued",
    "type": "Image Classification",
    "exception": None,
    "epoch": 0,
    "epoch_start_time": 0,
    "training_loss": 0,
    "val_loss": 0,
    "elapsed": 0,
    "time_per_epoch": 0},
    {"status": "Finished",
    "type": "Language Classification",
    "stop_type": "manual",
    "best_epoch": 80,
    "epoch": 100,
    "bext_training_loss": 0,
    "best_val_loss": 0,
    "training_start_time": 0.12345,
    "training_end_time": 1.2345,
    "elapsed": 1234.56,
    "training_dataset_accuracy": 0.9,
    "val_dataset_accuracy": 0.8}]

@app.get("/")
async def root():
    IP, _, webserver_url = GetWebData()
    return {"status": "ok", "url": webserver_url, "ip": IP}

@app.post("/train/{type}")
async def train(request: TrainRequest):
    hyperparameters = request.hyperparameters
    print(hyperparameters)
    # Process hyperparameters and add new model to queue
    return {"status": "ok"}

@app.get("/models")
async def get_models():
    # Return list of models and their status
    for i in range(len(test_data)):
        test_data[i]["training_loss"] = random.uniform(0, 3)
        test_data[i]["val_loss"] = random.uniform(0, 3)
    return {"status": "ok", "training_data": test_data}

@app.get("/models/{model_id}")
async def get_model(model_id):
    # Return status of a specific model
    return {"status": "ok"}

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

run_frontend = True
debug = False
if not debug:
    log_level = "error"
else:
    log_level = "debug"

def start_backend():
    IP, frontend_url, webserver_url = GetWebData()
    print(f"Webserver URL: {webserver_url}")
    print(f"Frontend URL: {frontend_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=log_level)

def start_frontend():
    os.system("cd ui && npm run dev")

threading.Thread(target=start_backend).start()
if run_frontend:
    threading.Thread(target=start_frontend).start()
