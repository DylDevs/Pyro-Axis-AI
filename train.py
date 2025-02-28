def run():
    # Import custom print function and colors
    from utils.modules import print, Colors

    # Flags for starting processes
    start_window = True
    start_frontend = True
    start_webserver = True
    debug = False

    init_line = "Initializing..." if not debug else "Initializing in DEBUG mode..."
    print(f"Welcome to Pyro Axis AI Training Dashboard. {init_line}", color=Colors.BLUE)
    print("Importing libraries...", color=Colors.BLUE, reprint=True)

    # Import libraries and Pyro Axis modules
    import shutil
    import torch
    import os
    from utils import modules
    import utils.webserver as webserver
    import utils.window as window

    print("Imported libraries", color=Colors.GREEN, reprint=True)
    modules.reset_reprint()
    if torch.cuda.is_available():
        print("Using GPU for training (by default)", color=Colors.GREEN)
    else:
        print("GPU not detected, using CPU for training (by default)", color=Colors.YELLOW)

    modules.empty_line()

    # Path variables
    PATH = os.path.dirname(__file__)
    MODEL_TYPES_PATH = PATH + "/model_types"
    if os.path.exists(os.path.join(MODEL_TYPES_PATH, "__pycache__")):
        shutil.rmtree(os.path.join(MODEL_TYPES_PATH, "__pycache__"))

    files = os.listdir(MODEL_TYPES_PATH)
    for filename in files:
        if not filename.endswith(".json"):
            files.remove(filename) # Remove from the local list (not directory)
            
    if len(files) == 0:
        print("No models found, exiting", color=Colors.RED)
        exit(1)

    # Initiate model loader to load model types and keep them updated
    model_loader = modules.ModelTypeLoader(files)
    webserver.SetModelLoader(model_loader)

    # Start the frontend and backend
    frontend_url, backend_url = webserver.run(frontend=start_frontend, backend=start_webserver, debug=debug)
    if frontend_url: print(f"Frontend URL: {frontend_url} (localhost:3000)", color=Colors.GREEN)
    if backend_url: print(f"Backend URL: {backend_url} (localhost:8000)", color=Colors.GREEN)
    print("Awaiting client connection...", color=Colors.BLUE, reprint=True)

    # Start the window
    window.run(start_frontend) if start_window else None

    # Wait for client connection
    client_ip = webserver.WaitForClient()
    print(f"Client connected at {client_ip}!\n", color=Colors.GREEN, reprint=True)

# Only run if this file is the main file
if __name__ == '__main__':
    run()