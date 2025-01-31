def run():
    from modelTypes.modules import print # Edited print function with color and reprint
    from modelTypes import modules # Modules and utilities for training models
    import utils.webserver as webserver # Local webserver
    import utils.window as window # Creates a window for the frontend

    start_frontend = True
    start_webserver = True
    debug = False

    print("Importing libraries...", color=modules.Colors.BLUE, reprint=True)

    import traceback # Used for debugging
    import shutil # Used for deleting __pycache__ tree
    import torch # Used for training and CUDA detection
    import os # Used for file management

    print("Imported libraries", color=modules.Colors.GREEN, reprint=True)
    modules.reset_reprint()
    if torch.cuda.is_available():
        print("Using GPU for training (by default)", color=modules.Colors.GREEN)
    else:
        print("GPU not detected, using CPU for training (by default)", color=modules.Colors.YELLOW)

    modules.empty_line()

    # Path vars
    PATH = os.path.dirname(__file__)
    MODEL_TYPES_PATH = PATH + "/modelTypes"
    if os.path.exists(os.path.join(MODEL_TYPES_PATH, "__pycache__")):
        shutil.rmtree(os.path.join(MODEL_TYPES_PATH, "__pycache__"))

    # Constants
    excluded_file_names = ["modules.py", "__init__.py"]
    files = os.listdir(MODEL_TYPES_PATH)
    for filename in files:
        if not filename.endswith(".py") or filename in excluded_file_names:
            files.remove(filename) # Remove from the local list (not directory)
            
    if len(files) == 0:
        print("No models found, exiting", color=modules.Colors.RED)
        exit(1)

    # Initiate model loader to load model types and keep them updated
    model_loader = modules.ModelTypeLoader(files)

    webserver.SetModelLoader(model_loader)
    frontend_url, backend_url = webserver.run(frontend=start_frontend, backend=start_webserver, debug=debug)
    if frontend_url:
        print(f"Frontend URL: {frontend_url} (localhost:3000)", color=modules.Colors.GREEN)
    if backend_url:
        print(f"Backend URL: {backend_url} (localhost:8000)", color=modules.Colors.GREEN)

    print("Awaiting client connection...", color=modules.Colors.BLUE, reprint=True)

    try:
        window.start_webpage(frontend_started=start_frontend)
    except Exception as e:
        traceback.print_exc()

    client_ip = webserver.WaitForClient()
    print(f"Client connected at {client_ip}!", color=modules.Colors.GREEN, reprint=True)

if __name__ == '__main__':
    run()