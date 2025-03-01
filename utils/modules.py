import numpy as np
import jsonschema
import threading
import importlib
import traceback
import datetime
import inspect
import torch
import json
import time
import sys
import os

MODEL_TYPES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_types")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SUPPORTED_CONTROLLER_PBS = ["epoch", "time"]
SUPPORTED_CONTROLLER_GRAPHS = ["loss"]

class LoadExceptions:
    class MissingFile(Exception): pass
    class JSONError(Exception): pass
    class MissingVar(Exception): pass
    class HypNotList(Exception): pass
    class TypeException(Exception): pass
    class IncorrectType(Exception): pass

class ModelExceptions:
    class MissingHyperparameter(Exception): pass
    class MissingVar(Exception): pass
    class InvalidVar(Exception): pass

class ModelTypeLoader:
    def __init__(self, model_files):
        self.model_files = model_files
        self.json_schema = json.load(open(os.path.join(os.path.dirname(__file__), "schema.json"), "r"))
        self.model_types : list[Model] = []
        self.model_file_mod_times = [{"json": None, "py": None} for _ in model_files]
        self.updating = False

        self.InitialLoad()
        threading.Thread(target=self._listener_thread, daemon=True).start()

    def InitialLoad(self):
        print(f"Loading {len(self.model_files)} model " + ("architecture" if len(self.model_files) == 1 else "architectures") + "...", color=Colors.BLUE)
        for i, filename in enumerate(self.model_files):
            try:
                model = self.LoadModelType(filename)
                print(f"Loaded {model.json_data['name']} model", color=Colors.GREEN)
                self.model_types.append(model)
            except Exception as e:
                try:
                    # Load name from JSON file if available
                    data = json.load(open(os.path.join(MODEL_TYPES_PATH, filename), "r"))
                    title = data["name"]
                except:
                    title = filename.replace('_', ' ').replace('.json', '').title()
                print(f"Failed to load model {title}: {e}\n{traceback.format_exc()}", color=Colors.RED)

        if len(self.model_types) == 0:
            print("No valid models found, exiting", color=Colors.RED)
            exit(1)

        print(f"Loaded {len(self.model_types)} model " + ("architecture" if len(self.model_types) == 1 else "architectures"), color=Colors.GREEN, reprint=True)
        empty_line()

    def _listener_thread(self):
        '''Listens for changes in model definitions file to reload model type'''
        while True:
            edited = False
            self.updating = True
            for i, file in enumerate(self.model_files):
                current_json_mod_time = os.path.getmtime(os.path.join(MODEL_TYPES_PATH, file))
                current_py_mod_time = os.path.getmtime(os.path.join(MODEL_TYPES_PATH, self.model_types[i].json_data["functions_py"]))
                if self.model_file_mod_times[i]["json"] != current_json_mod_time or self.model_file_mod_times[i]["py"] != current_py_mod_time:
                    model_name = self.model_types[i].json_data["name"]
                    edited = True
                    print(f"{model_name} has been updated. Reloading...", color=Colors.BLUE, reprint=True)
                    self.model_types[i] = self.LoadModelType(file)
                    print(f"{model_name} has been reloaded.", color=Colors.GREEN, reprint=True)
                    self.model_file_mod_times[i]["json"] = current_json_mod_time
                    self.model_file_mod_times[i]["py"] = current_py_mod_time
            empty_line() if edited else None
            self.updating = False
            time.sleep(5) # Lock to 0.2FPS
    
    def LoadModelType(self, file: str):
        # Check for valid JSON file
        if not os.path.exists(os.path.join(MODEL_TYPES_PATH, file)):
            raise LoadExceptions.MissingFile(f"{file} does not exist (Check the in-app documentation for more info)")
        if not file.endswith(".json"):
            raise LoadExceptions.IncorrectType(f"{file} is not a .json file (Check the in-app documentation for more info)")
        
        # Load JSON
        self.model_file_mod_times[self.model_files.index(file)]["json"] = os.path.getmtime(os.path.join(MODEL_TYPES_PATH, file))
        json_data = json.load(open(os.path.join(MODEL_TYPES_PATH, file), "r"))

        # Use the schema to validate the model information
        try:
            jsonschema.validate(instance=json_data, schema=self.json_schema)
        except jsonschema.ValidationError as ve:
            path = "data" + "".join([f'["{p}"]' for p in ve.path])
            raise LoadExceptions.JSONError(f"JSON Validation Error at {path} - {ve.message}")
        except jsonschema.SchemaError as se:
            raise LoadExceptions.JSONError(f"Schema error - {se.message}")
        
        # Load Python file and ensure correct data
        if not os.path.exists(os.path.join(MODEL_TYPES_PATH, json_data["functions_py"])):
            raise LoadExceptions.MissingFile(f"{json_data['functions_py']} does not exist (Check the in-app documentation for more info)")
        
        try:
            path = os.path.join(MODEL_TYPES_PATH, json_data["functions_py"])
            spec = importlib.util.spec_from_file_location(json_data["functions_py"].replace(".py", ""), path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[json_data["functions_py"].replace(".py", "")] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise LoadExceptions.MissingFile(f"Failed to import {json_data['functions_py']}: {e} (Check the in-app documentation for more info)")
        self.model_file_mod_times[self.model_files.index(file)]["py"] = os.path.getmtime(os.path.join(MODEL_TYPES_PATH, json_data["functions_py"]))

        if not hasattr(module, "Model"):
            raise LoadExceptions.MissingVar(f"Model class not found in {json_data['functions_py']} (Check the in-app documentation for more info)")
        model_class = getattr(module, "Model")
        if not issubclass(model_class, ModelTemplate):
            raise LoadExceptions.MissingVar(f"Model class is not a subclass of ModelTemplate (Check the in-app documentation for more info)")

        # [function, [required_args]]
        required_methods = [[json_data["initialize_function"], []], [json_data["train_function"], []], [json_data["save_function"], []]]
        for method in required_methods:
            if not hasattr(model_class, method[0]) or not callable(getattr(model_class, method[0])):
                raise LoadExceptions.MissingVar(f"Required method {method[0]} not found in {json_data['functions_py']} Model class (Check the in-app documentation for more info)")
            args = list(inspect.signature(getattr(model_class, method[0])).parameters.keys())
            for arg in method[1]:
                if arg not in args:
                    raise LoadExceptions.MissingVar(f"{method[0]} is required to have argument {arg} (Check the in-app documentation for more info)")

        # Check JSON data (Python checks will be conducted along the way)
        for var in json_data["hyperparameters"]:
            name = var["name"]
            if isinstance(var["default"], (str, bool)):
                default_type = str if isinstance(var["default"], str) else bool
                if "min_value" in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has a min_value set (Check the in-app documentation for more info)")
                if "max_value" in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has a max_value set (Check the in-app documentation for more info)")
                if "incriment" in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has an incriment set (Check the in-app documentation for more info)")
            elif isinstance(var["default"], (int, float)):
                default_type = int if isinstance(var["default"], int) else float
                if "min_value" not in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has no min_value set (Check the in-app documentation for more info)")
                if "max_value" not in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has no max_value set (Check the in-app documentation for more info)")
                if "incriment" not in var:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has no incriment set (Check the in-app documentation for more info)")
                if type(var["incriment"]) != int and type(var["incriment"]) != float:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has an incriment that is not an int or float (Check the in-app documentation for more info)")
                if var["incriment"] <= 0:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a {default_type} but has an incriment less than or equal to 0 (Check the in-app documentation for more info)")

                min_value = float("-inf") if var["min_value"] == None else var["min_value"]
                max_value = float("inf") if var["max_value"] == None else var["max_value"]
                if min_value >= max_value:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has a min_value greater than or equal to the max_value (Check the in-app documentation for more info)")
                if var["incriment"] <= 0:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has an incriment less than or equal to 0 (Check the in-app documentation for more info)")
                if var["incriment"] > max_value - min_value:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has an incriment greater than the max_value minus the min_value (Check the in-app documentation for more info)")
                if var["default"] < min_value or var["default"] > max_value:
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has a default value outside of the range (Check the in-app documentation for more info)")
            else:
                raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has an invalid default type (Check the in-app documentation for more info)")

            if "options" in var and "special_type" not in var:
                raise LoadExceptions.TypeException(f"Hyperparameter '{name}' has options but is not a dropdown (Check the in-app documentation for more info)")

            if "special_type" in var:
                if var["special_type"] == "path" and not isinstance(var["default"], str):
                    raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a path but has no default path (Check the in-app documentation for more info)")
                elif var["special_type"] == "dropdown":
                    if "options" not in var:
                        raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a dropdown but has no options set (Check the in-app documentation for more info)")
                    if len(var["options"]) == 0:
                        raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a dropdown but has no options set (Check the in-app documentation for more info)")
                    for option in var["options"]:
                        if not isinstance(option, (str, int, float)):
                            raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a dropdown but has an option that is not a string, int, or float (Check the in-app documentation for more info)")
                    if var["default"] not in var["options"]:
                        raise LoadExceptions.TypeException(f"Hyperparameter '{name}' is a dropdown but the default value is not in the options (Check the in-app documentation for more info)")

                if "description" not in var:
                    var["description"] = None

        if not "Epochs" in [var["name"] for var in json_data["hyperparameters"]]:
            raise LoadExceptions.MissingVar(f"Hyperparameter 'Epochs' is required (Check the in-app documentation for more info)")
        if json_data["hyperparameters"][[var["name"] for var in json_data["hyperparameters"]].index("Epochs")]["incriment"] != 1:
            raise LoadExceptions.TypeException(f"Required hyperparameter 'Epochs' must have an incriment of 1 (Check the in-app documentation for more info)")

        for var in json_data["progress_bars"]:
            if var["title"].startswith("controller."):
                found = False
                for pb in SUPPORTED_CONTROLLER_PBS:
                    if var["title"] == f"controller.{pb}":
                        found = True
                        break
                if not found:
                    raise LoadExceptions.TypeException(f"Progress bar '{name}' is attempting to use a non-supported controller progress bar (Check the in-app documentation for more info)")
                else:
                    continue # Skip the rest of the checks as this is a default progress bar

            # `current` and `total` values will be checked each time the UI is updated as they may not exist yet

            if "description" not in var:
                var["description"] = None

            if "{0}" not in var["progress_text"] or "{1}" not in var["progress_text"]:
                raise LoadExceptions.TypeException( f"Progress bar '{name}' progress_text must contain '{{0}}' and '{{1}}' (Check the in-app documentation for more info)")

        for var in json_data["graphs"]:
            if var["title"].startswith("controller."):
                found = False
                for graph in SUPPORTED_CONTROLLER_GRAPHS:
                    if var["title"] == f"controller.{graph}":
                        found = True
                        break
                if not found:
                    raise LoadExceptions.TypeException(f"Graph '{var['title']}' is attempting to use a non-supported controller graph (Check the in-app documentation for more info)")
                else:
                    continue # Skip the rest of the checks as this is a default graph

            if "description" not in var:
                var["description"] = None

            # 'values' will be checked each time the UI is updated as they may not exist yet

        for var in json_data["info_dropdowns"]:
            for data in var["data"]:
                if "description" not in data:
                    data["description"] = None
            
                # `value` will be checked each time the UI is updated as they may not exist yet

            if "description" not in var:
                var["description"] = None

        return Model(json_data, model_class)
    
    def GetModelTypes(self):
        return self.model_types

class Colors:
    '''Colors for printing in the console'''
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    DARK_GREY = "\033[90m"
    NORMAL = "\033[0m"

def timestamp(caller = None):
    caller = f" | {caller}.py" if caller != None else ""
    return Colors.DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}{caller}] " + Colors.NORMAL
    # [00:00:00 | file.py] 

last_print_type = "normal"
def print(message: str, color: Colors = Colors.NORMAL, end: str = "\n", reprint : bool = False, show_timestamp: bool = True):
    '''Custom print function for logging purposes'''
    global last_print_type
    if show_timestamp:
        start = f"{timestamp(os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0])}"
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
    '''Empty line in the console'''
    print("", show_timestamp=False)
    
def reset_reprint():
    '''Allows for reprints in a row'''
    print("", end="", show_timestamp=False)

class ModelTemplate:
    def __init__(self, json_data: dict) -> None:
        """
        Parent class for all Python model definitions
        """
        self.json_data = json_data
        self.model : torch.nn.Module = None

        # Get the initialize, train, and save functions from their name
        self.initialize_function = getattr(self, json_data["initialize_function"])
        self.train_function = getattr(self, json_data["train_function"])
        self.save_function = getattr(self, json_data["save_function"])
        self.after_train_function = getattr(self, json_data["after_train_function"]) if "after_train_function" in json_data else None

        # Paths
        self.data_path = DATA_PATH
        self.models_path = MODEL_PATH

        # Losses
        self.train_losses = []
        self.val_losses = []

        # For error handling
        self.error = None
        self.traceback = None

    def GetHyp(self, name: str):
        '''Get a hyperparameter value by it's name'''
        for hyp in self.json_data["hyperparameters"]:
            if hyp["name"] == name:
                return hyp["value"]

        self.RaiseException(ModelExceptions.MissingHyperparameter(f"Hyperparameter '{name}' does not exist"))

    def __initialize__(self):
        try:
            self.initialize_function()
        except Exception as e:
            self.RaiseException(e)

    def __train__(self):
        try:
            self.train_function()
        except Exception as e:
            self.RaiseException(e)

    def __after_train__(self, controller):
        if self.after_train_function:
            try:
                self.after_train_function(controller)
            except Exception as e:
                self.RaiseException(e)

    def __save__(self, model: torch.nn.Module):
        try:
            self.save_function(model)
        except Exception as e:
            self.RaiseException(e)

    def RaiseException(self, e : Exception, traceback_str : str = None):
        '''Raise an exception (stops training process and shows error in UI)'''
        self.error = e
        self.traceback = "".join(traceback.format_exception(type(e), e, e.__traceback__)) if not traceback_str else traceback_str

class Model:
    json_data: dict
    model_class: ModelTemplate

    def __init__(self, json_data: dict, model_class: ModelTemplate) -> None:
        self.json_data = json_data
        self.model_class = model_class

    def FrontendData(self):
         return {
            "name": self.json_data["name"],
            "description": self.json_data["description"],
            "data_type": self.json_data["data_type"],
            "hyperparameters": self.json_data["hyperparameters"],
        }

id_tracker = 0
def GetID():
    '''Get ID for new model'''
    global id_tracker
    id_tracker += 1
    return id_tracker

class TrainingController:
    def __init__(self, model: Model) -> None:
        '''Creates a new training session'''
        self.json_data = model.json_data
        self.model_class = model.model_class
        self.id = GetID()
        self.train_thread = threading.Thread(target=self._train_thread)

        self.epoch = 0
        self.status = "Initializing"

        # For the webserver to send to frontend for display
        self.error_str = None 
        self.error_tb = None

        self.times_per_epoch = []
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_model = None
        self.best_epoch = 0

        self.start_time = 0
        self.end_time = 0

        self.model : ModelTemplate = model.model_class(self.json_data) # Initialize the model's controller (not the model itself)
        self.model.__initialize__()
        if self.model.error: # Error in model initialization
            self.status = "Error"
            self.ErrorHandler(self.model.error, self.model.traceback)
            return
        
        # Add hyperparameters to the info dropdowns list
        self.json_data["info_dropdowns"].append({
            "title": "Hyperparameters",
            "description": "Hyperparameters selected for this model",
            "data": [{ "title": hp["name"], "value": f"hyperparameter.{hp['name']}" } for hp in self.json_data["hyperparameters"]],
        })
    
    def Train(self):
        self.start_time = time.time()
        self.train_thread.start()

    def _train_thread(self):
        try:
            for epoch in range(self.model.GetHyp("Epochs")):
                epoch_start_time = time.time()
                self.status = "Initializing" if epoch == 0 else "Training"
                self.epoch = epoch

                self.model.__train__()
                if self.model.error: # Error in model training
                    self.status = "Error"
                    self.ErrorHandler(self.model.error, self.model.traceback)
                    return
                
                if self.model.val_losses[-1] < self.best_val_loss or self.best_model is None:
                    self.best_val_loss = self.model.val_losses[-1]
                    self.best_epoch = self.epoch
                    self.best_model : torch.nn.Module = self.model.model

                # Gves access to controller values (self) for optional post training operatipns
                self.model.__after_train__(self) 
                if self.model.error: # Error in model post training
                    self.status = "Error"
                    self.ErrorHandler(self.model.error, self.model.traceback)
                    return
                
                self.times_per_epoch.append(time.time() - epoch_start_time)
        except Exception as e:
            self.status = "Error"
            self.ErrorHandler(e, traceback.format_exc())
            return

        self.model.__save__(self.best_model)
        if self.model.error: # Error in model saving
            self.status = "Error"
            self.ErrorHandler(self.model.error, self.model.traceback)
            return
        self.status = "Finished"

    def ErrorHandler(self, error, traceback_str = None):
        prev_status = self.status
        self.status = "Error"

        error_root = f"Initialize" if prev_status == "Initializing" else f"Epoch {self.epoch}"
        self.error_str = f"Error in {self.json_data['name']} model {self.id} ({error_root}): {error}"
        self.error_tb = traceback_str if traceback_str else "Traceback is not available"
        print(f"{self.error_str}\n{self.error_tb}", color=Colors.RED)

    def FormatTime(self, seconds: int) -> str:
        '''String format time (ex. 130 seconds, )'''
        if seconds <= 0:
            return "0 seconds"
        
        days = seconds // (24 * 3600)
        seconds %= 24 * 3600
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        parts = []
        if days > 0:
            parts.append(f"{days:.0f} days")
        if hours > 0:
            parts.append(f"{days:.0f} hours")
        if minutes > 0:
            parts.append(f"{minutes:.0f} minutes")
        if seconds > 0 or not parts:
            parts.append(f"{seconds:.0f} seconds")
        
        return ", ".join(parts)

    def EstTimeRemaining(self) -> float | int:
        if self.status != "Training": return 0
        if len(self.times_per_epoch) == 0: return 0
        return round((np.mean(self.times_per_epoch) * self.model.GetHyp("Epochs")) - (time.time() - self.start_time), 2)

    def PresetProgressBar(self, name: str):
        if name == "epoch":
            return {
                "title": "Epochs",
                "description": "Completed training iterations out of total", 
                "current": self.epoch,
                "total": self.model.GetHyp("Epochs"),
                "progress_text": f"Epoch {self.epoch} out of {self.model.GetHyp('Epochs')}",
            }
        elif name == "time":
            elapsed = round(time.time() - self.start_time, 2)
            remaining = self.EstTimeRemaining()
            return {
                "title": "Time",
                "description": "Time elapsed since start of training out of total completion time", 
                "current": elapsed,
                "total": elapsed + remaining,
                "progress_text": f"{self.FormatTime(elapsed)} out of {self.FormatTime(elapsed + remaining)}",
            }
        else:
            self.model.RaiseException(ModelExceptions.InvalidVar(f"Invalid default progress bar type: {name}"))
            self.ErrorHandler(self.model.error, self.model.traceback)
            return None

    def PresetGraph(self, name: str):
        if name == "loss":
            return {
                "title": "Train & Val Loss",
                "tooltip": "Loss of the model on the validation set",
                "lines": [
                    {
                        "title": "Train Loss",
                        "color": "#ff0000",
                        "values": self.model.train_losses,
                    },
                    {
                        "title": "Val Loss",
                        "color": "#00ff00",
                        "values": self.model.val_losses,
                    },
                ],
            }
        else:
            self.model.RaiseException(ModelExceptions.InvalidVar(f"Invalid default graph type: {name}"))
            self.ErrorHandler(self.model.error, self.model.traceback)
            return None
    
    def GetModelAttr(self, name: str, value_type: str):
        if name.startswith("hyperparameter."):
            name = name.replace("hyperparameter.", "")
            value = self.model.GetHyp(name)
            if value is None:
                self.ErrorHandler(self.model.error, self.model.traceback)
                return None
            return value
        try:
            return getattr(self.model, name)
        except:
            self.model.RaiseException(ModelExceptions.MissingVar(f"Model attribute {name} does not exist attempting to get {value_type}"))
            self.ErrorHandler(self.model.error, self.model.traceback)
            return None

    def ConstructData(self):
        try:
            # Construct progress bar data
            progress_bars = []
            for pb in self.json_data["progress_bars"]:
                if pb["title"].startswith("controller."):
                    bar = self.PresetProgressBar(pb["title"].replace("controller.", ""))
                    if bar is None: return None
                    progress_bars.append(bar)
                else:
                    current = self.GetModelAttr(pb["current"], f"{pb['title']} progress bar current")
                    if current is None: return None
                    total = self.GetModelAttr(pb["total"], f"{pb['title']} progress bar total")
                    if total is None: return None

                    if "special_type" in pb:
                        if pb["special_type"] == "time":
                            text_current = self.FormatTime(current)
                            text_total = self.FormatTime(total)
                        else:
                            text_current = current
                            text_total = total
                    else:
                        text_current = current
                        text_total = total

                    progress_bars.append({
                        "title": pb["title"],
                        "tooltip": pb["description"], # NoneType cases are handled in loader
                        "current": current,
                        "total": current + total,
                        "progress_text": pb["progress_text"].replace("{0}", str(text_current)).replace("{1}", str(text_total)),
                    })

            # Construct graph data (not implemented yet)
            graphs = []
            for graph in self.json_data["graphs"]:
                if graph["title"].startswith("controller."):
                    graph = self.PresetGraph(graph["title"].replace("controller.", ""))
                    if graph is None: return None
                    graphs.append(graph)
                else:
                    lines = []
                    for line in graph["data"]:
                        values = self.GetModelAttr(line["values"], f"{line['title']} graph values")
                        if values is None: return None
                        lines.append({
                            "title": line["title"],
                            "color": line["color"],
                            "values": values,
                        })
                    graphs.append({
                        "title": graph["title"],
                        "tooltip": graph["description"],
                        "lines": lines,
                    })

            # Construct dropdown data
            dropdowns = []
            for dropdown in self.json_data["info_dropdowns"]:
                data = []
                for item in dropdown["data"]:
                    value = self.GetModelAttr(item["value"], f"{item['title']} dropdown value")
                    if value is None: return None
                    data.append({
                        "title": item["title"],
                        "value": str(value),
                        "tooltip": item["description"] if "description" in item else None
                    })
                dropdowns.append({
                    "title": dropdown["title"],
                    "tooltip": dropdown["description"],
                    "data": data,
                })

            return {
                "progress_bars": progress_bars,
                "graphs": graphs,
                "dropdowns": dropdowns
            }
        except Exception as e:
            self.model.RaiseException(e)
            self.ErrorHandler(self.model.error, self.model.traceback)
            raise e

    def FrontendData(self):
        if self.error_str: return None
        try:
            ui_data = self.ConstructData()
            if ui_data is None: return None # Error occurred while trying to construct data
            data = {
                "type": self.json_data["name"],
                "data_type": self.json_data["data_type"],
                "status": self.status,
                "epoch": self.epoch,
                "epochs": self.model.GetHyp("Epochs"),
                "estimated_time": self.FormatTime(self.EstTimeRemaining()),
                "progress_bars": ui_data["progress_bars"],
                "graphs": ui_data["graphs"],
                "dropdowns": ui_data["dropdowns"],
            }
            return data
        except Exception as e:
            self.error_str = str(e)
            self.error_tb = traceback.format_exc()
            if not self.error_tb: self.error_tb = "Traceback is not available"
            return None