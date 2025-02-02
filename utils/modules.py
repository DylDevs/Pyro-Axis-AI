from typing import Literal
import numpy as np
import threading
import importlib
import datetime
import torch
import time
import sys
import os

class LoadExceptions:
    class MissingVar(Exception): pass
    class HypNotList(Exception): pass
    class TypeException(Exception): pass
    class IncorrectType(Exception): pass

class ModelTypeLoader:
    def __init__(self, model_files):
        self.model_files = model_files
        self.model_types : list[Model] = []
        self.file_mod_times = {file: os.path.getmtime(os.path.join(os.path.dirname(__file__), file)) for file in model_files}
        self.InitialLoad()
        threading.Thread(target=self._listener_thread, daemon=True).start()

    def InitialLoad(self):
        print(f"Loading {len(self.model_files)} model " + ("architecture" if len(self.model_files) == 1 else "architectures") + "...", color=Colors.BLUE)
        for i, filename in enumerate(self.model_files):
            try:
                model = self.LoadModelType(filename)
                print(f"Loaded {model.name} model", color=Colors.GREEN)
                self.model_types.append(model)
            except Exception as e:
                print(f"Failed to load model {filename.replace('_', ' ').title()}: {e}", color=Colors.RED)

        if len(self.model_types) == 0:
            print("No valid models found, exiting", color=Colors.RED)
            exit(1)

        print(f"Loaded {len(self.model_types)} model " + ("architecture" if len(self.model_types) == 1 else "architectures"), color=Colors.GREEN, reprint=True)
        empty_line()

    def _listener_thread(self):
        while True:
            edited = False
            for i, file in enumerate(self.model_files):
                current_mod_time = os.path.getmtime(os.path.join(os.path.dirname(__file__), file))
                if current_mod_time != self.file_mod_times[file]:
                    model_name = self.model_types[i].name
                    edited = True
                    print(f"{model_name} has been updated. Reloading...", color=Colors.BLUE, reprint=True)
                    self.model_types[i] = self.LoadModelType(file)
                    print(f"{model_name} has been reloaded.", color=Colors.GREEN, reprint=True)
                    self.file_mod_times[file] = current_mod_time
            empty_line() if edited else None
            time.sleep(5) # Lock to 0.2FPS
    
    def LoadModelType(self, file: str):
        model_import = importlib.import_module(f"model_types.{file[:-3]}")
        model_name = file[:-3] if "name" not in model_import.__dict__ else model_import.name

        # Check for required variables
        required_variables = ["name", "description", "hyperparameters", "data_type"]
        for var in required_variables:
            if var not in model_import.__dict__:
                raise LoadExceptions.MissingVar(f"Model {model_name} is missing the {var} variable (Check the in-app docs for more info)")

        variable_types = [[model_import.name, str], [model_import.description, str], [model_import.data_type, str], [model_import.hyperparameters, list]]
        for variable in variable_types:
            if not isinstance(variable[0], variable[1]):
                raise LoadExceptions.TypeException(f"Model {model_name} {variable[0].split('.')[-1]} is not a {variable[1]} (Check the in-app docs for more info)")
        
        for hyperparameter in model_import.hyperparameters:
            if not isinstance(hyperparameter, Hyperparameter):
                raise TypeError(f"Model {model_name} hyperparameter {hyperparameter.name} is not a RequiredHyperparameter object (Check the in-app docs for more info)")
            
        if not "Epochs" in [hyperparameter.name for hyperparameter in model_import.hyperparameters]:
            raise LoadExceptions.MissingVar(f"Model {model_name} is missing the Epochs hyperparameter (Check the in-app docs for more info)")

        if model_import.data_type not in ["text", "image", "audio", "other"]:
            raise LoadExceptions.IncorrectType(f"Model {model_name} data_type is not a valid data type (Should be one of 'text', 'image', 'audio', 'other')")

        if "Model" not in model_import.__dict__:
            raise LoadExceptions.MissingVar(f"Model {model_name} is missing the Model class (Check the in-app docs for more info)")

        model = model_import.Model
        if not issubclass(model, ModelTemplate):
            raise LoadExceptions.TypeException(f"Model {model_name} Model class is not a subclass of ModelTemplate (Check the in-app docs for more info)")

        required_funcs = ["Initialize", "Train", "Save"]
        for func in required_funcs:
            if func not in model.__dict__:
                raise LoadExceptions.MissingVar(f"Model {model_name} is missing the {func} function (Check the in-app docs for more info)")

        model = Model(
            name = model_import.name, # "name" variable in file
            description = model_import.description, # "description" variable in file
            data_type = model_import.data_type, # "data_type" variable in file
            model_class = model, # "Model" class in file
            hyperparameters = model_import.hyperparameters # "hyperparameters" variable in file
        )

        return model

    def Getmodel_types(self):
        return self.model_types

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

class Hyperparameter:
    name : str
    value : str | int | float | bool | None = None
    
    default : str | int | float | bool
    min_value : int | float | None
    max_value : int | float | None
    incriment : int | float | None
    special_type : Literal["path", "dropdown", None]
    options : list[str | int | float | bool] | None
    description : str

    def __init__(self, name: str, default: str | int | float | bool, min_value: int | float | None = None, max_value: int | float | None = None, incriment: int | float | None = None, special_type: Literal["path", "dropdown", None] = None, options: list[str | int | float | bool] | None = None, description: str = "") -> None:
        """
        Hyperparameter for a model, will be used to ask the user for input on the frontend

        Parameters:
            name (str): The name of the hyperparameter.
            default (str | int | float | bool): The default value of the hyperparameter.
            min_value (int | float | None, optional): The minimum value for the hyperparameter (only for number values).
            max_value (int | float | None, optional): The maximum value for the hyperparameter (only for number values).
            incriment (int | float | None, optional): The increment step for the hyperparameter (only for number values).
            special_type (Literal["path", "dropdown", None], optional): The special type of the hyperparameter.
            options (list[str | int | float | bool] | None, optional): The list of valid options if special_type is "dropdown".
            description (str, optional): The description of the hyperparameter.
        
        Raises:
            TypeError: If any parameter is not of the expected type.
            ValueError: If parameters do not meet certain conditions.
        """
        # Type checking
        if not isinstance(name, str): raise TypeError("Name must be a string")
        if not isinstance(description, str): raise TypeError("Description must be a string")
        if not isinstance(default, (str, int, float, bool)): raise TypeError("Hyperparameters can only be strings, integers, floats, or booleans")
        if min_value != None and not isinstance(min_value, (int, float)): raise TypeError("Min value must be an integer, float, or None")
        if max_value != None and not isinstance(max_value, (int, float)): raise TypeError("Max value must be an integer, float, or None")
        if special_type != None and not isinstance(special_type, str): raise TypeError("Special type must be 'path', 'dropdown', or None")
        if options != None and not isinstance(options, list): raise TypeError("Options must be a list of strings, integers, floats, or booleans")
        
        # Condition checking
        if special_type == "dropdown" and options == None: raise ValueError("Options must be provided if special type is dropdown")
        if special_type == "path" and not isinstance(default, str): raise ValueError("Default must be a string if special type is path")
        if min_value != None or max_value != None:
            if not isinstance(default, (int, float)): raise ValueError("Default must be an integer or float if min and max values are provided")
            if min_value != None:
                if not isinstance(min_value, (int, float)): raise ValueError("Min value must be an integer or float")
                if default < min_value: raise ValueError("Default must be greater than or equal to min value")
            if max_value != None:
                if not isinstance(max_value, (int, float)): raise ValueError("Max value must be an integer or float")
                if default > max_value: raise ValueError("Default must be less than or equal to max value")
            if min_value != None and max_value != None:
                if min_value > max_value: raise ValueError("Min value must be less than or equal to max value")
        if options != None:
            for option in options:
                if not isinstance(option, (str, int, float, bool)): raise ValueError("Options must be a list of strings, integers, floats, or booleans")
            if default not in options: raise ValueError("Default must be in options")
            
        if isinstance(default, (int, float)):
            if incriment == None: incriment = 1
        else:
            if min_value != None: raise ValueError("Min value must be None if default is not an integer or float")
            if max_value != None: raise ValueError("Max value must be None if default is not an integer or float")
            if incriment != None: raise ValueError("Incriment must be None if default is not an integer or float")

        self.name = name
        self.value = default
        self.min_value = min_value
        self.max_value = max_value
        self.incriment = incriment
        self.special_type = special_type
        self.options = options
        self.description = description

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "incriment": self.incriment,
            "special_type": self.special_type,
            "options": self.options,
            "description": self.description
        }

class AdditionalTrainingData:
    name : str
    value : str | int | float | bool

    def __init__(self, name: str, value: str | int | float | bool) -> None:
        """
        A model must have an attribute called self.additional_training_data that is a list of AdditionalTrainingData instances
        It should hold any extra data that is updated every epoch You should update this attribute every time that the train function is called.

        Args:
            name (str): Name of the additional training data.
            value (str | int | float | bool): Value of the additional training data.
        """
        if not isinstance(name, str): raise TypeError("Name must be a string")
        if not isinstance(value, (str, int, float, bool)): raise TypeError("Value must be a string, integer, float, or boolean")

        self.name = name
        self.value = value

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value
        }

class ModelData:
    name : str
    value : str | int | float | bool

    def __init__(self, name: str, value: str | int | float | bool) -> None:
        """
        A model can have an attribute called self.model_data that is a list of OptionalModelData instances
        It should hold any extra data about your model and training utilities. You should update this attribute in the Initialize function

        Args:
            name (str): Name of the optional model data.
            value (str | int | float | bool): Value of the optional model data.
        """
        if not isinstance(name, str): raise TypeError("Name must be a string")
        if not isinstance(value, (str, int, float, bool)): raise TypeError("Value must be a string, integer, float, or boolean")

        self.name = name
        self.value = value
    
    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value
        }

class HyperparameterFetcher:
    def __init__(self, hyperparameters : list[Hyperparameter]) -> None:
        if not isinstance(hyperparameters, list): raise TypeError("Hyperparameters must be a list of Hyperparameters")
        for hyperparameter in hyperparameters: 
            if not isinstance(hyperparameter, Hyperparameter): raise TypeError("Hyperparameter must be an instance of Hyperparameter")

        self.hyperparameters = hyperparameters

    def GetHyp(self, name : str) -> str | int | float | bool:
        """
        Extract a hyperparameter from a list of hyperparameters by name.

        Args:
            name (str): Name of the hyperparameter to extract.
            hyperparameters (list[Hyperparameter]): List of hyperparameters.

        Returns:
            str | int | float | bool: Value of the hyperparameter.
        """
        for hyperparameter in self.hyperparameters:
            if hyperparameter.name == name:
                return hyperparameter.value
        raise ValueError(f"Hyperparameter {name} not found")
    
    def GetAllHyps(self) -> list[Hyperparameter]:
        """
        Return the hyperparameters that the class was initialized with.

        Returns:
            list[Hyperparameter]: List of hyperparameters.
        """
        return self.hyperparameters

    def GetAllHypsAsDict(self) -> list[dict]:
        """
        Return the hyperparameters that the class was initialized with as a list of dictionaries.

        Returns:
            list[dict]: List of hyperparameters as dictionaries.
        """
        return [hyperparameter.to_dict() for hyperparameter in self.hyperparameters]

class ModelTemplate:
    def __init__(self, error_handler = None) -> None:
        """
        Initialize a ModelTemplate instance.
        """
        self.model : torch.nn.Module = None
        self.model_data : list[ModelData] = []
        self.additional_training_data : list[AdditionalTrainingData] = []
        self.training_loss : list[float] = []
        self.validation_loss : list[float] = []

        self.error : str = None
        self.error_handler : TrainingController.ErrorHandler = error_handler

    def Initialize(self) -> None:
        pass

    def Train(self) -> None:
        pass

    def Save(self) -> None:
        pass

    def RaiseError(self, error : str, traceback : str) -> None:
        self.error = True
        self.error_handler(error, traceback)

class Model:
    name : str
    description : str
    data_type : str
    model_class : ModelTemplate
    hyperparameters : list[Hyperparameter]

    def __init__(self, name: str, description: str, data_type: str, model_class: ModelTemplate, hyperparameters: list[Hyperparameter]) -> None:
        self.name = name
        self.description = description
        self.data_type = data_type
        self.model_class = model_class
        self.hyperparameters = hyperparameters
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "hyperparameters": [hyperparameter.to_dict() for hyperparameter in self.hyperparameters]
        }

id_tracker = 0
def GetID():
    global id_tracker
    id_tracker += 1
    return id_tracker

class TrainingController:
    def __init__(self, model: Model) -> None:
        self.name = model.name
        self.description = model.description
        self.data_type = model.data_type
        self.hyperparameters = HyperparameterFetcher(model.hyperparameters) # Initialize the hyperparameter fetcher
        self.id = GetID()
        self.train_thread = threading.Thread(target=self._train_thread)

        self.epoch = 0
        self.status = "Initializing"

        self.times_per_epoch = []
        self.best_training_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_model = None
        self.best_epoch = 0

        self.start_time = 0
        self.end_time = 0

        self.error = None
        self.traceback = None

        self.model : ModelTemplate = model.model_class(self.hyperparameters, self.ErrorHandler) # Initialize the model's controller (not the model itself)
        self.model.Initialize()
        # If self.error is true (self.ErrorHandler has been triggered), then the webserver will handle the error

    def GetHyperparameters(self):
        return self.hyperparameters.GetAllHyps()
    
    def Train(self):
        self.start_time = time.time()
        self.train_thread.start()

    def _train_thread(self):
        for epoch in range(self.hyperparameters.GetHyp("Epochs")):
            epoch_start_time = time.time()
            self.status = "Initializing" if epoch == 0 else "Training"
            self.epoch = epoch

            self.model.Train()
            if self.error: # self.ErrorHandler has been triggered
                return
            
            if self.model.validation_loss[-1] < self.best_val_loss:
                self.best_val_loss = self.model.validation_loss[-1]
                self.best_epoch = self.epoch
                self.best_model : torch.nn.Module = self.model.model
            if self.model.training_loss[-1] < self.best_training_loss:
                self.best_training_loss = self.model.training_loss[-1]
            
            self.times_per_epoch.append(time.time() - epoch_start_time)

        self.model.Save(self.best_model)
        self.status = "Finished"

    def ErrorHandler(self, error, traceback = None):
        self.error = error
        self.traceback = traceback
        prev_status = self.status
        self.status = "Error"

        error_message = f"Initialize" if prev_status == "Initializing" else f"Epoch {self.epoch}"
        traceback_message = f"\n{traceback}" if traceback else error
        print(f"Error in {self.name} model {self.id} ({error_message}): {traceback_message}", color=Colors.RED)

    def EstTimeRemaining(self) -> float | str:
        if self.status != "Training": return 0 if self.status == "Finished" else "Unknown"
        if len(self.times_per_epoch) == 0: return 0
        return round((np.mean(self.times_per_epoch) * self.hyperparameters.GetHyp("Epochs")) - (time.time() - self.start_time), 2)

    def GetFrontendData(self):
        return {
            "type": self.name,
            "data_type": self.data_type,
            "status": self.status,
            "epoch": self.epoch,
            "epochs": self.hyperparameters.GetHyp("Epochs"),
            "training_losses": [round(loss, 8) for loss in self.model.training_loss],
            "val_losses": [round(loss, 8) for loss in self.model.validation_loss],
            "best_epoch": self.best_epoch,
            "best_training_loss": round(self.best_training_loss, 8) if self.best_training_loss != float('inf') else None,
            "best_val_loss": round(self.best_val_loss, 8) if self.best_val_loss != float('inf') else None,
            "elapsed": round(time.time() - self.start_time, 2),
            "estimated_time": self.EstTimeRemaining(),
            "time_per_epoch": 0 if len(self.times_per_epoch) == 0 else np.mean(self.times_per_epoch),
            "model_data": [model_data.to_dict() for model_data in self.model.model_data],
            "additional_training_data": [additional_data.to_dict() for additional_data in self.model.additional_training_data],
            "hyperparameters": self.hyperparameters.GetAllHypsAsDict()
        }