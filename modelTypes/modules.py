from typing import Literal
import json

class Hyperparameter:
    name : str
    display_name : str
    value : str | int | float | bool | None = None
    
    default : str | int | float | bool
    min_value : int | float | None
    max_value : int | float | None
    incriment : int | float | None
    special_type : Literal["path", "dropdown", None]
    options : list[str | int | float | bool] | None
    description : str

    def __init__(self, name: str, display_name: str, default: str | int | float | bool, min_value: int | float | None = None, max_value: int | float | None = None, incriment: int | float | None = None, special_type: Literal["path", "dropdown", None] = None, options: list[str | int | float | bool] | None = None, description: str = "") -> None:
        """
        Hyperparameter for a model, will be used to ask the user for input on the frontend

        Parameters:
            name (str): The name of the hyperparameter.
            display_name (str): The display name for the hyperparameter.
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
        if not isinstance(display_name, str): raise TypeError("Display name must be a string")
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
            if min_value == None: min_value = 0
            if max_value == None: max_value = float("inf")
            if incriment == None: incriment = 1
        else:
            if min_value != None: raise ValueError("Min value must be None if default is not an integer or float")
            if max_value != None: raise ValueError("Max value must be None if default is not an integer or float")
            if incriment != None: raise ValueError("Incriment must be None if default is not an integer or float")

        self.name = name
        self.display_name = display_name
        self.value = default
        self.min_value = min_value
        self.max_value = max_value
        self.incriment = incriment
        self.special_type = special_type
        self.options = options

class AdditionalTrainingData:
    name : str
    display_name : str
    value : str | int | float | bool

    def __init__(self, name: str, display_name: str, value: str | int | float | bool) -> None:
        """
        A model must have an attribute called self.additional_training_data that is a list of AdditionalTrainingData instances
        It should hold any extra data that is updated every epoch You should update this attribute every time that the train function is called.

        Args:
            name (str): Name of the additional training data.
            display_name (str): Display name of the additional training data.
            value (str | int | float | bool): Value of the additional training data.
        """
        if not isinstance(name, str): raise TypeError("Name must be a string")
        if not isinstance(display_name, str): raise TypeError("Display name must be a string")
        if not isinstance(value, (str, int, float, bool)): raise TypeError("Value must be a string, integer, float, or boolean")

        self.name = name
        self.display_name = display_name
        self.value = value

class ModelData:
    name : str
    display_name : str
    value : str | int | float | bool

    def __init__(self, name: str, display_name: str, value: str | int | float | bool) -> None:
        """
        A model can have an attribute called self.model_data that is a list of OptionalModelData instances
        It should hold any extra data about your model and training utilities. You should update this attribute in the Initialize function

        Args:
            name (str): Name of the optional model data.
            display_name (str): Display name of the optional model data.
            value (str | int | float | bool): Value of the optional model data.
        """
        if not isinstance(name, str): raise TypeError("Name must be a string")
        if not isinstance(display_name, str): raise TypeError("Display name must be a string")
        if not isinstance(value, (str, int, float, bool)): raise TypeError("Value must be a string, integer, float, or boolean")

        self.name = name
        self.display_name = display_name
        self.value = value

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

class ModelTemplate:
    def __init__(self) -> None:
        """
        Initialize a ModelTemplate instance.

        This constructor should initialize any necessary attributes for your model
        """
        pass

    def Initialize(self, hyperparameters : list[Hyperparameter]) -> None:
        """
        Initialize the model with hyperparameters.
        In this function:
            - Data should be laoded
            - Model should be initialized
            - Any model training utilities should be initialized

        Args:
            hyperparameters (list[Hyperparameter]): List of hyperparameters to use for the model. (Will be passed by the training controller)
        """
        pass

    def Train(self, hyperparameters : list[Hyperparameter]) -> None:
        """
        Training function for the model.

        This function should train the model using the hyperparameters provided, this function will be ran every epoch

        Args:
            hyperparameters (list[Hyperparameter]): List of hyperparameters to use for the model. (Will be passed by teh training controller)

        Returns:
            None
        """
        pass

class ModelData:
    name : str
    display_name : str
    description : str
    data_type : str
    model_class : ModelTemplate
    hyperparameters : list[Hyperparameter]

    def __init__(self, name: str, display_name: str, description: str, data_type: str, model_class: ModelTemplate, hyperparameters: list[Hyperparameter]) -> None:
        self.name = name
        self.display_name = display_name
        self.description = description
        self.data_type = data_type
        self.model_class = model_class
        self.hyperparameters = hyperparameters
    
    def __dict__(self):
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "data_type": self.data_type,
            "model_class": self.model_class,
            "hyperparameters": self.hyperparameters
        }