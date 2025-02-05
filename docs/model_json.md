---
icon: code
date: 2025-2-4
authors:
    - name: DylDev
      link: https://github.com/DylDevs
      avatar: https://avatars.githubusercontent.com/u/110776467?v=4
---

# JSON Model Definition
---
To define models for training in PyroAxis, you must create a JSON file which includes data about your model, along with a python file that contains your initialize, train, and save functions. These are simple to make, you can have brand new model training in about 10 minutes!

To start off, create a JSON file in the `model_types` directory. The first required thing in the file is the schema import:
```json
{
    "$schema": "../utils/schema.json"
}
```
This line will ensure that you are including all required keys with the correct data types, if the schema import is not detected on atartup, the model will not be added to the models list.

!!! Variables and Functions
Any time that a variable or function is referenced in the JSON file, it must be part of your Model class
!!!
Now we'll start off with what needs to be and can be included in the file:

==- [!badge variant="ghost" text="Required"] name
**Description**\
The name of your model. This will be displayed on the dashboard, so make sure it is in title format.

**Example**
```json
{
    "name": "GPT"
}
```
==- [!badge variant="ghost" text="Required"] description
**Description**\
A short statement that describes your models function

**Example**
```json
{
    "description": "Generates human like text based on a prompt"
}
```
==- [!badge variant="ghost" text="Required"] update_rate
**Description**\
The amount of seconds between data updates
!!!
The minimum value is 0.25 seconds
!!!

**Example**
```json
{
    "update_rate": 5
}
```
==- [!badge variant="ghost" text="Required"] functions_py
**Description**\
Path to the python file which contains your model functions. The path is relative to the model_types folder
!!!
The value must end with `.py` or an exception will be thrown
!!!

**Example**
```json
{
    "functions_py": "gpt.py"
}
```
==- [!badge variant="ghost" text="Required"] data_type
**Description**\
The type of data that your model outputs. Must be one of the following: 
- "text"
- "image"
- "audio"
- "other"

**Example**
```json
{
    "data_type": "text"
}
```
==- [!badge variant="ghost" text="Required"] hyperparameters
!!! warning
The Epochs hyperparameter is defined by the training controller, do not define it in your file.
!!!
**Description**\
An array of dictionaries that each have a different parameter for the user to tweak. Each dictioanry has information about it's name, description, and data type.

**Dictionary Values**
+++ name
!!!
This value is required
!!!
**Description**\
The name of your model. This will be displayed on the dashboard, so make sure it is in title format.

**Example**
```json
{
    "hyperparameters" : [
        {
            "name": "Batch Size"
        }
    ]
}
```
\
+++ default
!!!
This value is required
!!!
**Description**\
The default value of the hyperparameter. This value's type will be assumed as they hyperparameter's type, NoneType values will raise an exception.

**Example**
```json
{
    "hyperparameters" : [
        {
            "default": 32
        }
    ]
}
```
+++ min_value
!!!
This value is required for number values, do not fill for other types
!!!
**Description**\
The minimum value of the hyperparameter. This value must be a number or null for no limit, other values will raise an exception

**Example**
```json
{
    "hyperparameters" : [
        {
            "min_value": 8
        }
    ]
}
```
+++ max_value
!!!
This value is required for number values, do not fill for other types
!!!
**Description**\
The maximum value of the hyperparameter. This value must be a number or null for no limit, other values will raise an exception

**Example**
```json
{
    "hyperparameters" : [
        {
            "max_value": 512
        }
    ]
}
```
+++ incriment
!!!
This value is required for number values, do not fill for other types
!!!
**Description**\
The amount that the parameter will incriment when up and down arrows are used. This value must be a number, other values will raise an exception.

**Example**
```json
{
    "hyperparameters" : [
        {
            "incriment": 1
        }
    ]
}
```
+++ special_type
!!!
This value is optional
!!!
**Description**\
Special data types which will show different selectors in the UI, currently only `path` and `dropdown` are supported, other values will raise an exception.

**Example**
```json
{
    "hyperparameters" : [
        {
            "special_type": "dropdown"
        }
    ]
}
```
+++ options
!!!
This value is required if `special_type` is set to `dropdown`
!!!
**Description**\
Options for the dropdown, values should all be of the same type and the default must be one of the dropdown values.

**Example**
```json
{
    "hyperparameters" : [
        {
            "options": ["CPU", "GPU", "Automatic"]
        }
    ]
}
```
+++ description
!!!
This value is optional
!!!
**Description**\
A description of the hyperparameter, this will be displayed in a tooltip if defined.

**Example**
```json
{
    "hyperparameters" : [
        {
            "description": "Whether to use CPU or GPU for training, if Automatic is selected, GPU will be used if available."
        }
    ]
}
```
+++

**Example**
```json
{
    "hyperparameters": [
        {
            "name": "Batch Size",
            "default": 32,
            "min_value": 8,
            "max_value": 512,
            "incriment": 1,
            "description": "Amount of datapoints fed to the model per forward and backward pass"
        },
        {
            "name": "Device",
            "default": "Automatic",
            "special_type": "dropdown",
            "options": ["CPU", "GPU", "Automatic"],
            "description": "Amount of datapoints fed to the model per forward and backward pass"
        },
    ]
}
```
==- [!badge variant="ghost" text="Required"] initialize_function
**Description**\
Name of the function in your Python file which initializes your model and training utilities
!!!
The value must be exact, for instance, if your definition line in Python is `def initialize_model():`, refer to the example to see what it should be in JSON.
!!!

**Example**
```json
{
    "initialize_function": "initialize_model"
}
```
==- [!badge variant="ghost" text="Required"] train_function
**Description**\
Name of the function in your Python file which trains the model for an epoch.
!!!
The value must be exact, for instance, if your definition line in Python is `def train_model():`, refer to the example to see what it should be in JSON.
!!!

**Example**
```json
{
    "train_function": "train_model"
}
```
==- [!badge variant="ghost" text="Required"] save_function
**Description**\
Name of the function in your Python file which saves your model to the disk
!!!
The value must be exact, for instance, if your definition line in Python is `def save_model():`, refer to the example to see what it should be in JSON.
!!!

**Example**
```json
{
    "save_function": "save_model"
}
```
==- [!badge variant="ghost" text="Optional"] progress_bars
!!!
To get an epoch or time progress bar, use `controller.epoch` or `controller.time`, the rest of the values will be ignored.
!!!
**Description**\
An array of dictionaries that each have information for a progress bar in the UI

**Dictionary Values**
+++ name
!!!
This value is required
!!!
**Description**\
The name of your progress bar. This will be displayed on the dashboard, so make sure it is in title format.

**Example**
```json
{
    "progress_bars" : [
        {
            "name": "Patience"
        }
    ]
}
```
\
+++ description
!!!
This value is optional
!!!
**Description**\
A short description of your progress bar, this will be shown in a tooltip if defined.

**Example**
```json
{
    "progress_bars" : [
        {
            "description": "Shows how many epochs the model has gone since the best epoch and how close it is to the patience value"
        }
    ]
}
```
+++ type
!!!
This value is required
!!!
**Description**\
Whether the value is a `number` or `time` value. Any other value will result in an exception.

**Example**
```json
{
    "progress_bars" : [
        {
            "type": "number"
        }
    ]
}
```
+++ current
!!!
This value is required
!!!

**Description**\
Variable which defines the current progress value. There are 2 ways these values can be selected. The first way is stating a variable name which is part of your Model class. The second way is using `hyperparameter.` in front of a hyperparameter name. This value must be a number.

**Example**
```json
{
    "progress_bars" : [
        {
            "current": "epochs_until_patience"
        }
    ]
}
```
+++ total
!!!
This value is required
!!!

**Description**\
Variable which defines the max value. There are 2 ways these values can be selected. The first way is stating a variable name which is part of your Model class. The second way is using `hyperparameter.` in front of a hyperparameter name. This value must be a number.

**Example**
```json
{
    "progress_bars" : [
        {
            "total": "hyperparameter.patience"
        }
    ]
}
```
+++ progress_test
!!!
This value is required
!!!

**Description**\
The text to display under the progress bar, it must have {0} and {1} as a placeholder for current and total values.

**Example**
```json
{
    "progress_bars" : [
        {
            "progress_text": "{0} epochs out of {1} until early stop"
        }
    ]
}
```
+++
==- [!badge variant="ghost" text="Optional"] graphs
!!! Coming Soon
I haven't yet found a good way to add this functionality in a clean and easy to use way. It will be added once the rest of the app is modular.
!!!
==- [!badge variant="ghost" text="Optional"] info_dropdowns
**Description**\
An array of dictionaries that each have information for a dropdown with information about the model

**Dictionary Values**
+++ title
!!!
This value is required
!!!

**Description**\
Title of the dropdown

**Example**
```json
{
    "info_dropdowns" : [
        {
            "title": "Model Data"
        }
    ]
}
```
\
\
\
\
\
\
+++ description
!!!
This value is optional
!!!

**Description**\
Description of the dropdown, will be shown in a tooltip if sefined

**Example**
```json
{
    "info_dropdowns" : [
        {
            "description": "Shows information about the model architechture"
        }
    ]
}
```
+++ data
!!!
This value is required
!!!
!!! warning
View information about the dictionary values at the bottom of this panel
!!!

**Description**\
An array of dictionaries that give information on the datapoints to include in the dropdown

**Example**
```json
{
    "info_dropdowns" : [
        {
            "data": [
                {
                    "title": "Model Parameters",
                    "value": "model_parameters",
                    "description": "Total parameter count of the model"
                }
            ]
        }
    ]
}
```
+++

**Data Dictionary Values**
+++ title
!!!
This value is required
!!!

**Description**\
Title of the datapoint

**Example**
```json
{
    "info_dropdowns" : [
        {
            "data": [
                {
                    "title": "Model Parameters"
                }
            ]
        }
    ]
}
```
+++ value
!!!
This value is required
!!!

**Description**\
The name of a number variable in your Model class that holds the value of the datapoint

**Example**
```json
{
    "info_dropdowns" : [
        {
            "data": [
                {
                    "value": "model_parameters"
                }
            ]
        }
    ]
}
```
+++ description
!!!
This value is optional
!!!

**Description**\
Short description of the data point, it will be displayed in a tooltip if defined.

**Example**
```json
{
    "info_dropdowns" : [
        {
            "data": [
                {
                    "description": "Total parameter count of the model"
                }
            ]
        }
    ]
}
```
+++
===