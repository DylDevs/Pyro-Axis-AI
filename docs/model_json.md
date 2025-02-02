---
icon: code
date: 2025-2-2
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

Now we'll start off with what needs to be and can be included in the file:

==- [!badge variant="danger" text="Required"] name
==- [!badge variant="danger" text="Required"] description
==- [!badge variant="danger" text="Required"] definitions_py
==- [!badge variant="danger" text="Required"] data_type
==- [!badge variant="danger" text="Required"] hyperparameters
==- [!badge variant="danger" text="Required"] initialize_function
==- [!badge variant="danger" text="Required"] train_function
==- [!badge variant="danger" text="Required"] save_function
==- [!badge variant="success" text="Optional"] progress_bars
==- [!badge variant="success" text="Required"] graphs
!!! info Coming Soon
I haven't yet found a good way to add this functionality in a clean and easy to use way. It will be added once the rest of the app is modular.
!!!
==- [!badge variant="success" text="Required"] info_dropdowns
===