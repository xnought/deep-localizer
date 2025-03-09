# Usage 

This library takes in a task (like face localization) and computes activations for the data and computes the top percent of activations and can compute the model with those activations ablated.

A task is just a Pandas dataframe with two required columns and one optional:

- `data`: contains the data or path to the data (you define how to fetch data in `model_forward` later on).
- `positive`: True or False. True when part of positive stimuli (ie faces). False when part of negative control stimuli (ie objects in the face localizer case).
- Optional `validation`: True or False. False means part of localization training. True means part of unseen validation that can be used later on to test the performance.

See [`face_data_viewer.ipynb`](https://github.com/xnought/deeplocalizer/blob/main/face_data_viewer.ipynb) to view an example task.

**Next** see how to operate on the task to localize your model below!

## Install

```bash
pip install deeplocalizer
```

or 

```bash
uv add deeplocalizer
```

**Import**

```python
from deeplocalizer import DeepLocalizer # or import other functions in API below
```

## Example

An example using all of the core and visualization functions, see the face localization on Resnet example:  [`resnet34_example.ipynb`](https://github.com/xnought/deeplocalizer/blob/main/resnet34_example.ipynb).

## API

### Core

::: deeplocalizer.load_task

::: deeplocalizer.DeepLocalizer
    options:
        members: ["compute_activations", 
                  "load_activations", 
                  "save_activations",
                  "top_percent_activations",
                  "regular_model_forward",
                  "ablate_model_forward"]
        show_source: false

### Visualization

::: deeplocalizer.visualize_activations

::: deeplocalizer.visualize_top_per_layer

::: deeplocalizer.visualize_top_activations

### Types

::: deeplocalizer.ModelForwardFunc  

::: deeplocalizer.ModelForwardReturn  

::: deeplocalizer.SaveActivationsFunc 

::: deeplocalizer.AblateActivationsFunc    

::: deeplocalizer.AblateIdxs   