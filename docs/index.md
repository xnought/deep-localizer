# Usage 

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