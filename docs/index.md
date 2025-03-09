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

For a face localization on Resnet see [`resnet34_example.ipynb`](https://github.com/xnought/deeplocalizer/blob/main/resnet34_example.ipynb).

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
        show_source: true

### Types

::: deeplocalizer.ModelForwardFunc  

::: deeplocalizer.ModelForwardReturn  

::: deeplocalizer.SaveActivationsFunc 

::: deeplocalizer.AblateActivationsFunc    

::: deeplocalizer.AblateIdxs   