<img src="https://github.com/user-attachments/assets/b00dc021-8e31-44be-9a69-ba33ed8054c6" width="800px">

**DeepLocalizer**: Find subnetworks in deep neural networks most active during a certain task/concept.

This library is meant to provide easy functions to accomplish :arrow_up: and provide evidence if we've actually found those subnetworks or not. 

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) to other models and data.

**Roadmap**

- [x] Replicate some parts of original paper (https://github.com/xnought/paper-implement/tree/main/language_network)
- [ ] Set up face localizer example w/ goal of applying to a resnet model
- [ ] Basic API around pandas for data examples, with positive and negative/control examples
- [ ] Easy way to extract activations from the model
- [ ] Localization analysis API around activations of the data points given each model
- [ ] Visualize outputs
- [ ] Setup way for other people to use

## Usage

(STILL EXPERIMENTAL, NOT YET IMPLEMENTED)

The API minimally needs a pandas dataframe with the following properties:
- Has atleast the three columns: id, positive, and activations.
- id is a unique identifier for each data point, positive is boolean and determines if part of task (true) or control (false), and activation is numpy ndarray where every ndarray in every row must be the same shape

Then you can easily

**Localize**

```python
import deeplocalizer as dl
task_with_acts: pd.DataFrame = ... # you define this (has 'id', 'positive', and 'activations' as columns)
result = dl.localize(task_with_acts)
```

which returns the top k activations for the task.

**Torch API**

If you have a torch model you want to localize for a given task, you can instead define the task, and we compute the activations for you.

You must specifically define which layers we should use the activations for!

```python
import deeplocalizer as dl
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
resnet_forward = lambda image: model(**preprocessor(image))

task = ... # you define this (has 'id' and 'positive' columns)
task_with_acts = dl.torch_activations(
	task, 
	forward=resnet_forward,
	activations_from=[model.l1, model.l2, model.l3], # get outputs from l1, l2, and l3 as activations
	device=DEVICE
)
result = dl.localize(task_with_acts)
```

## Development

Have https://docs.astral.sh/uv/ installed.

```bash
uv sync
uv run example.py
```