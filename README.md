<img src="https://github.com/user-attachments/assets/b00dc021-8e31-44be-9a69-ba33ed8054c6" width="800px">

**DeepLocalizer**: A library to quickly find functional specialization in deep neural networks.

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) and [Brain-like functional specialization emerges spontaneously in deep neural networks](https://web.mit.edu/bcs/nklab/media/pdfs/Dobs_2022.pdf) to other models and data.

**Roadmap**

- [x] Replicate some parts of original paper (https://github.com/xnought/paper-implement/tree/main/language_network)
- [ ] Set up face localizer example w/ goal of applying to a resnet model
	- [ ] Create example dataset with 8k positive (faces) examples and 8k negative examples (non-faces like objects, scenes, ...)
- [ ] Basic API around pandas for data examples, with positive and negative/control examples
- [ ] Easy way to extract activations from the model
- [ ] Localization analysis API around activations of the data points given each model
- [ ] Visualize outputs
- [ ] Setup way for other people to use
- [ ] Optimize for lower memory usage and faster execution (batches)

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
import PIL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
resnet_forward = lambda image: model(**preprocessor(image))
fetch_image = lambda image_id: PIL.Image.open(image_id)

task = ... # you define this (has 'id' and 'positive' columns)
task_with_acts = dl.torch_activations(
	task, 
	data_loader=fetch_image,
	model_forward=resnet_forward,
	extract_activations=[model.l1, model.l2, model.l3], # get outputs from l1, l2, and l3 as activations
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