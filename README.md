
[![PyPI - Version](https://img.shields.io/pypi/v/deeplocalizer.svg)](https://pypi.org/project/deeplocalizer)
<img src="https://github.com/user-attachments/assets/b00dc021-8e31-44be-9a69-ba33ed8054c6" width="800px">
**DeepLocalizer**: Quickly find functional specialization in deep neural networks. 

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) and [Brain-like functional specialization emerges spontaneously in deep neural networks](https://web.mit.edu/bcs/nklab/media/pdfs/Dobs_2022.pdf) to other models and data.

Examples:

- [Face Localizer Resnet Report](https://www.donnybertucci.com/project/deeplocalizer)
- [Face Localizer Resnet Notebook](./resnet34_example.ipynb)

**Roadmap**

- [x] Replicate some parts of original paper (https://github.com/xnought/paper-implement/tree/main/language_network)
- [x] Write code from scratch to do analysis on face with resnet
	- [x] Set up face localizer example w/ goal of applying to a resnet model
		- [x] 5k positive (faces) from CelebA
		- [x] 5k negative (objects) from COCO
	- [x] Extract activations from the resnet model
		- [x] test track the activations
		- [x] store activations on disk
	- [x] Contrast positive vs negative activations
	- [x] Ablation w/ statistical tests on resnet
		- [x] write code to ablate torch models easily 
		- [x] ablate given the top percent face activations
		- [x] Compare performance after ablation
- [x] Write general API from most helpful functions so others can easily use the library 
	- [x] Activation computation
	- [x] Analysis computation
		- [x] Top percent global
		- [x] Visualizations
		- [x] Ablate model with the top percent
		- [x] Compute statistics on ablated model
- [x] Write report on the resnet example and if localization seems to work and what evidence (here -> https://www.donnybertucci.com/project/deeplocalizer)


## Usage

**API**

See [`resnet34_example.ipynb`](./resnet34_example.ipynb) for doing localization on a torch model with a custom dataset/task. 


```bash
uv add deeplocalizer
```

Or if you are old school

**Install**
```bash
pip install deeplocalizer
```

## Development

```bash
cd deeplocalizer # this git repo
```

Make sure to have https://docs.astral.sh/uv/ installed.

**Install and Run**

```bash
uv sync
uv run deeplocalizer/deeplocalizer.py
```

or run an example python notebook within the .env generated.

## References

**papers**
- https://arxiv.org/abs/2411.02280
- https://web.mit.edu/bcs/nklab/media/pdfs/Dobs_2022.pdf

**code/datasets**
- https://huggingface.co/datasets/tpremoli/CelebA-attrs
- https://huggingface.co/datasets/phiyodr/coco2017
- https://huggingface.co/microsoft/resnet-50
