# DeepLocalizer

[![PyPI - Version](https://img.shields.io/pypi/v/deeplocalizer.svg)](https://pypi.org/project/deeplocalizer) 

[Blog Post](https://www.donnybertucci.com/project/deeplocalizer#blog)


**Quickly find functional specialization in PyTorch models.**

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) to other models and data.


Examples:

- [Face Localizer Resnet Report](https://www.donnybertucci.com/project/deeplocalizer)
- [Face Localizer Resnet Notebook](./resnet34_example.ipynb)
- [Face Localizer Data Viewer](./face_data_viewer.ipynb)

## Task Usage

Tasks are just pandas dataframes with a `data`, `positive`, and `validation` columns. Each row is a different data point.

See [`face_data_viewer.ipynb`](./face_data_viewer.ipynb) to see a real example of a face localizer data (face images from CelebA vs. objects images from COCO).


**What do the columns mean?**

- `data` is the data itself (eg text) or points to data (eg image filename)
- `positive` is `True` for the task and `False` for the control (eg face images have `True` and control images have `False`)
- `validation` is technically optional. If you want to notate some rows to only be used later on to test performance and not for the main localization, you can indicate a subset of the rows as `True`. The main dataset used for localization is then `False`.

Again see [`face_data_viewer.ipynb`](./face_data_viewer.ipynb) if you're still confused.

## Library Usage

**Install**

```bash
uv add deeplocalizer
```

or

```bash
pip install deeplocalizer
```

**API Usage**

See [`resnet34_example.ipynb`](./resnet34_example.ipynb) for doing localization on a torch model with a custom dataset/task. 

TODO: better documentation.


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
