import torch
import pandas as pd
from tqdm import tqdm
from typing import Callable, Any


class ActivationTracker:
    def __init__(self, layers):
        self.activations = {}
        self.layers = layers
        self.hooks = []
        for l in layers:
            self.hooks.append(self.register_hook(l))

    @property
    def list_activations(self):
        return [self.activations[l] for l in self.layers]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = []

    def register_hook(self, layer):
        def hook(module, inputs, outputs):
            self.activations[layer] = outputs.detach().cpu()
            return outputs

        return layer.register_forward_hook(hook)

    def shapes(self):
        shapes = []
        for l in self.layers:
            a = self.activations[l]
            shapes.append(list(a.shape))
        return shapes

    def flat_activations(self):
        data = []
        for l in self.layers:
            data.append(self.activations[l].flatten())
        return torch.concat(data)

    def activations_inorder(self):
        data = []
        for l in self.layers:
            data.append(self.activations[l])
        return data

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.remove_hooks()


def accumulate_activations_naive(
    task: pd.DataFrame,
    model_forward: Callable[[pd.DataFrame], Any],
    layers_activations: list[torch.nn.Module],
    batch_size=32,
) -> list[torch.Tensor]:
    act_col = []
    with ActivationTracker(layers=layers_activations) as tracker:
        for i in tqdm(range(0, len(task), batch_size)):
            batch = task.iloc[i : i + batch_size]
            model_forward(batch)
            act_col.append(tracker.activations_inorder())

    accumulated = []
    for j in range(len(act_col[0])):
        col = [act_col[i][j] for i in range(len(act_col))]
        accumulated.append(torch.vstack(col).mean(0))

    return accumulated


def mean_accumulate_tensors(
    accumulator: list[torch.Tensor | None],
    tensors: list[torch.Tensor],
    total_length: int,
):
    # accumulate activations so we don't have to store them all!
    for i, t in enumerate(tensors):
        batch_mean = t.sum(dim=0) / total_length  # mean down batch dim
        if accumulator[i] is None:
            accumulator[i] = batch_mean
        else:
            accumulator[i] += batch_mean


def accumulate_activations(
    task: pd.DataFrame,
    model_forward: Callable[[pd.DataFrame], Any],
    layers_activations: list[torch.nn.Module],
    batch_size=32,
) -> list[torch.Tensor]:
    N = len(task)
    assert N > 0, "Must have data!"
    assert len(layers_activations) > 0, "Must have layers we take activations from!"
    assert batch_size > 0, "batch size must be greater than 0"

    accumulate = [None] * len(layers_activations)  # we will be accumulating the tensors
    with ActivationTracker(layers=layers_activations) as tracker:
        for i in tqdm(range(0, N, batch_size)):
            batch = task.iloc[i : i + batch_size]
            model_forward(batch)
            mean_accumulate_tensors(
                accumulator=accumulate, tensors=tracker.list_activations, total_length=N
            )

    return accumulate


def compute_task_activations(
    df: pd.DataFrame,
    model_forward: Callable[[pd.DataFrame], Any],
    layers_activations: list[torch.nn.Module],
    batch_size=32,
):
    assert "positive" in df.columns, (
        "Must have a column named 'positive' which is either True (task) or False (control)"
    )

    task = df[df["positive"] == True]
    control = df[df["positive"] == False]

    task_acts = accumulate_activations(
        task, model_forward, layers_activations, batch_size
    )
    control_acts = accumulate_activations(
        control, model_forward, layers_activations, batch_size
    )

    # subtract out the control from the task
    regions_of_interest = [t - c for t, c in zip(task_acts, control_acts)]

    return regions_of_interest


def layer_name(i: int):
    return str(i)


def save_activations_to_disk(activations: list[torch.Tensor], filename: str):
    from safetensors.torch import save_file

    export = {layer_name(i): t for i, t in enumerate(activations)}
    save_file(export, filename)


def load_activations_from_disk(filename: str) -> list[torch.Tensor]:
    from safetensors.torch import safe_open
    import os

    assert os.path.exists(filename), "File must exist!"

    tensors = {}
    with safe_open(filename, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return [tensors[layer_name(i)] for i in range(len(tensors))]


if __name__ == "__main__":
    from transformers import AutoImageProcessor, ResNetForImageClassification
    from PIL import Image
    import os

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-34")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-34")
    model = model.to(DEVICE)
    print("*Loaded Resnet34 from Huggingface")

    task = pd.read_parquet("./experiments/data/tasks/face_task2k.parquet")
    print("*Loaded Face Localizer Task")

    # DEFINE HOW THE MODEL COMPUTES ACTIVATIONS
    @torch.no_grad()
    def resnet_forward(task_batch: pd.DataFrame):
        image_paths = task_batch["data"]
        images = [
            Image.open(f"./experiments/data/{p}").convert("RGB") for p in image_paths
        ]
        inputs = processor(images, return_tensors="pt").to(DEVICE)
        return model(**inputs)

    resnet_blocks = [
        layer for stage in model.resnet.encoder.stages for layer in stage.layers
    ]

    CACHED_ACTIVATIONS = "resnet_face.safetensors"
    activations = None
    if not os.path.exists(CACHED_ACTIVATIONS):
        activations = compute_task_activations(
            df=task,
            model_forward=resnet_forward,
            layers_activations=resnet_blocks,
            batch_size=32,
        )
        print("*Computed Activations")

        save_activations_to_disk(activations, CACHED_ACTIVATIONS)
        print("*Saved Activations to Disk")
    else:
        activations = load_activations_from_disk(CACHED_ACTIVATIONS)
        print(f"*Loaded Activations from {CACHED_ACTIVATIONS}")

    print(activations)
