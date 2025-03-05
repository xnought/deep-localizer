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
    accumulate = [None] * len(layers_activations)  # we will be accumulating the tensors
    with ActivationTracker(layers=layers_activations) as tracker:
        for i in tqdm(range(0, N, batch_size)):
            batch = task.iloc[i : i + batch_size]
            model_forward(batch)
            mean_accumulate_tensors(
                accumulator=accumulate, tensors=tracker.list_activations, total_length=N
            )

    return accumulate


if __name__ == "__main__":
    from transformers import AutoImageProcessor, ResNetForImageClassification
    from PIL import Image

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-34")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-34")
    model = model.to(DEVICE)
    print("*Loaded Resnet34 from Huggingface")

    task = pd.read_parquet("./experiments/data/tasks/face_task2k.parquet")
    small_task = task.sample(500)
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
    acts = accumulate_activations(
        task=small_task,
        model_forward=resnet_forward,
        layers_activations=resnet_blocks,
        batch_size=32,
    )
    print("*Computed Activations")
