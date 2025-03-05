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
    data: list[Any],
    model_forward: Callable[[list[Any]], Any],
    layers_activations: list[torch.nn.Module],
    batch_size=32,
) -> list[torch.Tensor]:
    N = len(data)
    assert N > 0, "Must have data!"
    assert len(layers_activations) > 0, "Must have layers we take activations from!"
    assert batch_size > 0, "batch size must be greater than 0"

    accumulate = [None] * len(layers_activations)  # we will be accumulating the tensors
    with ActivationTracker(layers=layers_activations) as tracker:
        for i in tqdm(range(0, N, batch_size)):
            batch = data[i : i + batch_size]
            model_forward(batch)
            mean_accumulate_tensors(
                accumulator=accumulate, tensors=tracker.list_activations, total_length=N
            )

    return accumulate


def compute_task_activations(
    df: pd.DataFrame,
    model_forward: Callable[[list[Any]], Any],
    layers_activations: list[torch.nn.Module],
    batch_size=32,
):
    assert "data" in df.columns, "Must have a column named data"
    assert "positive" in df.columns, (
        "Must have a column named 'positive' which is either True (task) or False (control)"
    )

    task = df[df["positive"] == True]
    control = df[df["positive"] == False]

    task_acts = accumulate_activations(
        task["data"], model_forward, layers_activations, batch_size
    )
    control_acts = accumulate_activations(
        control["data"], model_forward, layers_activations, batch_size
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


def load_activations_from_disk(filename: str, device="cpu") -> list[torch.Tensor]:
    from safetensors.torch import safe_open
    import os

    assert os.path.exists(filename), "File must exist!"

    tensors = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return [tensors[layer_name(i)] for i in range(len(tensors))]


def map_flat(tensor_dict):
    return torch.hstack([tensor_dict[l].flatten() for l in tensor_dict])


def prod(l):
    p = 1
    for i in l:
        p *= i
    return p


def squarify(t):
    d = int(prod(t.shape) ** (0.5))
    return t.flatten()[: d * d].reshape((d, d))


def visualize_activations(activations: list[torch.Tensor], grid=None, cmap="viridis"):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    if grid is None:
        # first combine all the layers so can do argpartition
        for i, a in enumerate(activations):
            plt.title(f"i={i}")
            plt.imshow(squarify(a), cmap=cmap)
            plt.colorbar()
            plt.show()
    else:
        fig, axes = plt.subplots(*grid, figsize=(16, 9))
        for i, ax in enumerate(axes.flat):
            a = activations[i]
            im = ax.imshow(squarify(a), cmap=cmap)
            ax.set_title(f"i={i}")
            plt.colorbar(im, ax=ax)
            ax.axis("off")
        plt.show()


def _map(func, arr):
    return list(map(func, arr))


def overall_activation(activations: list[torch.Tensor]):
    return _map(lambda t: t.abs(), activations)


def append_array_at_key(_dict, k, v):
    if k in _dict:
        _dict[k].append(v)
    else:
        _dict[k] = [v]


def flat_idx_to_layer_idx(flat_idx, edges):
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i + 1]
        if flat_idx >= start and flat_idx < end:
            return i, flat_idx - start
    raise Exception("Didn't find any index")


def flat_idx_to_layer_idx_fast(flat_idxs, edges):
    layer_idxs = torch.searchsorted(edges, flat_idxs, right=True) - 1
    layer_flat_idxs = flat_idxs - edges[layer_idxs]
    return torch.stack((layer_idxs, layer_flat_idxs), dim=1)


def format_topk_global_return(tensors, layers_idxs):
    result_idxs = [[] for _ in range(len(tensors))]
    result_values = [[] for _ in range(len(tensors))]
    for layer_idx, layer_flat_idx in layers_idxs:
        t = tensors[layer_idx]
        result_idxs[layer_idx].append(layer_flat_idx.item())
        result_values[layer_idx].append(t.view(-1)[layer_flat_idx].item())
    return result_idxs, result_values


def topk_global(tensors: list[torch.Tensor], k):
    # flatten all the tensors, find the topk, then map those flat idxs back to the layers indices
    flat_concat = torch.hstack([t.flatten() for t in tensors])
    global_values, global_idxs = flat_concat.topk(k)

    edges = torch.tensor([0] + [prod(t.shape) for t in tensors]).cumsum(0)
    layers_idxs = flat_idx_to_layer_idx_fast(global_idxs, edges)
    return format_topk_global_return(tensors, layers_idxs)


def top_percent_global(tensors: list[torch.Tensor], percent=1):
    total_length = sum(prod(t.shape) for t in tensors)
    k = int((percent / 100) * total_length)
    return topk_global(tensors, k)


def visualize_top_activations(top_idxs, top_values, activation):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    canvas = torch.zeros(activation.shape)
    act = torch.tensor(top_values)
    idxs = torch.tensor(top_idxs)

    canvas.view(-1)[idxs] = act

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].set_title("Original")
    axes[0].imshow(squarify(activation), cmap="inferno")

    axes[1].set_title("Top P Percent")
    axes[1].imshow(squarify(canvas), cmap="inferno")
    plt.show()


class AblateTorchModel:
    def __init__(self, layers, ablate, scaler=0):
        self.layers = layers
        self.ablate = ablate
        self.hooks = []
        for l, a in zip(layers, ablate):
            self.hooks.append(self.register_hook(l, a, scaler))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_hook(self, layer, ablate, scaler):
        def hook(module, inputs, outputs):
            if len(ablate) > 0:
                B = outputs.shape[0]
                flat = outputs.view(B, -1)
                flat[:, ablate] *= scaler
            return outputs

        return layer.register_forward_hook(hook)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.remove_hooks()


def combine_batches(model_results: list[torch.Tensor] | list[tuple]):
    assert len(model_results) > 0
    if isinstance(model_results[0], tuple):
        results = [None for _ in range(len(model_results[0]))]
        for j in range(len(model_results[0])):
            col = [model_results[i][j] for i in range(len(model_results))]
            for c in col:
                print(c.shape)
            results[j] = torch.cat(col)
        return tuple(results)
    else:
        return torch.cat(model_results)


def regular_inference(
    data: list[Any],
    model_forward: Callable[[list[Any]], Any],
    batch_size=32,
):
    N = len(data)
    model_results = []
    for i in tqdm(range(0, N, batch_size)):
        batch = data[i : i + batch_size]
        out = model_forward(batch)
        model_results.append(out)
    return combine_batches(model_results)


def ablated_inference(
    data: list[Any],
    model_forward: Callable[[list[Any]], Any],
    layers_activations: list[torch.nn.Module],
    to_ablate: list[list[int]],
    batch_size=32,
    ablate_factor=0,
) -> torch.Tensor | tuple:
    with AblateTorchModel(layers_activations, to_ablate, ablate_factor):
        return regular_inference(data, model_forward, batch_size)


def task_ablated(
    df: pd.DataFrame,
    model_forward: Callable[[pd.DataFrame], Any],
    layers_activations: list[torch.nn.Module],
    to_ablate: dict[int, list[int]],
    batch_size=32,
    ablate_factor=0,
):
    assert "data" in df.columns, "Must have a column named data"
    assert "positive" in df.columns, (
        "Must have a column named 'positive' which is either True (task) or False (control)"
    )

    task = df[df["positive"] == True]
    ablated_task = ablated_inference(
        task["data"],
        model_forward,
        layers_activations,
        to_ablate,
        batch_size,
        ablate_factor,
    )
    regular_task = regular_inference(
        task["data"],
        model_forward,
        batch_size,
    )

    control = df[df["positive"] == False]
    ablated_control = ablated_inference(
        control["data"],
        model_forward,
        layers_activations,
        to_ablate,
        batch_size,
        ablate_factor,
    )
    regular_control = regular_inference(
        control["data"],
        model_forward,
        batch_size,
    )

    return (ablated_task, regular_task), (ablated_control, regular_control)


if __name__ == "__main__":
    from PIL import Image
    import os

    CACHED_ACTIVATIONS = "resnet_face.safetensors"
    activations = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VIS = False

    from transformers import AutoImageProcessor, ResNetForImageClassification

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-34")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-34")
    model = model.to(DEVICE)
    print("*Loaded Resnet34 from Huggingface")

    task = pd.read_parquet("./experiments/data/tasks/face_task2k.parquet")
    print("*Loaded Face Localizer Task")

    # DEFINE HOW THE MODEL COMPUTES ACTIVATIONS
    @torch.no_grad()
    def resnet_forward(image_paths):
        images = [
            Image.open(f"./experiments/data/{p}").convert("RGB") for p in image_paths
        ]
        inputs = processor(images, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        return outputs.logits

    resnet_blocks = [
        layer for stage in model.resnet.encoder.stages for layer in stage.layers
    ]

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
        activations = load_activations_from_disk(CACHED_ACTIVATIONS, DEVICE)
        print(f"*Loaded Activations from {CACHED_ACTIVATIONS}")

    if VIS:
        visualize_activations(activations, (4, 4))
        visualize_activations(overall_activation(activations), (4, 4), cmap="inferno")

    top_idxs, top_values = top_percent_global(activations, 1)
    print("*Computed top 1 percent activations to ablate")

    if VIS:
        for idxs, values, tensor in zip(top_idxs, top_values, activations):
            if len(idxs) == 0:
                continue
            visualize_top_activations(idxs, values, tensor)

    (ablated_task, regular_task), (ablated_control, regular_control) = task_ablated(
        task.sample(100), resnet_forward, resnet_blocks, top_idxs
    )
    print("*Computed Ablated Model versus Regular")
