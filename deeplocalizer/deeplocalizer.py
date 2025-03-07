import torch
import pandas as pd
from tqdm import tqdm
from typing import Callable, Any
import os
import matplotlib.pyplot as plt

plt.style.use("dark_background")

SaveActivationsFunc = Callable[[torch.Tensor], torch.Tensor]
ModelForwardFunc = Callable[[list[Any]], Any]
AblateIdxs = torch.Tensor | list[int]
AblateActivationsFunc = Callable[[torch.Tensor, AblateIdxs, float], torch.Tensor]


def default_save_activations(acts):
    return acts


class ActivationTracker:
    def __init__(
        self,
        layers: list[torch.nn.Module],
        save_activations: SaveActivationsFunc = default_save_activations,
    ):
        self.activations = {}
        self.layers = layers
        self.hooks = []
        for l in layers:
            self.hooks.append(self.register_hook(l, save_activations))

    @property
    def list_activations(self):
        return [self.activations[l] for l in self.layers]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = []

    def register_hook(
        self,
        layer: torch.nn.Module,
        save_activations: SaveActivationsFunc = default_save_activations,
    ):
        def hook(module, inputs, outputs):
            self.activations[layer] = save_activations(outputs).detach()
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
    model_forward: ModelForwardFunc,
    layers_activations: list[torch.nn.Module],
    batch_size=32,
    save_activations: SaveActivationsFunc = default_save_activations,
) -> list[torch.Tensor]:
    act_col = []
    with ActivationTracker(layers_activations, save_activations) as tracker:
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
    model_forward: ModelForwardFunc,
    layers_activations: list[torch.nn.Module],
    batch_size=32,
    save_activations: SaveActivationsFunc = default_save_activations,
) -> list[torch.Tensor]:
    N = len(data)
    assert N > 0, "Must have data!"
    assert len(layers_activations) > 0, "Must have layers we take activations from!"
    assert batch_size > 0, "batch size must be greater than 0"

    accumulate = [None] * len(layers_activations)  # we will be accumulating the tensors
    with ActivationTracker(layers_activations, save_activations) as tracker:
        for i in tqdm(range(0, N, batch_size)):
            batch = data[i : i + batch_size]
            model_forward(batch)
            mean_accumulate_tensors(
                accumulator=accumulate, tensors=tracker.list_activations, total_length=N
            )

    return accumulate


def compute_task_activations(
    df: pd.DataFrame,
    model_forward: ModelForwardFunc,
    layers_activations: list[torch.nn.Module],
    batch_size=32,
    save_activations: SaveActivationsFunc = default_save_activations,
):
    assert "data" in df.columns, "Must have a column named data"
    assert "positive" in df.columns, (
        "Must have a column named 'positive' which is either True (task) or False (control)"
    )

    task = df[df["positive"] == True]
    control = df[df["positive"] == False]

    task_acts = accumulate_activations(
        task["data"], model_forward, layers_activations, batch_size, save_activations
    )
    control_acts = accumulate_activations(
        control["data"], model_forward, layers_activations, batch_size, save_activations
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
    if grid is None:
        # first combine all the layers so can do argpartition
        for i, a in enumerate(activations):
            plt.title(f"Layer {i}")
            plt.imshow(squarify(a), cmap=cmap, aspect="auto")
            plt.colorbar()
            plt.show()
    else:
        fig, axes = plt.subplots(*grid, figsize=(16, 9))
        fig.suptitle("Absolute activations")
        for i, ax in enumerate(axes.flat):
            a = activations[i]
            im = ax.imshow(squarify(a), cmap=cmap, aspect="auto")
            ax.set_title(f"Layer {i}")
            plt.colorbar(im, ax=ax)
            no_ticks(ax)
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


def format_topk_global_return(tensors: list[torch.Tensor], layers_idxs: torch.Tensor):
    result_idxs = [[] for _ in range(len(tensors))]
    result_values = [[] for _ in range(len(tensors))]
    for layer_idx, layer_flat_idx in layers_idxs:
        t = tensors[layer_idx]
        result_idxs[layer_idx].append(layer_flat_idx.item())
        result_values[layer_idx].append(t.view(-1)[layer_flat_idx].item())
    return result_idxs, result_values


def global_to_layers(tensors, global_idxs):
    edges = torch.tensor([0] + [prod(t.shape) for t in tensors]).cumsum(0)
    layers_idxs = flat_idx_to_layer_idx_fast(global_idxs, edges)
    return layers_idxs


def topk_global(tensors: list[torch.Tensor], k: int):
    # flatten all the tensors, find the topk, then map those flat idxs back to the layers indices
    flat_concat = torch.hstack([t.flatten() for t in tensors])
    global_values, global_idxs = flat_concat.topk(k)
    layers_idxs = global_to_layers(tensors, global_idxs)
    return format_topk_global_return(tensors, layers_idxs)


def percent_to_k(tensors, percent):
    total_length = sum(prod(t.shape) for t in tensors)
    k = int((percent / 100) * total_length)
    return k


def top_percent_global(tensors: list[torch.Tensor], percent=1):
    k = percent_to_k(tensors, percent)
    return topk_global(tensors, k)


def no_ticks(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_top_activation(ax, top_idxs, top_values, activation, title=""):
    canvas = torch.zeros(activation.shape)
    if len(top_idxs) > 0:
        act = torch.tensor(top_values)
        idxs = torch.tensor(top_idxs)
        canvas.view(-1)[idxs] = act
    ax.set_title(title)
    no_ticks(ax)
    im = ax.imshow(squarify(canvas), cmap="inferno", aspect="auto")
    plt.colorbar(im, ax=ax)
    return ax


def visualize_top_activations(
    top_idxs, top_values, activations, grid=(4, 4), title="Top % activations showing"
):
    fig, axes = plt.subplots(*grid, figsize=(16, 9))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        a = activations[i]
        top_idx = top_idxs[i]
        top_value = top_values[i]
        visualize_top_activation(ax, top_idx, top_value, a, title=f"Layer {i}")
    plt.show()


def visualize_top_per_layer(
    top_idxs, activations, title="Percentage Top activations per layer"
):
    import seaborn as sns
    import numpy as np

    total_lengths = [prod(a.shape) for a in activations]
    percentages = (
        np.array([len(top_idxs[i]) / l for i, l in enumerate(total_lengths)]).reshape(
            (-1, 1)
        )
        * 100
    )
    labels = np.array([f"{p[0]:.2f}%" for p in percentages]).reshape(percentages.shape)

    plt.figure(figsize=(4, 6))
    ax = sns.heatmap(
        percentages,
        cmap="inferno",
        annot=labels,
        annot_kws={"fontsize": 10},
        fmt="s",
        linecolor="white",
        linewidths=1,
    )
    ax.set(
        title=title,
        xticklabels=[],
        xticks=[],
        ylabel="Layers",
    )

    plt.show()


def default_flat_idxs_ablate(
    activations: torch.Tensor, ablate: AblateIdxs, scalar: float
):
    B = activations.shape[0]
    flat = activations.view(B, -1)
    flat[:, ablate] *= scalar
    return activations


class AblateTorchModel:
    def __init__(
        self,
        layers: list[torch.nn.Module],
        to_ablate: list[AblateIdxs],
        scalar=0,
        ablate_activations: AblateActivationsFunc = default_flat_idxs_ablate,
    ):
        self.layers = layers
        self.to_ablate = to_ablate
        self.hooks = []
        for layer, ablate in zip(layers, to_ablate):
            if len(ablate) > 0:
                self.hooks.append(
                    self.register_hook(layer, ablate, scalar, ablate_activations)
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_hook(
        self,
        layer: torch.nn.Module,
        ablate: AblateIdxs,
        scalar: float,
        ablate_activations: AblateActivationsFunc,
    ):
        def hook(module, inputs, outputs):
            return ablate_activations(outputs, ablate, scalar)

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
    model_forward: ModelForwardFunc,
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
    model_forward: ModelForwardFunc,
    layers_activations: list[torch.nn.Module],
    to_ablate: list[AblateIdxs],
    batch_size=32,
    ablate_factor=0,
    ablate_activations: AblateActivationsFunc = default_flat_idxs_ablate,
) -> torch.Tensor | tuple:
    with AblateTorchModel(
        layers_activations, to_ablate, ablate_factor, ablate_activations
    ):
        return regular_inference(data, model_forward, batch_size)


def task_ablated(
    df: pd.DataFrame,
    model_forward: ModelForwardFunc,
    layers_activations: list[torch.nn.Module],
    to_ablate: list[AblateIdxs],
    batch_size=32,
    ablate_factor=0,
    ablate_activations: AblateActivationsFunc = default_flat_idxs_ablate,
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
        ablate_activations,
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
        ablate_activations,
    )
    regular_control = regular_inference(
        control["data"],
        model_forward,
        batch_size,
    )

    return (ablated_task, regular_task), (ablated_control, regular_control)


def load_task(filename: str):
    assert os.path.exists(filename), "task file must exist"

    df = pd.read_parquet(filename)
    task = df[df["validation"] == False]
    validation = df[df["validation"] == True]
    return task, validation


class DeepLocalizer:
    def __init__(
        self,
        task: pd.DataFrame,
        layers_activations: list[torch.nn.Module],
        model_forward: ModelForwardFunc,
        save_activations_func: SaveActivationsFunc = default_save_activations,
        ablate_activations_func: AblateActivationsFunc = default_flat_idxs_ablate,
        ablate_factor: float = 0.0,
        batch_size=32,
    ):
        self.layers_activations = layers_activations
        self.task = task
        self.model_forward = model_forward
        self.save_activations_func = save_activations_func
        self.ablate_activations_func = ablate_activations_func
        self.ablate_factor = ablate_factor
        self.batch_size = batch_size
        self.activations = None

    def load_activations(self, filename, device="cpu"):
        self.activations = load_activations_from_disk(filename, device)
        return self

    def compute_activations(self):
        print("[DeepLocalizer] Computing Activations")
        self.activations = compute_task_activations(
            df=self.task,
            model_forward=self.model_forward,
            layers_activations=self.layers_activations,
            batch_size=self.batch_size,
        )
        return self

    def assert_activations(self):
        assert self.activations, (
            "Must compute_activations() or load_activations(filename) first"
        )

    def save_activations(self, filename):
        self.assert_activations()
        save_activations_to_disk(self.activations, filename)

    @torch.no_grad()
    def regular_model_forward(self, df: pd.DataFrame = None):
        self.assert_activations()

        if df is None:
            df = self.task

        positive = df[df["positive"] == True]
        control = df[df["positive"] == False]

        print("[DeepLocalizer] Computing Model Forward")
        return regular_inference(
            positive["data"],
            self.model_forward,
            self.batch_size,
        ), regular_inference(
            control["data"],
            self.model_forward,
            self.batch_size,
        )

    def top_percent_activations(self, top_percent: float, transform=lambda x: x.abs()):
        self.assert_activations()
        return top_percent_global(_map(transform, self.activations), top_percent)

    @torch.no_grad()
    def ablate_model_forward(
        self,
        ablate_activations: list[AblateIdxs],
        df: pd.DataFrame = None,
    ):
        self.assert_activations()

        if df is None:
            df = self.task

        print("[DeepLocalizer] Computing Model Forward ABLATED")
        with AblateTorchModel(
            layers=self.layers_activations,
            to_ablate=ablate_activations,
            scalar=self.ablate_factor,
            ablate_activations=self.ablate_activations_func,
        ):
            return self.regular_model_forward(df)


def percent_label_stayed_same(before_preds, after_preds):
    return (100 * (before_preds == after_preds).sum() / len(before_preds)).item()


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    np.random.seed(0)  # for reproducibility since I use df.sample() from numpy

    CACHED_ACTIVATIONS = "./face_data/task_face_resnet34_acts.safetensors"
    activations = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VIS = False

    from transformers import AutoImageProcessor, ResNetForImageClassification

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-34")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-34")
    model = model.to(DEVICE)
    print("*Loaded Resnet34 from Huggingface")

    # DEFINE HOW THE MODEL COMPUTES ACTIVATIONS
    @torch.no_grad()
    def resnet_forward(image_paths):
        images = [Image.open(f"./face_data/{p}").convert("RGB") for p in image_paths]
        inputs = processor(images, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        return outputs.logits

    resnet_blocks = [
        layer for stage in model.resnet.encoder.stages for layer in stage.layers
    ]

    task, valid = load_task("./face_data/task_face_localizer.parquet")
    valid = valid.sample(32 * 1)
    print("*Loaded Face Localizer Task")

    d = DeepLocalizer(
        task=task,
        layers_activations=resnet_blocks,
        model_forward=resnet_forward,
    )
    if os.path.exists(CACHED_ACTIVATIONS):
        d.load_activations(CACHED_ACTIVATIONS, DEVICE)
    else:
        d.compute_activations()
        d.save_activations(CACHED_ACTIVATIONS)

    ps = [1, 0.5, 0.25, 0.125, 0.0625]
    perf_task = []
    perf_control = []

    regular_task, regular_control = d.regular_model_forward(valid)

    for p in ps:
        print(p)
        top_indices, top_values = d.top_percent_activations(p)

        if VIS:
            a = [a.abs() for a in d.activations]
            visualize_top_per_layer(top_indices, a)
            visualize_activations(a, (4, 4), cmap="inferno")
            visualize_top_activations(top_indices, top_values, a, (4, 4))

        ablated_task, ablated_control = d.ablate_model_forward(
            df=valid, ablate_activations=top_indices
        )

        perf_on_faces = percent_label_stayed_same(
            before_preds=regular_task.argmax(-1), after_preds=ablated_task.argmax(-1)
        )
        perf_task.append(perf_on_faces)
        print(
            f"[After Ablation] {perf_on_faces:.3f}% predictions stayed the same for face images"
        )

        perf_on_objects = percent_label_stayed_same(
            before_preds=regular_control.argmax(-1),
            after_preds=ablated_control.argmax(-1),
        )
        perf_control.append(perf_on_objects)

        print(
            f"[After Ablation] {perf_on_objects:.3f}% predictions stayed the same for object images"
        )

    plt.scatter(ps, perf_task, label="[FACES] % labels same")
    plt.scatter(ps, perf_control, label="[OBJECTS] % labels same")
    frac = lambda n, d: r"$\frac{" + str(n) + "}" + "{" + str(d) + "}$"
    plt.xticks(ps, labels=["$1$", frac(1, 2), frac(1, 4), frac(1, 8), frac(1, 16)])
    plt.xlabel("Top % activations ablated (candidate face networks)")
    plt.ylabel("% prediction stayed the same after ablation")
    plt.legend()
    plt.show()
