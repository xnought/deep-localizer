import pandas as pd
import asyncio
import os
import requests
from datasets import load_dataset
import numpy as np

np.random.seed(0)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def download_image(url, name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(name, "wb") as f:
            f.write(response.content)
        print(f"Saved {name}")


async def download_coco_subset():
    splits = {
        "train": "data/train-00000-of-00001-0084e041f1902997.parquet",
        "validation": "data/validation-00000-of-00001-e3c37e369512a3aa.parquet",
    }
    df = pd.read_parquet("hf://datasets/phiyodr/coco2017/" + splits["validation"])
    if not os.path.exists("./data/val2017"):
        os.mkdir("./data/val2017")
    tasks = []
    for i, row in df.iterrows():
        tasks.append(download_image(row["coco_url"], "./data/" + row["file_name"]))
    await asyncio.gather(*tasks)
    neg_df = pd.DataFrame(
        {"data": df["file_name"].tolist(), "positive": [False] * len(df)}
    )
    return neg_df


def download_celeba_subset():
    ds = load_dataset("tpremoli/CelebA-attrs", split="test", streaming=True)

    i = 0
    names = []
    if not os.path.exists("./data/celeba"):
        os.mkdir("./data/celeba")
    for d in ds.take(5_000):
        n = f"celeba/{i}.jpg"
        d["image"].save("./data/" + n)
        print(f"Saved {'./data/' + n}")
        names.append(n)
        i += 1

    pos_df = pd.DataFrame({"data": names, "positive": [True] * 5_000})
    return pos_df


def sample(df, n):
    return df.loc[np.random.choice(df.index.values, n, replace=False)]


async def main():
    if not os.path.exists("./data"):
        os.mkdir("data")

    celeba_df = download_celeba_subset()
    coco_df = await download_coco_subset()

    N = 3_000
    task_df = sample(celeba_df, N)
    control_df = sample(coco_df, N)

    # Label 1k with validation, and rest as regular
    task_df["validation"] = [i < 1_000 for i in range(N)]
    control_df["validation"] = [i < 1_000 for i in range(N)]

    df = pd.concat((task_df, control_df))
    df.to_parquet("./data/task_face_localizer.parquet", index=False)


if __name__ == "__main__":
    asyncio.run(main())
