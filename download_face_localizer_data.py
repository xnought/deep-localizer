import pandas as pd
import asyncio
import os
import requests
from datasets import load_dataset
from tqdm.notebook import tqdm


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


async def download_celeba_subset():
    ds = load_dataset("tpremoli/CelebA-attrs", split="test", streaming=True)

    i = 0
    names = []
    if not os.path.exists("./data/celeba"):
        os.mkdir("./data/celeba")
    for d in tqdm(ds.take(5_000), total=5_000):
        n = f"celeba/{i}.jpg"
        d["image"].save("./data/" + n)
        names.append(n)
        i += 1

    pos_df = pd.DataFrame({"data": names, "positive": [True] * 5_000})
    return pos_df


async def main():
    coco_df = await download_coco_subset()
    celeba_df = await download_celeba_subset()
    df = pd.concat((celeba_df, coco_df))
    df.to_parquet("./data/tasks/face_task.parquet", index=False)


if __name__ == "__main__":
    main()
