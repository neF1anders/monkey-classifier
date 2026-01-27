import os
import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = "data/raw/monkeys.csv"
OUT_DIR = "data/processed/images"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row["image_url"]
    label = row["common_name"]

    try:
        img_data = requests.get(url, timeout=10).content
        class_dir = os.path.join(OUT_DIR, label.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{idx}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_data)
    except Exception:
        continue
