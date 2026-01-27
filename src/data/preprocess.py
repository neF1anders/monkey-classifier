import os
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "data/processed/images"
OUTPUT_DIR = "data/processed/ready"
IMG_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(in_path, out_path):
    try:
        img = Image.open(in_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(out_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


for class_name in tqdm(os.listdir(INPUT_DIR)):
    class_in_dir = os.path.join(INPUT_DIR, class_name)
    class_out_dir = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(class_in_dir):
        continue

    os.makedirs(class_out_dir, exist_ok=True)

    for fname in os.listdir(class_in_dir):
        in_path = os.path.join(class_in_dir, fname)
        out_path = os.path.join(class_out_dir, fname)

        process_image(in_path, out_path)
