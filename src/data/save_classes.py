from pathlib import Path
import json

root = Path("data/processed/ready")
classes = sorted([p.name for p in root.iterdir() if p.is_dir()])

with open("classes.json", "w") as f:
    json.dump(classes, f)

print("Saved classes:", classes)
