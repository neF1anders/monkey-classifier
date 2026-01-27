import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json

from ..inference.embedding import get_backbone, image_to_vector

DATA_ROOT = Path("data/processed/ready")
COLLECTION = "monkeys"

client = QdrantClient(host="qdrant", port=6333)

def main():
    with open("classes.json") as f:
        classes = json.load(f)

    backbone = get_backbone(len(classes))

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    points = []
    idx = 0

    for class_name in classes:
        for img_path in (DATA_ROOT / class_name).glob("*.jpg"):
            vec = image_to_vector(str(img_path), backbone)

            points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload={
                        "class": class_name,
                        "path": str(img_path)
                    }
                )
            )
            idx += 1

            if len(points) == 64:
                client.upsert(collection_name=COLLECTION, points=points)
                points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print("Indexing complete!")


if __name__ == "__main__":
    main()

