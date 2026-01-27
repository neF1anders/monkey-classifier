import sys
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ..inference.embedding import get_backbone, image_to_vector
from ..inference.predict import predict

COLLECTION = "monkeys"

client = QdrantClient(host="qdrant", port=6333)


def search(image_path):
    # 1. predict class
    pred_class = predict(image_path)

    # 2. embedding
    with open("classes.json") as f:
        classes = json.load(f)

    backbone = get_backbone(len(classes))
    vec = image_to_vector(image_path, backbone)

    # 3. search inside this class only
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="class",
                    match=MatchValue(value=pred_class)
                )
            ]
        )
    ).points


    results = [hit.payload["path"] for hit in hits]

    return pred_class, results


if __name__ == "__main__":
    cls, imgs = search(sys.argv[1])
    print("Predicted:", cls)
    print("Similar images:")
    for p in imgs:
        print(p)
