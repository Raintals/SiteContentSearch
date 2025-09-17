from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import time

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)


def ensure_collection(collection_name, dim=768):
    """
    Create a Milvus collection if it doesn't exist.
    """
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        coll.load()
        return coll

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="HTML chunks collection")
    coll = Collection(name=collection_name, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",  # Use inner product with normalized embeddings
        "params": {"M": 8, "efConstruction": 200}
    }
    coll.create_index(field_name="embedding", index_params=index_params)
    coll.load()
    return coll


def insert_chunks(collection, embeddings: np.ndarray, texts, url):
    """
    Insert unique chunks into Milvus collection.
    Prevent duplicates based on text.
    """
    # Get existing texts to avoid duplicates
    try:
        existing_texts = set(e["text"] for e in collection.query(expr="text != ''", output_fields=["text"]))
    except Exception:
        existing_texts = set()

    filtered_texts = []
    filtered_embeddings = []
    filtered_chunk_indices = []

    for i, t in enumerate(texts):
        if t not in existing_texts:
            filtered_texts.append(t)
            filtered_embeddings.append(embeddings[i])
            filtered_chunk_indices.append(i)

    if not filtered_texts:
        return []

    n = len(filtered_texts)
    pks = list(range(int(time.time()*1000), int(time.time()*1000) + n))

    collection.insert([pks, np.array(filtered_embeddings).tolist(), filtered_texts, [url]*n, filtered_chunk_indices])
    collection.load()
    return pks


def search_collection(collection, query_emb, top_k=10):
    """
    Search Milvus collection using query embedding and return unique top-k hits.
    """
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    res = collection.search([query_emb.tolist()], "embedding", param=search_params, limit=top_k*3,
                            output_fields=["text", "chunk_index"])

    seen_texts = set()
    hits = []
    for hits_list in res:
        for h in hits_list:
            text = h.entity.get("text")
            if text not in seen_texts:
                hits.append({
                    "score": float(h.distance),
                    "text": text,
                    "chunk_index": h.entity.get("chunk_index")
                })
                seen_texts.add(text)
            if len(hits) >= top_k:
                break
        if len(hits) >= top_k:
            break
    return hits
