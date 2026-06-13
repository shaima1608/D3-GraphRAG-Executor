from functools import lru_cache
from typing import Dict, List

import numpy as np
from pymongo import MongoClient, ASCENDING
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _mongo_client() -> MongoClient:
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)


def get_mongo():
    """Return a cached Mongo database connection.

    Caching avoids reconnecting to MongoDB for every /ask or evaluation query,
    which was one of the main causes of high D3 latency.
    """
    client = _mongo_client()
    client.admin.command('ping')
    db = client[settings.mongo_db]
    db[settings.mongo_docs_collection].create_index([('paper_id', ASCENDING)], unique=True)
    db[settings.mongo_chunks_collection].create_index([('chunk_id', ASCENDING)], unique=True)
    db[settings.mongo_chunks_collection].create_index([('paper_id', ASCENDING)])
    return db


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    Before this optimization, the model could be loaded repeatedly inside search,
    vector-only evaluation, hybrid evaluation, and GraphRAG calls. Reusing it is
    the biggest latency improvement.
    """
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Return a cached Qdrant client."""
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def reset_qdrant_collection(client: QdrantClient, vector_size: int):
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def seed_mongo(documents: List[Dict], chunks: List[Dict]):
    db = get_mongo()
    db[settings.mongo_docs_collection].delete_many({})
    db[settings.mongo_chunks_collection].delete_many({})
    if documents:
        db[settings.mongo_docs_collection].insert_many(documents)
    if chunks:
        db[settings.mongo_chunks_collection].insert_many(chunks)
    return {'documents': len(documents), 'chunks': len(chunks)}


def seed_qdrant(chunks: List[Dict], batch_size: int = 64):
    model = load_embedding_model()
    sample = model.encode(['dimension probe'], normalize_embeddings=True)
    vector_size = int(sample.shape[1])
    client = get_qdrant()
    reset_qdrant_collection(client, vector_size)
    points = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = model.encode([x['text'] for x in batch], normalize_embeddings=True, show_progress_bar=False)
        for local_i, (chunk, vector) in enumerate(zip(batch, vectors)):
            payload = {k: v for k, v in chunk.items() if k != 'text'}
            payload['text'] = chunk['text'][:1500]
            points.append(PointStruct(id=start + local_i, vector=vector.tolist(), payload=payload))
        if len(points) >= 256:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
            points = []
    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
    return {'qdrant_points': len(chunks), 'vector_size': vector_size}
