from src.config import settings
from src.ingestion import build_corpus
from src.search import build_bm25_index
from src.stores import seed_mongo, seed_qdrant

if __name__ == '__main__':
    docs, chunks = build_corpus(settings.pdf_dir, settings.chunk_size, settings.chunk_overlap)
    print('Built corpus:', len(docs), 'documents and', len(chunks), 'chunks')
    print('Mongo:', seed_mongo(docs, chunks))
    print('Qdrant:', seed_qdrant(chunks))
    print('BM25 index:', build_bm25_index(chunks))
