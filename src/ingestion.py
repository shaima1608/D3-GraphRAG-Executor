import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]


def read_pdf_pages(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text())
        except Exception:
            text = ""
        if text:
            pages.append({'page': i, 'text': text})
    return pages


def guess_title(pages: List[Dict], filename: str) -> str:
    if not pages:
        return Path(filename).stem
    first = pages[0]['text']
    # Prefer the text before Abstract if it is not too long.
    m = re.search(r"\babstract\b", first, flags=re.I)
    candidate = first[:m.start()].strip() if m else first[:220]
    candidate = re.sub(r"[^A-Za-z0-9 ,:;\-()]+", " ", candidate)
    candidate = clean_text(candidate)
    return candidate[:180] or Path(filename).stem


def guess_authors(pages: List[Dict]) -> List[str]:
    # Conservative fallback because PDF author extraction is noisy.
    if not pages:
        return ['Unknown Author']
    first = pages[0]['text'][:1200]
    # Common pattern: author names after title and before abstract.
    m = re.search(r"\babstract\b", first, flags=re.I)
    before_abs = first[:m.start()] if m else first[:500]
    # Split possible author line by commas; remove emails/affiliations.
    before_abs = re.sub(r"\S+@\S+", " ", before_abs)
    chunks = [clean_text(x) for x in re.split(r",|;| and ", before_abs)]
    authors = []
    for x in chunks[1:6]:
        if 2 <= len(x.split()) <= 5 and not re.search(r"university|department|abstract|school", x, re.I):
            authors.append(x[:80])
    return authors or ['Unknown Author']


def infer_topics(texts: List[str], filenames: List[str], max_topics: int = 3) -> Dict[str, List[str]]:
    if not texts:
        return {}
    n_features = min(1500, max(10, sum(len(t.split()) for t in texts)))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    out = {}
    for i, name in enumerate(filenames):
        row = X[i].toarray().ravel()
        if row.max() == 0:
            out[name] = ['general ai']
        else:
            top = terms[row.argsort()[-max_topics:][::-1]].tolist()
            out[name] = [t.replace('_', ' ') for t in top]
    return out


def chunk_pages(pages: List[Dict], chunk_size: int = 900, overlap: int = 180) -> List[Dict]:
    chunks = []
    step = max(1, chunk_size - overlap)
    for p in pages:
        text = p['text']
        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]
            if len(chunk_text.strip()) < 80:
                continue
            chunks.append({'text': chunk_text, 'page_start': p['page'], 'page_end': p['page']})
    return chunks


def build_corpus(pdf_dir: str, chunk_size: int = 900, overlap: int = 180) -> Tuple[List[Dict], List[Dict]]:
    pdf_paths = sorted(Path(pdf_dir).glob('*.pdf'))
    if not pdf_paths:
        raise FileNotFoundError(f'No PDF files found in {pdf_dir}. Put your D1 PDF corpus there.')

    paper_texts, paper_files, raw_pages = [], [], {}
    for path in pdf_paths:
        pages = read_pdf_pages(str(path))
        raw_pages[path.name] = pages
        paper_texts.append(' '.join(p['text'] for p in pages[:3])[:5000])
        paper_files.append(path.name)

    topics_by_file = infer_topics(paper_texts, paper_files)
    documents, chunks = [], []

    for path in pdf_paths:
        pages = raw_pages[path.name]
        paper_id = stable_id(path.name)
        title = guess_title(pages, path.name)
        authors = guess_authors(pages)
        topics = topics_by_file.get(path.name, ['general ai'])
        doc = {
            'paper_id': paper_id,
            'title': title,
            'authors': authors,
            'venue': 'Unknown Venue',
            'year': None,
            'doi': None,
            'pdf_path': str(path),
            'filename': path.name,
            'topics': topics,
            'num_pages': len(pages),
            'provenance': {'source': 'local_pdf', 'path': str(path)},
        }
        documents.append(doc)
        for j, ch in enumerate(chunk_pages(pages, chunk_size, overlap)):
            chunk_id = f'{paper_id}_{j:04d}'
            chunks.append({
                'chunk_id': chunk_id,
                'paper_id': paper_id,
                'title': title,
                'filename': path.name,
                'text': ch['text'],
                'page_start': ch['page_start'],
                'page_end': ch['page_end'],
                'topics': topics,
                'authors': authors,
                'citation': f"{path.name}, p. {ch['page_start']}",
                'provenance': {'source': 'local_pdf', 'path': str(path), 'page': ch['page_start']},
            })
    return documents, chunks
