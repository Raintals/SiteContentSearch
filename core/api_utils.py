import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

USER_AGENT = "Mozilla/5.0 (compatible; Bot/1.0)"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Initialize tokenizer and embedder once
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def fetch_html(url, timeout=10):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url=url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.text


def extract_text_blocks(html):
    soup = BeautifulSoup(html, "lxml")
    for s in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        s.decompose()
    blocks = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append(text)
    return blocks


def chunk_text_by_token_limit(blocks, max_tokens=500, overlap=50):
    """
    Chunk text blocks using sliding window with overlap.
    """
    chunks = []
    current = []
    current_len = 0
    for block in blocks:
        block_len = len(tokenizer.encode(block))
        if current_len + block_len <= max_tokens:
            current.append(block)
            current_len += block_len
        else:
            if current:
                chunks.append(" ".join(current))
            # start new chunk with overlap
            if overlap < len(current):
                current = current[-overlap:] + [block]
            else:
                current = [block]
            current_len = sum(len(tokenizer.encode(x)) for x in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed_texts(texts, batch_size=32):
    """
    Return normalized embeddings as float32 numpy array.
    """
    embs = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")
