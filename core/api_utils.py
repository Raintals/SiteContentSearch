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
            blocks.append((text, str(tag)))  # Return both text and HTML
    return blocks


def chunk_text_by_token_limit(blocks, max_tokens=500, overlap=50):
    """
    Chunk text blocks using sliding window with overlap.
    Each block is a (text, html) tuple.
    """
    chunks = []
    current_text = []
    current_html = []
    current_len = 0
    for block_text, block_html in blocks:
        block_len = len(tokenizer.encode(block_text))
        if current_len + block_len <= max_tokens:
            current_text.append(block_text)
            current_html.append(block_html)
            current_len += block_len
        else:
            if current_text:
                chunks.append({
                    "text": " ".join(current_text),
                    "html": "".join(current_html)
                })
            # start new chunk with overlap
            if overlap < len(current_text):
                current_text = current_text[-overlap:] + [block_text]
                current_html = current_html[-overlap:] + [block_html]
            else:
                current_text = [block_text]
                current_html = [block_html]
            current_len = sum(len(tokenizer.encode(x)) for x in current_text)
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "html": "".join(current_html)
        })
    return chunks


def embed_texts(texts, batch_size=32):
    """
    Return normalized embeddings as float32 numpy array.
    """
    embs = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")
