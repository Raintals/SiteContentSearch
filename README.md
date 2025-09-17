# SiteContentSearch Backend

This project provides a backend API for searching and indexing web page content using Milvus vector database and transformer-based embeddings.

## Features

- Fetches and parses HTML content from a given URL.
- Extracts and chunks text blocks from HTML.
- Embeds text using Sentence Transformers.
- Stores and searches embeddings in Milvus for semantic search.
- REST API endpoint for search queries.

## Project Structure

- [`core/api_utils.py`](core/api_utils.py): Utilities for fetching HTML, extracting text, chunking, and embedding.
- [`core/milvus_utils.py`](core/milvus_utils.py): Functions for managing Milvus collections, inserting chunks, and searching.
- [`core/views.py`](core/views.py): Django REST API view for handling search requests.

## Requirements

- Python 3.8+
- Django
- djangorestframework
- pymilvus
- sentence-transformers
- transformers
- numpy
- requests
- beautifulsoup4
- lxml
- Milvus (running locally or via Docker)

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd SiteContentSearch
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment.

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```sh
pip install django djangorestframework pymilvus sentence-transformers transformers numpy requests beautifulsoup4 lxml
```

### 3. Start Milvus

You can use Docker Compose. Example snippet from [`docker-compose.yml`](docker-compose.yml):

```sh
docker-compose up -d
```

Ensure Milvus is running on `localhost:19530`.

### 4. Run Django Migrations

```sh
python manage.py migrate
```

### 5. Start the Django Server

```sh
python manage.py runserver
```

## API Usage

### Search API

**Endpoint:**  
`POST /api/search/`

**Request Body:**

```json
{
  "url": "https://example.com",
  "query": "your search query"
}
```

**Response:**

```json
{
  "status": 1,
  "details": {
    "url": "https://example.com",
    "query": "your search query",
    "results": [
      {
        "score": 0.92,
        "text": "...",
        "html": "...",
        "chunk_index": 0
      },
      ...
    ]
  }
}
```

## Code Overview

- [`core/api_utils.py`](core/api_utils.py):  
  - `fetch_html(url, timeout)`: Fetches HTML content.
  - `extract_text_blocks(html)`: Extracts text and HTML blocks.
  - `chunk_text_by_token_limit(blocks, max_tokens, overlap)`: Chunks text for embedding.
  - `embed_texts(texts)`: Returns normalized embeddings.

- [`core/milvus_utils.py`](core/milvus_utils.py):  
  - `ensure_collection(collection_name, dim)`: Ensures Milvus collection exists.
  - `insert_chunks(collection, embeddings, chunks, url)`: Inserts unique chunks.
  - `search_collection(collection, query_emb, top_k)`: Searches for similar chunks.

- [`core/views.py`](core/views.py):  
  - `SearchAPIView`: Handles `/api/search/` POST requests.

## Notes

- Make sure Milvus is running before starting the Django server.
- The embedding model is loaded at startup for efficiency.
- For production, configure allowed hosts, security settings, and persistent Milvus storage.

---

For any issues, please open an issue or contact the maintainer.