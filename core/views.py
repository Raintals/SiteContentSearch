from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .api_utils import *
from .milvus_utils import *
import uuid


class SearchAPIView(APIView):
    def post(self, request):
        url = request.data.get("url")
        query = request.data.get("query")
        if not url or not query:
            return Response(data={"status": 0, "details": "url and query required"},
                            status=status.HTTP_400_BAD_REQUEST)

        html = fetch_html(url=url, timeout=30)
        blocks = extract_text_blocks(html)
        chunks = chunk_text_by_token_limit(blocks, max_tokens=500, overlap=50)

        embeddings = embed_texts([chunk["text"] for chunk in chunks])
        dim = 768

        collection_name = "html_chunks_v1"
        coll = ensure_collection(collection_name, dim=dim)

        insert_chunks(coll, embeddings, chunks, url)

        q_emb = embed_texts([query])[0]

        hits = search_collection(coll, q_emb, top_k=10)

        return Response(data={
            "status": 1,
            "details": {
                "url": url,
                "query": query,
                "results": hits
            }
        }, status=status.HTTP_200_OK)