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

        # Fetch HTML content
        html = fetch_html(url=url, timeout=30)  # Uncomment for live fetch
        # html = request.data.get("text")  # For testing

        blocks = extract_text_blocks(html)
        chunks = chunk_text_by_token_limit(blocks, max_tokens=500, overlap=50)

        embeddings = embed_texts(chunks)
        dim = 768  # Must match embedder dim

        collection_name = "html_chunks_v1"
        coll = ensure_collection(collection_name, dim=dim)

        # Insert unique chunks
        insert_chunks(coll, embeddings, chunks, url)

        # Embed query
        q_emb = embed_texts([query])[0]

        # Search top 10
        hits = search_collection(coll, q_emb, top_k=10)

        return Response(data={
            "status": 1,
            "details": {
                "url": url,
                "query": query,
                "results": hits
            }
        }, status=status.HTTP_200_OK)
        # except Exception as e:
        #     return Response(data={
        #         "status": 0,
        #         "details": f"failed to fetch: {e}"
        #     }, status=status.HTTP_400_BAD_REQUEST)