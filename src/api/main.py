from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_pipeline
from api.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Vietnam Legal RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    try:
        pipeline = get_pipeline()

        result = pipeline.run(
            query=query,
            retrieval_top_k=30,
            rerank_top_k=10,
        )

        return ChatResponse(
            query=result.query,
            normalized_query=result.normalized_query,
            retrieved_count=result.retrieved_count,
            reranked_count=result.reranked_count,
            selected_document_count=(
                result.final_context.document_count
            ),
            selected_chunk_count=(
                result.final_context.deduplicated_chunk_count
            ),
            answer=result.answer,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc