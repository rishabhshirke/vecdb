import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from vectordb import VectorDB
from api.routes import collections, vectors


PERSIST_DIR = os.getenv("VECDB_PERSIST_DIR", "./data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = VectorDB(persist_dir=PERSIST_DIR)
    try:
        db.load()
        print(f"Loaded existing DB from {PERSIST_DIR}")
    except FileNotFoundError:
        print("Starting fresh DB (no saved state found)")
    app.state.db = db
    yield
    try:
        db.save()
        print(f"DB saved to {PERSIST_DIR}")
    except Exception:
        pass  # Vercel serverless has no writable disk — skip silently


app = FastAPI(
    title="VecDB",
    description="A minimal vector database with HNSW indexing",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(collections.router)
app.include_router(vectors.router)


@app.get("/")
def root():
    return {"name": "VecDB", "version": "0.1.0", "status": "running"}


@app.post("/save")
def manual_save(request: Request):
    request.app.state.db.save()
    return {"message": "DB saved"}
