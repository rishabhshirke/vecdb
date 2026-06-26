from fastapi import APIRouter, HTTPException, Request
from api.models import CreateCollectionRequest

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/", status_code=201)
def create_collection(body: CreateCollectionRequest, request: Request):
    db = request.app.state.db
    try:
        db.create_collection(body.name, body.dim, body.metric, body.index_type)
        return {"message": f"Collection '{body.name}' created.", "dim": body.dim, "metric": body.metric}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/")
def list_collections(request: Request):
    return request.app.state.db.list_collections()


@router.delete("/{name}", status_code=204)
def delete_collection(name: str, request: Request):
    try:
        request.app.state.db.delete_collection(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
