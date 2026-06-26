from fastapi import APIRouter, HTTPException, Request
from api.models import InsertRequest, InsertManyRequest, UpdateRequest, SearchRequest

router = APIRouter(prefix="/collections/{collection_name}", tags=["vectors"])


def _col(request: Request, collection_name: str):
    try:
        return request.app.state.db.get_collection(collection_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/vectors", status_code=201)
def insert_vector(collection_name: str, body: InsertRequest, request: Request):
    col = _col(request, collection_name)
    try:
        col.insert(body.id, body.vector, body.metadata)
        return {"message": f"Inserted '{body.id}'"}
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vectors/batch", status_code=201)
def insert_many(collection_name: str, body: InsertManyRequest, request: Request):
    col = _col(request, collection_name)
    errors = []
    inserted = 0
    for item in body.vectors:
        try:
            col.insert(item.id, item.vector, item.metadata)
            inserted += 1
        except Exception as e:
            errors.append({"id": item.id, "error": str(e)})
    return {"inserted": inserted, "errors": errors}


@router.get("/vectors/{id}")
def get_vector(collection_name: str, id: str, request: Request):
    col = _col(request, collection_name)
    try:
        return col.get(id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/vectors/{id}")
def update_vector(collection_name: str, id: str, body: UpdateRequest, request: Request):
    col = _col(request, collection_name)
    try:
        col.update(id, body.vector, body.metadata)
        return {"message": f"Updated '{id}'"}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/vectors/{id}", status_code=204)
def delete_vector(collection_name: str, id: str, request: Request):
    col = _col(request, collection_name)
    try:
        col.delete(id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/search")
def search(collection_name: str, body: SearchRequest, request: Request):
    col = _col(request, collection_name)
    try:
        results = col.search(body.vector, body.k, body.filters, body.include_vector)
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/count")
def count(collection_name: str, request: Request):
    col = _col(request, collection_name)
    return {"collection": collection_name, "count": col.count()}
