# backend/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from .db import get_session, engine
from .ingest import ingest_netcdf_file
import uvicorn

app = FastAPI(title="FloatChat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload-netcdf")
async def upload_netcdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, session: AsyncSession = Depends(get_session)):
    if not file.filename.endswith(".nc") and not file.filename.endswith(".nc4"):
        raise HTTPException(status_code=400, detail="Please upload a NetCDF (.nc/.nc4) file")
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # start ingestion in background (non-blocking)
    background_tasks.add_task(ingest_netcdf_file, save_path, session)
    return {"status": "accepted", "filename": file.filename}

@app.get("/api/profiles")
async def list_profiles(limit: int = 100, min_lat: float = None, max_lat: float = None,
                        min_lon: float = None, max_lon: float = None, session: AsyncSession = Depends(get_session)):
    q = "SELECT id, float_id, cycle_number, timestamp, lat, lon, variables_summary, raw_parquet_path FROM profiles"
    conditions = []
    params = {}
    if min_lat is not None and max_lat is not None:
        conditions.append("lat BETWEEN :min_lat AND :max_lat")
        params.update({"min_lat": min_lat, "max_lat": max_lat})
    if min_lon is not None and max_lon is not None:
        conditions.append("lon BETWEEN :min_lon AND :max_lon")
        params.update({"min_lon": min_lon, "max_lon": max_lon})
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    q += " ORDER BY timestamp DESC NULLS LAST LIMIT :limit"
    params["limit"] = limit
    result = await session.execute(q, params)
    rows = result.fetchall()
    # Convert to JSON response
    profiles = []
    for r in rows:
        profiles.append({
            "id": r[0],
            "float_id": r[1],
            "cycle_number": r[2],
            "timestamp": str(r[3]) if r[3] else None,
            "lat": r[4],
            "lon": r[5],
            "summary": r[6],
            "raw_parquet_path": r[7]
        })
    return {"profiles": profiles}

# Basic chat websocket (stateless wrapper over RAG)
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            # Expected message: {"type":"chat", "message":"..."}
            if data.get("type") == "chat":
                question = data.get("message", "")
                # Here call a RAG function to answer (synchronous)
                # For brevity, we return an echo + placeholder
                # Replace this with your rag_query(question) function
                answer = f"(PoC) Received: {question}"
                await ws.send_json({"type": "response", "answer": answer})
            else:
                await ws.send_json({"type": "error", "error": "invalid message type"})
    except Exception as e:
        await ws.close()

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
