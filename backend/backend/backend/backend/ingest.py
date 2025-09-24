# backend/ingest.py
import os
import uuid
import json
import tempfile
import pandas as pd
import xarray as xr
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import FloatModel, Profile, EmbeddingDoc
from .embeddings import embed_text
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "/data/chroma")
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
collection = chroma_client.get_collection("profiles") if "profiles" in [c.name for c in chroma_client.list_collections()] else chroma_client.create_collection("profiles")

RAW_DIR = os.getenv("RAW_PROFILE_DIR", "/data/raw_profiles")
os.makedirs(RAW_DIR, exist_ok=True)

async def ingest_netcdf_file(path: str, session: AsyncSession):
    """
    Parse an Argo NetCDF profile file, store metadata in Postgres, raw data as parquet,
    and add a text summary + embedding to Chroma.
    """
    ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True, decode_coords=True)
    # Try to find profiling dims and variables - handle common Argo shapes.
    # Many profile files use dims N_PROF and N_LEVELS with TEMP(N_PROF,N_LEVELS), etc.
    # We'll attempt a resilient extraction: for each profile index, extract arrays.
    # Find dims:
    dims = ds.dims
    # heuristics:
    if "N_PROF" in dims and "N_LEVELS" in dims:
        nprof = int(ds.dims["N_PROF"])
        nlevels = int(ds.dims["N_LEVELS"])
        for ip in range(nprof):
            try:
                # pick adjusted if present
                temp_var = "TEMP_ADJUSTED" if "TEMP_ADJUSTED" in ds.variables else ("TEMP" if "TEMP" in ds.variables else None)
                pres_var = "PRES_ADJUSTED" if "PRES_ADJUSTED" in ds.variables else ("PRES" if "PRES" in ds.variables else None)
                psal_var = "PSAL_ADJUSTED" if "PSAL_ADJUSTED" in ds.variables else ("PSAL" if "PSAL" in ds.variables else None)

                temps = ds[temp_var].isel(N_PROF=ip).values if temp_var else None
                press = ds[pres_var].isel(N_PROF=ip).values if pres_var else None
                psal = ds[psal_var].isel(N_PROF=ip).values if psal_var else None

                # coordinates / metadata
                lat = float(ds["LATITUDE"].isel(N_PROF=ip).values) if "LATITUDE" in ds.variables else None
                lon = float(ds["LONGITUDE"].isel(N_PROF=ip).values) if "LONGITUDE" in ds.variables else None
                # juld: time variable could be JULD or similar
                timestamp = None
                if "JULD" in ds.variables:
                    jv = ds["JULD"].isel(N_PROF=ip)
                    try:
                        timestamp = pd.to_datetime(float(jv.values), unit="D", origin=pd.Timestamp("1950-01-01"))
                    except Exception:
                        timestamp = None

                # summary
                s_mean = None
                if temps is not None:
                    finite = pd.Series(temps).dropna()
                    if not finite.empty:
                        s_mean = float(finite.mean())

                summary = f"Profile {ip}: lat={lat}, lon={lon}, time={timestamp}, mean_temp={s_mean}"
                # Save raw arrays as parquet
                df = pd.DataFrame({
                    "pres": press.tolist() if press is not None else [],
                    "temp": temps.tolist() if temps is not None else [],
                    "psal": psal.tolist() if psal is not None else []
                })
                profile_uuid = str(uuid.uuid4())
                raw_path = os.path.join(RAW_DIR, f"profile_{profile_uuid}.parquet")
                df.to_parquet(raw_path, index=False)

                # Insert metadata into Postgres
                new_profile = Profile(
                    float_id=None,
                    cycle_number=int(ip),
                    timestamp=timestamp,
                    lat=lat,
                    lon=lon,
                    geom=f"SRID=4326;POINT({lon} {lat})" if (lat is not None and lon is not None) else None,
                    variables_summary=summary,
                    raw_parquet_path=raw_path,
                    metadata={}
                )
                session.add(new_profile)
                await session.flush()  # get new_profile.id

                # create embedding & add to chroma
                embed_texts = [summary]
                embeddings = embed_text(embed_texts)
                collection.add(
                    documents=embed_texts,
                    metadatas=[{"profile_id": new_profile.id, "lat": lat, "lon": lon}],
                    ids=[str(new_profile.id)],
                    embeddings=embeddings
                )
                # commit after each profile (optionally batch)
                await session.commit()
            except Exception as e:
                await session.rollback()
                print("Error ingesting profile:", e)
    else:
        # fallback: try to convert dataset into profiles by scanning variables with a profile dim
        print("Dataset structure not recognised (no N_PROF/N_LEVELS). Please expand parser for other file types.")
    ds.close()
