# backend/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from geoalchemy2 import Geometry
from .db import Base

class FloatModel(Base):
    __tablename__ = "floats"
    id = Column(Integer, primary_key=True)
    wmo_id = Column(String, unique=True, index=True)  # or author ID
    manufacturer = Column(String, nullable=True)
    platform_type = Column(String, nullable=True)
    # other metadata as needed

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    float_id = Column(Integer, ForeignKey("floats.id"), index=True)
    cycle_number = Column(Integer, nullable=True)
    timestamp = Column(DateTime, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    geom = Column(Geometry("POINT", srid=4326))
    variables_summary = Column(Text, nullable=True)
    raw_parquet_path = Column(String, nullable=True)
    metadata = Column(JSONB, nullable=True)

class EmbeddingDoc(Base):
    __tablename__ = "embedding_docs"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), index=True)
    text = Column(Text)
    meta = Column(JSONB)
    # embeddings are stored in Chroma/FAISS instead of DB for efficiency
