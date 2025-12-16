from sqlmodel import SQLModel,Field,Column,JSON
from typing import Optional
from pgvector.sqlalchemy import Vector

class RAG_TABLE(SQLModel,table=True):
    id: int | None = Field(default=None,primary_key=True)
    embedding: list[float] = Field(sa_column=Column(Vector(768)))
    content: str

    
