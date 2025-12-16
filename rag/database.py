from sqlmodel import SQLModel, Session,create_engine
from dotenv import load_dotenv
import os
DB_URL = os.getenv("NEON_DB_URL")

if not DB_URL:
    print(f"{DB_URL} NOT WORKING")

engine = create_engine(DB_URL)
