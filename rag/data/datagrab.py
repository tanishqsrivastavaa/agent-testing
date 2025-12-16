import requests
import json
import asyncio
from sqlmodel import Session,create_engine
from dotenv import load_dotenv
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import RAG_TABLE
from database import engine
from google.genai import Client
from bs4 import BeautifulSoup


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embedding_client = Client(api_key=GEMINI_API_KEY)

BASE_URL = "https://scp-data.tedivm.com/data/scp/tales"
INDEX_URL = f"{BASE_URL}/index.json"


def fetch_index(limit:int = 3) -> list[dict]:
    try:
        response = requests.get(INDEX_URL,timeout=10)
        response.raise_for_status()
        index_data = response.json()

        tales = list(index_data.values())[:limit]
        
        # Debug: print first tale structure
        if tales:
            print("Sample tale keys:", tales[0].keys())
            print("Sample tale:", json.dumps(tales[0], indent=2)[:500])
        
        print(f"Fetched {len(tales)} tales from index")
        return tales
    
    except Exception as e:
        print(f"Error fectching index: {e}")
        return []
    

def fetch_tale_content(tale: dict) -> str:
    try:
        # Get the content file (e.g., content_2022.json)
        content_file = tale.get("content_file")
        tale_link = tale.get("link")  # The unique tale identifier
        
        if not content_file or not tale_link:
            print(f"  Missing content_file or link")
            return ""
        
        print(f"  Looking for tale '{tale_link}' in {content_file}")
        
        # Fetch the content file
        content_url = f"{BASE_URL}/{content_file}"
        response = requests.get(content_url, timeout=10)
        response.raise_for_status()
        
        # Parse the content file - it's a dict with tale links as keys
        all_tales = response.json()
        
        # Debug: show what keys are available
        print(f"  Content file has {len(all_tales)} tales")
        if all_tales:
            sample_keys = list(all_tales.keys())[:3]
            print(f"  Sample keys: {sample_keys}")
        
        # Get this specific tale's content
        tale_data = all_tales.get(tale_link, {})
        
        if not tale_data:
            print(f"  Tale '{tale_link}' not found in content file")
            return ""
        
        html_content = tale_data.get('raw_content', '') or tale_data.get('raw_source', '')
        
        if not html_content:
            print(f"  No html/text in tale data. Keys: {tale_data.keys()}")
            return ""
        
        # Strip HTML tags using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n\n', strip=True)
        
        return text
    
    except Exception as e:
        print(f"  Error fetching content: {e}")
        return ""
    

def chunk_text(text: str,chunk_size: int = 500) -> list[str]:

    paragraphs = text.split("\n\n")
    chunks= []
    current_chunk= ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def embed_text(text: str) -> list[float]:
    try:
        result = embedding_client.models.embed_content(
            model="text-embedding-004",
            contents= text
        )
        embedding = result.embeddings[0].values
        print(f"Embedding dim: {len(embedding)}")
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
    

def ingest_tales():
    print("Starting tale ingestion...")

    tales = fetch_index(limit=3)
    if not tales:
        print("No tales fetched, exiting")
        return
    
    total_inserted = 0

    for tale in tales:
        title = tale.get("title", "Unknown")
        
        print(f"Processing: {title}")

        # Pass the entire tale dict, not just content_file
        content = fetch_tale_content(tale)
        if not content:
            print(f"  No content fetched, skipping")
            continue

        chunks = chunk_text(content)
        print(f"  Created {len(chunks)} chunks")

        with Session(engine) as session:
            for i, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                if not embedding:
                    print(f"  Chunk {i}: embedding failed")
                    continue

                try:
                    record = RAG_TABLE(
                        content=chunk,
                        embedding=embedding
                    )

                    session.add(record)
                    session.commit()
                    session.refresh(record)

                    print(f"  Chunk {i}: inserted with ID {record.id}")
                    total_inserted += 1
                except Exception as e:
                    print(f"  Chunk {i}: DB error: {e}")
                    session.rollback()
        
        print(f"  Tale complete: {len(chunks)} chunks inserted")
    
    print(f"\nIngestion complete! Inserted {total_inserted} chunks total")
    

if __name__ == "__main__":
    ingest_tales()