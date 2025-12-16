# Agentic RAG: SCP Foundation Knowledge Base

An intelligent Retrieval-Augmented Generation (RAG) system that answers questions about SCP Foundation tales using semantic search and AI-powered reasoning.

## Overview

This project combines:
- **Semantic Search**: pgvector-powered similarity matching to find relevant tale chunks
- **LLM Reasoning**: Groq's Llama model for intelligent question answering
- **Embeddings**: Google Gemini's text-embedding-004 for high-quality vector representations
- **Vector Database**: NeonDB with pgvector extension for efficient similarity search

### Dataset
The system ingests tales from the [SCP Foundation data repository](https://github.com/scp-data/scp-api), a creative writing wiki featuring fictional anomalies and containment procedures.

## Architecture

```
┌─────────────────────┐
│   User Question     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Gemini Text Embedding API          │ ◄─── Query Embedding (768-dim)
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  NeonDB pgvector Similarity Search   │ ◄─── Retrieve top-k chunks
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Build Context + Prompt             │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Groq Llama Model                   │ ◄─── Generate Answer
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Answer to User    │
└─────────────────────┘
```

## Why Gemini Over Groq for Embeddings?

We use **Google Gemini's `text-embedding-004`** for embeddings and **Groq's Llama** for generation for these reasons:

| Aspect | Gemini Embeddings | Groq for Generation |
|--------|------------------|-------------------|
| **Quality** | Superior semantic understanding (768 dims) | Fast, high-quality reasoning |
| **Specialized** | Specifically optimized for text embeddings | Optimized for generation/chat |
| **Consistency** | Consistent 768-dim vectors for pgvector | Multi-turn conversation capability |
| **Latency** | Acceptable for indexing phase | <100ms for real-time responses |
| **Cost** | Efficient per-token pricing | Cost-effective for inference |

**Best Practice**: Use specialized models for each task rather than one general model. Gemini's embeddings provide superior semantic understanding with consistent dimensionality (768), while Groq excels at fast, cost-effective text generation with excellent reasoning capabilities.

## Setup

### Prerequisites
- Python 3.12+
- NeonDB account (PostgreSQL with pgvector)
- API keys:
  - `GROQ_API_KEY` from [Groq Console](https://console.groq.com)
  - `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com)
  - `NEON_DB_URL` from [NeonDB Dashboard](https://console.neon.tech)

### Installation

#### 1. Clone and navigate to project:
```bash
cd agent-testing/rag
```

#### 2. Install `uv` (Python package manager)

**What is `uv`?**
`uv` is a fast, Rust-based Python package manager that replaces `pip` and `poetry`. It's significantly faster and more reliable.

**Installation:**

On macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

**Using `uv` Commands:**
```bash
# Sync all dependencies from pyproject.toml
uv sync

# Install specific package
uv pip install package_name

# Run Python script (auto-activates virtual environment)
uv run script.py

# Run with specific Python version
uv run --python 3.12 script.py
```

#### 3. Create `.env` file with your credentials:
```bash
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
NEON_DB_URL=postgresql://user:password@host:5432/dbname?sslmode=require
EOF
```

### PostgreSQL Database Setup (NeonDB)

#### Step 1: Create NeonDB Project
1. Go to [NeonDB Console](https://console.neon.tech)
2. Click "Create New Project"
3. Enter project name and select region
4. Create a new database
5. Copy the connection string

#### Step 2: Format Connection String
Your NeonDB URL should look like:
```
postgresql://username:password@ep-xxxx-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
```

Add this to your `.env`:
```
NEON_DB_URL=postgresql://username:password@ep-xxxx-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
```

#### Step 3: Enable pgvector Extension
Connect to your NeonDB and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

You can do this via:
- NeonDB Web Console SQL Editor
- `psql` command line:
  ```bash
  psql "postgresql://username:password@ep-xxxx-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require" -c "CREATE EXTENSION IF NOT EXISTS vector;"
  ```

#### Step 4: Create RAG Table
```sql
CREATE TABLE rag_table (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR(768),
    content TEXT
);
```

#### Step 5: Create Index for Faster Search (optional, recommended for >1000 rows)
```sql
CREATE INDEX ON rag_table USING ivfflat (embedding vector_cosine_ops);
```

#### Step 6: Install Python Dependencies
```bash
uv sync
```

Or manually:
```bash
uv pip install google-genai pydantic-ai sqlmodel pgvector psycopg[binary] beautifulsoup4 requests
```

## Usage

### Step 1: Ingest SCP Tales into Database

Fetch SCP Foundation tales and store them with embeddings:

```bash
uv run data/datagrab.py
```

**Expected Output:**
```
Starting tale ingestion...
Fetched 3 tales from index
Processing: Underture - SCP Foundation
  Looking for tale 'underture' in content_2022.json
  Content file has 47 tales
  Created 5 chunks
  Embedding dim: 768
  Chunk 0: inserted with ID 1
  Chunk 1: inserted with ID 2
  Chunk 2: inserted with ID 3
  Chunk 3: inserted with ID 4
  Chunk 4: inserted with ID 5
  Tale complete: 5 chunks inserted
Processing: 01092018 - SCP Foundation
  Created 4 chunks
  Chunk 0: inserted with ID 6
  Chunk 1: inserted with ID 7
  Chunk 2: inserted with ID 8
  Chunk 3: inserted with ID 9
  Tale complete: 4 chunks inserted
Ingestion complete! Inserted 9 chunks total
```

**What happens:**
1. Fetches tale index from SCP API
2. Downloads each tale's HTML content
3. Chunks text into semantic paragraphs
4. Generates 768-dimensional embeddings via Gemini
5. Stores chunks + embeddings in NeonDB

### Step 2: Start Interactive RAG Chatbot

Launch the conversational interface:

```bash
uv run agentic_rag.py
```

**Example Chat Session:**
```
============================================================
SCP Foundation RAG Chatbot
============================================================
Ask questions about SCP tales!
Type 'exit', 'quit', or 'q' to end the conversation.

You: What is the Underture about?
============================================================
Question: What is the Underture about?
============================================================

Retrieving relevant context...
  [ID 1] Distance: 0.3452
  [ID 2] Distance: 0.4128
  [ID 3] Distance: 0.4521

Retrieved 3 chunks

Generating answer...

Answer:
The Underture is a creative tale that explores the intersection of music,
reality, and consciousness. It tells the story of how a musical composition
can reshape the fabric of existence itself...

You: Tell me about anomalies mentioned
============================================================
Question: Tell me about anomalies mentioned
============================================================

Retrieving relevant context...
  [ID 4] Distance: 0.3891
  [ID 5] Distance: 0.4234
  [ID 6] Distance: 0.4556

Retrieved 3 chunks

Generating answer...

Answer:
The tales mention several anomalies including dimensional rifts,
consciousness manipulation, and temporal distortions...

You: exit
Goodbye! 👋
```

### Step 3: Customize Data Ingestion

#### Change Data Source
Edit `data/datagrab.py`, line with `BASE_URL`:
```python
# For SCP objects instead of tales:
BASE_URL = "https://scp-data.tedivm.com/data/scp/items"

# For SCP hubs:
BASE_URL = "https://scp-data.tedivm.com/data/scp/hubs"
```

#### Adjust Chunk Size
In `data/datagrab.py`, modify `ingest_tales()`:
```python
# Smaller chunks = more specific but potentially fragmented
chunks = chunk_text(content, chunk_size=300)

# Larger chunks = more context but less specific
chunks = chunk_text(content, chunk_size=1000)
```

#### Ingest More Tales
In `data/datagrab.py`, modify `ingest_tales()`:
```python
# Default ingests 3 tales
tales = fetch_index(limit=10)  # Ingest 10 tales
```

#### Adjust Retrieval Parameters
In `agentic_rag.py`, modify `chat_loop()`:
```python
# Retrieve more context chunks
context_chunks = retrieve_similar_chunks(user_question, top_k=5)

# Use different distance metric
# Change <-> to <=> for cosine distance in SQL
```

## Project Structure

```
rag/
├── agentic_rag.py              # Main RAG chatbot
├── data/
│   └── datagrab.py             # Data ingestion pipeline
├── models.py                   # SQLModel (RAG_TABLE definition)
├── database.py                 # NeonDB connection setup
├── pyproject.toml              # Project dependencies
├── .env                        # API keys (git-ignored)
└── README.md                   # This file
```

## Key Components

### `agentic_rag.py` - RAG Chatbot
```python
# Main functions:
- embed_query(query: str) -> list[float]
  Generates embeddings for user questions

- retrieve_similar_chunks(query: str, top_k: int) -> list[str]
  Semantic search using pgvector similarity

- rag_query(user_question: str)
  Complete RAG pipeline: retrieve + generate answer

- chat_loop()
  Interactive conversation interface
```

### `data/datagrab.py` - Data Pipeline
```python
# Main workflow:
- fetch_index(limit: int) -> list[dict]
  Fetches tale metadata from SCP API

- fetch_tale_content(tale: dict) -> str
  Retrieves and cleans tale HTML content

- chunk_text(text: str, chunk_size: int) -> list[str]
  Splits text into semantic paragraphs

- embed_text(text: str) -> list[float]
  Generates 768-dim embeddings via Gemini

- ingest_tales()
  Orchestrates: fetch → chunk → embed → store
```

### `models.py` - Database Schema
```python
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

class RAG_TABLE(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    embeddings: list[float] = Field(sa_column=Column("embedding", Vector(768)))
    content: str
```

## Troubleshooting

### Error: "psycopg2.OperationalError: could not translate host name"
**Cause**: Invalid database URL format or network connectivity issue

**Fix**:
```bash
# Verify URL format
echo $NEON_DB_URL
# Should be: postgresql://user:pass@host:5432/db?sslmode=require

# Test connection
psql "$NEON_DB_URL" -c "SELECT 1;"
```

### Error: "column 'embeddings' of relation 'rag_table' does not exist"
**Cause**: Table was created with different column names

**Fix**:
```sql
-- Drop and recreate table
DROP TABLE rag_table;

CREATE TABLE rag_table (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR(768),
    content TEXT
);
```

### Error: "pgvector extension not installed"
**Cause**: Extension not enabled in NeonDB

**Fix**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

### No Results from Semantic Search
**Cause**: Either no data in table or similarity threshold too strict

**Fix**:
```bash
# Check if data was ingested
psql "$NEON_DB_URL" -c "SELECT COUNT(*) FROM rag_table;"

# Verify embeddings are stored
psql "$NEON_DB_URL" -c "SELECT id, content FROM rag_table LIMIT 1;"

# Try with higher top_k to see if results exist
# In agentic_rag.py, increase top_k from 3 to 10
```

### `uv` Command Not Found
**Cause**: `uv` not properly installed or not in PATH

**Fix**:
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH manually if needed
export PATH="$HOME/.cargo/bin:$PATH"

# Verify
uv --version
```

## Performance Optimization

### 1. Batch Embedding for Faster Ingestion
Modify `data/datagrab.py` to embed multiple chunks in parallel (advanced).

### 2. Vector Index for Large Datasets
After ingesting >1000 rows:
```sql
CREATE INDEX rag_table_embedding_idx 
ON rag_table USING ivfflat (embedding vector_cosine_ops);
```

### 3. Tune Chunk Size
- **Small (300 chars)**: More chunks, specific retrieval
- **Large (1000 chars)**: Fewer chunks, more context
- **Optimal**: 500-800 chars for most use cases

### 4. Adjust Retrieval K
- `top_k=3`: Fast, focused answers
- `top_k=5`: Balanced context
- `top_k=10`: Comprehensive but slower

## Future Enhancements

- [ ] Conversation memory and multi-turn context
- [ ] Streaming responses from Groq API
- [ ] Citation tracking (show which chunks generated each answer)
- [ ] Support for SCP items and hubs
- [ ] User feedback loop for relevance scoring
- [ ] Fine-tuning chunk boundaries
- [ ] Rate limiting and request queuing

## License

This project uses data from the [SCP Foundation](https://scp-wiki.wikidot.com/) licensed under CC-BY-SA 3.0.

## Resources

- [SCP Foundation](https://scp-wiki.wikidot.com/) - Original creative writing community
- [SCP Data Repository](https://github.com/scp-data/scp-api) - Structured data API
- [pgvector Documentation](https://github.com/pgvector/pgvector) - Vector search
- [NeonDB Docs](https://neon.tech/docs) - PostgreSQL hosting
- [Pydantic AI](https://ai.pydantic.dev/) - LLM framework
- [Groq API](https://groq.com/api) - Fast inference
- [Google Gemini API](https://ai.google.dev/) - Embeddings
- [uv Package Manager](https://astral.sh/uv/) - Python packaging
