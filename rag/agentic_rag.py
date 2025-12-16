from dotenv import load_dotenv
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from sqlmodel import Session, select
from sqlalchemy import text
import asyncio

from models import RAG_TABLE
from database import engine
from google.genai import Client

load_dotenv()

AGENT_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
base_agent = Agent(GroqModel("llama-3.3-70b-versatile", provider=GroqProvider(api_key=AGENT_API_KEY)))
embedding_client = Client(api_key=GEMINI_API_KEY)


def embed_query(query: str) -> list[float]:
    """Generate embedding for a query."""
    try:
        result = embedding_client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []


def retrieve_similar_chunks(query: str, top_k: int = 5) -> list[str]:
    """Retrieve top-k most similar chunks using pgvector similarity search."""
    
    # Generate query embedding
    query_embedding = embed_query(query)
    if not query_embedding:
        return []
    
    # Convert to pgvector format (string representation)
    embedding_str = str(query_embedding)
    
    with Session(engine) as session:
        # Use pgvector's <-> operator for L2 distance (lower = more similar)
        sql = text("""
            SELECT id, content, embedding <-> :query_vec AS distance
            FROM rag_table
            ORDER BY embedding <-> :query_vec
            LIMIT :limit
        """)
        
        result = session.execute(
            sql,
            {"query_vec": embedding_str, "limit": top_k}
        )
        
        chunks = []
        for row in result:
            chunks.append(row.content)
            print(f"  [ID {row.id}] Distance: {row.distance:.4f}")
        
        return chunks


async def rag_query(user_question: str):
    """Main RAG function: retrieve context and generate answer."""
    
    print("\n" + "="*60)
    print(f"Question: {user_question}")
    print("="*60)
    print("\nRetrieving relevant context...")
    
    # Retrieve similar chunks
    context_chunks = retrieve_similar_chunks(user_question, top_k=3)
    
    if not context_chunks:
        print("No relevant context found")
        return "I couldn't find relevant information to answer that question."
    
    # Combine chunks into context
    context = "\n\n".join(context_chunks)
    print(f"\nRetrieved {len(context_chunks)} chunks")
    
    # Build prompt with context
    prompt = f"""Based on the following context from SCP Foundation tales, answer the question.

Context:
{context}

Question: {user_question}

Answer:"""
    
    # Generate answer with agent
    print("\nGenerating answer...")
    result = await base_agent.run(prompt)
    
    print(f"\nAnswer:\n{result.output}\n")
    return result.output


async def chat_loop():
    """Interactive chat loop for asking multiple questions."""
    
    print("\n" + "="*60)
    print("SCP Foundation RAG Chatbot")
    print("="*60)
    print("Ask questions about SCP tales!")
    print("Type 'exit', 'quit', or 'q' to end the conversation.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', '']:
                print("\nGoodbye! 👋")
                break
            
            # Process the question
            await rag_query(user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


async def main():
    await chat_loop()


if __name__ == "__main__":
    asyncio.run(main())

