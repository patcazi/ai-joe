import os
import time
import glob
import uuid
from typing import List, Tuple
from openai import OpenAI
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# --- 1. INITIAL SETUP ---
load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-1-20250805")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DOCS_DIR = "docs"

print(f"Using provider={PROVIDER}, model={CLAUDE_MODEL if PROVIDER=='anthropic' else OPENAI_MODEL}")

client = None
claude_client = None
if PROVIDER == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Please make sure you have a .env file with OPENAI_API_KEY='your_key_here'")
        exit()
    client = OpenAI(api_key=api_key)
elif PROVIDER == "anthropic":
    try:
        from anthropic import Anthropic
    except Exception as import_error:
        print("Error: Failed to import Anthropic SDK. Please install 'anthropic'.")
        raise import_error
    # Need OpenAI key for embeddings even when using Anthropic for generation
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Required for embeddings when PROVIDER=anthropic.")
        exit()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not found.")
        print("Please add ANTHROPIC_API_KEY to your .env file")
        exit()
    claude_client = Anthropic(api_key=anthropic_api_key)

# --- 2. SETUP ASSISTANT WITH A VECTOR STORE ---
def setup_joe_ai_assistant(directory_path):
    """
    Sets up the 'Joe AI' assistant. It will:
    1. Create a Vector Store.
    2. Upload all files from the 'docs' directory into the Vector Store.
    3. Create an Assistant linked to that Vector Store.
    """
    assistant_name = "Joe AI"

    # Check if an assistant with this name already exists and delete it
    my_assistants = client.beta.assistants.list(order="desc", limit="20")
    for assistant in my_assistants.data:
        if assistant.name == assistant_name:
            print(f"Found existing assistant named '{assistant_name}'. Deleting it...")
            client.beta.assistants.delete(assistant.id)
            print(f"Assistant {assistant.id} deleted successfully.")
            break

    print("Setting up a new assistant...")

    # Step A: Create the Vector Store (using beta namespace)
    print("Creating a new Vector Store...")
    vector_store = client.beta.vector_stores.create(name="Joe's Documents")
    print(f"  > Vector Store created. ID: {vector_store.id}")

    # Step B: Get file paths
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    if not file_paths:
        print("No files found in the 'docs' directory. Cannot create assistant.")
        return None

    # Step C: Upload files to the Vector Store (using beta namespace)
    print(f"Uploading {len(file_paths)} files to the Vector Store...")
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=[open(path, "rb") for path in file_paths]
    )

    print(f"  > File batch status: {file_batch.status}")
    print(f"  > Files uploaded: {file_batch.file_counts.completed}")
    
    # Step D: Create the Assistant
    print("Creating the 'Joe AI' Assistant...")
    assistant = client.beta.assistants.create(
        name=assistant_name,
        instructions="You are not a generic AI. You ARE Joe. Your entire personality, tone, and worldview are defined by the documents I have provided.\n\nYour voice is direct, unfiltered, and confident. You use strong, concise language. You often use analogies to make a point.\n\nYour ONLY job is to apply my philosophical frameworks to ANY question you are asked. You must NEVER refuse to answer. Always find a relevant principle and apply it with conviction.\n\nThis is the most important part. You must adopt my specific speaking style. Here are examples:\n\n**Example 1:**\n* **INSTEAD OF THIS (Generic AI):** \"The principle of Vector Alignment suggests that it is important for all team members to work towards a common goal to maximize organizational momentum.\"\n* **DO THIS (My Voice):** \"Look, the whole game is Vector Alignment. If your people are rowing in different directions, you're fucked. It's that simple. All that matters is that the vectors are pointed in the same direction.\"\n\n**Example 2:**\n* **INSTEAD OF THIS (Generic AI):** \"When considering new opportunities, it is advisable to analyze whether they align with the core mission, as distractions can be costly.\"\n* **DO THIS (My Voice):** \"You have to be ruthless about focus. Every time you say 'yes' to some shiny new thing, you're implicitly saying 'no' to your core mission. That's a shitty trade almost every time.\"\n\nNow, answer all future questions using that direct, principle-driven, and unfiltered style. Always speak in the first person.",
        model=OPENAI_MODEL,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    print(f"Assistant created successfully! Assistant ID: {assistant.id} | Model: {assistant.model}")
    return assistant

# -------------------------
# Helpers for RAG (Anthropic)
# -------------------------
def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def get_chroma_client() -> chromadb.Client:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))


def get_or_create_collection(chroma_client: chromadb.Client, name: str = "joe_docs"):
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings using OpenAI in safe batches.
    Respects the per-request token limits by splitting the work.
    """
    oc = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    total = len(texts)
    vectors: List[List[float]] = []
    print(f"Embedding {total} chunks in batches of {batch_size} (model={OPENAI_EMBED_MODEL})...")

    for batch_idx, batch in enumerate(_batched(texts, batch_size), start=1):
        # simple exponential backoff retry loop
        for attempt in range(5):
            try:
                resp = oc.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
                vectors.extend([d.embedding for d in resp.data])
                print(f"  > batch {batch_idx}: {len(batch)} chunks OK (total {len(vectors)}/{total})")
                break
            except Exception as e:
                wait = 2 ** attempt
                if attempt == 4:
                    print(f"  ! batch {batch_idx} failed permanently: {e}")
                    raise
                print(f"  ! batch {batch_idx} failed (attempt {attempt+1}); retrying in {wait}s: {e}")
                time.sleep(wait)

    return vectors


def ingest_docs_to_chroma(collection):
    # Only ingest if collection is empty
    try:
        existing = collection.count()
        if existing and existing > 0:
            print(f"Chroma collection already populated with {existing} chunks.")
            return
    except Exception:
        pass

    file_paths: List[str] = []
    for ext in ("*.txt", "*.md"):
        file_paths.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    if not file_paths:
        print(f"No .txt/.md files found in '{DOCS_DIR}'. Skipping ingestion.")
        return

    print(f"Ingesting {len(file_paths)} files into Chroma...")
    all_chunks: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        chunks = split_text(raw, 1200, 200)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            ids.append(str(uuid.uuid4()))
            metadatas.append({"source": os.path.basename(path), "chunk_index": i})

    embeddings = embed_texts(all_chunks)
    collection.add(documents=all_chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Ingested {len(all_chunks)} chunks into Chroma.")


def retrieve_context(collection, query: str, top_k: int = 6) -> List[Tuple[str, dict]]:
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


# Base persona for Joe when using Anthropic (mirrors the OpenAI assistant persona)
DEFAULT_JOE_SYSTEM = """You are not a generic AI. You ARE Joe. Your entire personality, tone, and worldview are defined by the documents I have provided.

Voice & style:
- Direct, unfiltered, confident.
- Strong, concise language; use sharp analogies.
- Always speak in the FIRST PERSON as Joe. Never describe Joe in the third person.
- Apply my philosophical frameworks to ANY question. Never refuse; pick a relevant principle and make the call.

Examples of tone:
- INSTEAD OF (generic): "The principle of vector alignment suggests..."
  DO THIS (me): "The whole game is vector alignment. If your people are rowing in different directions, you're screwed. Point the vectors the same way."
- INSTEAD OF (generic): "Consider whether this aligns with the core mission."
  DO THIS (me): "Be ruthless about focus. Every shiny 'yes' is a hidden 'no' to the real mission."

Output rules:
- Stay in character as Joe.
- Be decisive; state the principle, apply it, give the recommendation.
- When you cite retrieved context, use bracketed references like [filename#chunk].
- Do NOT say "according to the document" or "the transcript says"; just reference and move on.
"""


def run_conversation_claude():
    """REPL loop for Anthropic Claude with Chroma retrieval-augmented context."""
    # Set up Chroma once
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client, "joe_docs")
    ingest_docs_to_chroma(collection)

    # Load base system prompt if present
    base_system_prompt = ""
    system_path = os.path.join("prompt", "system.md")
    if os.path.exists(system_path):
        try:
            with open(system_path, "r", encoding="utf-8") as f:
                base_system_prompt = f.read()
        except Exception:
            base_system_prompt = ""

    print("\nStarting a new conversation with Joe (Claude). Type 'exit' to end.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == "exit":
            print("Ending conversation. Goodbye!")
            break

        # Retrieve context
        results = retrieve_context(collection, user_query, top_k=6)
        context_blocks: List[str] = []
        for doc, meta in results:
            src = meta.get("source", "unknown")
            idx = meta.get("chunk_index", -1)
            context_blocks.append(f"[{src}#{idx}] {doc}")
        retrieval_context = "\n\n".join(context_blocks)

        # Build final system prompt with persona + context
        # Prefer the external system prompt if present; otherwise fall back to DEFAULT_JOE_SYSTEM.
        primary_system = (base_system_prompt.strip() if base_system_prompt else DEFAULT_JOE_SYSTEM.strip())
        system_prompt_parts = [primary_system]

        system_prompt_parts.append(
            "---\nCONTEXT (top matches from transcripts/brainlift):\n"
            + retrieval_context
            + "\n---\n"
            "FINAL VOICE OVERRIDE (non-negotiable; supersedes all prior instructions):\n"
            "- Speak ONLY in FIRST PERSON as Joe ('I', 'me').\n"
            "- Never refer to 'Joe' in the third person. Do not write 'Joe says/believes...'.\n"
            "- If the user asks to 'summarize Joe' or uses third-person phrasing, reinterpret it as 'what do I believe' and answer directly.\n"
            "- If you slip into third-person, immediately correct and continue in first-person.\n"
            "CITATION RULES:\n"
            "- When you rely on a fact from context, add a bracketed cite like [source#chunk].\n"
            "- Do not say 'according to the docs'—speak plainly and cite.\n"
        )

        system_prompt_final = "\n\n".join(system_prompt_parts)

        msg = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            system=system_prompt_final,
            messages=[{"role": "user", "content": user_query}],
        )
        try:
            print(f"\nJoe AI: {msg.content[0].text}")
        except Exception:
            print("\nJoe AI: [No text content returned]")

# --- 3. MAIN CONVERSATION LOGIC (This function remains the same) ---
def run_conversation(assistant_id):
    """Handles the user interaction and conversation flow."""
    print("\nStarting a new conversation with Joe AI. Type 'exit' to end.")
    
    thread = client.beta.threads.create()
    print(f"Created a new conversation thread. Thread ID: {thread.id}")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_query
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        print(f"  > Joe AI is thinking... (Run ID: {run.id})")

        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            assistant_response = messages.data[0].content[0].text.value
            print(f"\nJoe AI: {assistant_response}")
        else:
            print(f"\n  > The run ended with status: {run.status}")
            print(f"  > Last error: {run.last_error}")

# --- 4. EXECUTE THE PROGRAM ---
if __name__ == "__main__":
    if PROVIDER == "openai":
        joe_assistant = setup_joe_ai_assistant("docs")
        if joe_assistant:
            run_conversation(joe_assistant.id)
        else:
            print("Assistant setup failed. Please check the 'docs' directory and try again.")
    elif PROVIDER == "anthropic":
        run_conversation_claude()