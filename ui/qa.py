from typing import List, Tuple, Optional
import os

# Reuse existing backend (RAG + Anthropic) from main.py
from main import (
    get_chroma_client,
    get_or_create_collection,
    ingest_docs_to_chroma,
    retrieve_context,
    CLAUDE_MODEL,
    claude_client,
)


# Set up Chroma once on import
_chroma_client = get_chroma_client()
_collection = get_or_create_collection(_chroma_client, "joe_docs")
ingest_docs_to_chroma(_collection)


def _load_base_system_prompt() -> str:
    system_path = os.path.join("prompt", "system.md")
    if os.path.exists(system_path):
        try:
            with open(system_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""


_BASE_SYSTEM = _load_base_system_prompt()


# Load the full BrainLift once (experimental: always inject into system)
def _load_brainlift() -> str:
    path = os.getenv("BRAINLIFT_PATH", "docs/brainliftV0.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


_BRAINLIFT = _load_brainlift()

# --- Post-processing helpers ---
import re

_META_PATTERNS = [
    r"^\s*I['’]m speaking in my own voice\.?\s*$",
    r"^\s*I am speaking in my own voice\.?\s*$",
    r"^\s*\*\*?SPOV anchor\*\*?:?.*$",
    r"^\s*\*\*?Application\*\*?:?.*$",
    r"^\s*\*\*?Action\*\*?:?.*$",
    r"^\s*\*\*?Risks/?Countermoves\*\*?:?.*$",
    r"^\s*Output Shape:?.*$",
    r"^\s*##+\s.*$",
]

def _strip_meta_labels(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(pat, line.strip(), flags=re.IGNORECASE) for pat in _META_PATTERNS):
            continue
        cleaned.append(line)
    out = "\n".join(cleaned).strip()
    out = re.sub(r"^(Here[’']?s why:)\s*", "", out, flags=re.IGNORECASE)
    return out


def _enforce_first_person(text: str) -> str:
    # Soft nudge: remove obvious third-person preambles
    return re.sub(r"\b(J|j)oe\s+(would|does|thinks|believes)\b.*", "", text)


def answer(message: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
    if claude_client is None:
        return "Backend not configured for Anthropic. Set PROVIDER=anthropic in .env."

    # Retrieve context for the current question
    results = retrieve_context(_collection, message, top_k=6)
    context_blocks: List[str] = []
    for doc, meta in results:
        src = meta.get("source", "unknown")
        idx = meta.get("chunk_index", -1)
        context_blocks.append(f"[{src}#{idx}] {doc}")
    retrieval_context = "\n\n".join(context_blocks)

    parts: List[str] = []

    # Persona (system.md)
    if _BASE_SYSTEM:
        parts.append(_BASE_SYSTEM.strip())

    # Full BrainLift (always include for experiment)
    if _BRAINLIFT:
        parts.append("---\nBRAINLIFT (full):\n" + _BRAINLIFT)

    # Retrieved context (keep citations)
    parts.append(
        "---\nCONTEXT (top matches from transcripts/brainlift):\n"
        + retrieval_context
        + "\n---\nUse only what is relevant. Cite [source#chunk] when you pull facts."
    )

    system_prompt_final = "\n\n".join(parts)

    # Call Anthropic
    msg = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        system=system_prompt_final,
        messages=[{"role": "user", "content": message}],
    )
    raw_text = msg.content[0].text if getattr(msg, "content", None) else ""

    # Post-process to remove meta labels and enforce first person
    text = _strip_meta_labels(raw_text)
    text = _enforce_first_person(text)
    return text


