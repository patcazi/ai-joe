from typing import List, Tuple, Optional
import os
import re
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

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


def _today_dt():
    tzname = os.getenv("APP_TZ", "America/Chicago")
    tz = ZoneInfo(tzname) if ZoneInfo else timezone.utc
    return datetime.now(tz)

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

def _normalize_dates_in_context(text: str) -> str:
    """
    Convert ambiguous phrases like 'this July', 'by August 1', 'end of June' into
    absolute & past-aware phrasing relative to today's date. We do NOT guess years
    unless obvious; we append '(past)' if the date is before today in the current year.
    """
    today = _today_dt().date()
    year = today.year

    def mark_past(dt):
        return " (past)" if dt < today else ""

    # 'this July' / 'in July' -> 'July <year>(past|)'
    def repl_month(m):
        mon = m.group(1).lower()
        m_num = _MONTHS.get(mon)
        if not m_num:
            return m.group(0)
        try:
            dt = datetime(year, m_num, 1).date()
            return f"{mon.capitalize()} {year}{mark_past(dt)}"
        except Exception:
            return m.group(0)

    # 'by August 1' / 'on August 1' -> 'August 1, <year>(past|)'
    def repl_month_day(m):
        mon = m.group(1).lower()
        day = int(m.group(2))
        m_num = _MONTHS.get(mon)
        if not m_num:
            return m.group(0)
        try:
            dt = datetime(year, m_num, day).date()
            return f"{mon.capitalize()} {day}, {year}{mark_past(dt)}"
        except Exception:
            return m.group(0)

    # 'by July' / 'before Sept' -> 'July <year>(past|)'
    def repl_by_month(m):
        mon = m.group(1).lower()
        m_num = _MONTHS.get(mon)
        if not m_num:
            return m.group(0)
        try:
            dt = datetime(year, m_num, 1).date()
            return f"{mon.capitalize()} {year}{mark_past(dt)}"
        except Exception:
            return m.group(0)

    # 'end of June' / 'mid July' -> 'late June <year>(past)' / 'mid July <year>(past)'
    def repl_qualifier(m):
        qual = m.group(1).lower()  # end|late|mid|early
        mon  = m.group(2).lower()
        m_num = _MONTHS.get(mon)
        if not m_num:
            return m.group(0)
        approx_day = {"early": 5, "mid": 15, "late": 25, "end": 28}.get(qual, 15)
        try:
            dt = datetime(year, m_num, approx_day).date()
            return f"{qual} {mon.capitalize()} {year}{mark_past(dt)}"
        except Exception:
            return m.group(0)

    # Patterns (support full names and common abbreviations; accept ordinals)
    text = re.sub(
        r"\b(?:this|in)\s+(january|february|march|april|may|june|july|august|september|october|"
        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        repl_month, text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:by|before|until)\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        repl_by_month, text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:by|on|before)\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?\b",
        repl_month_day, text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(early|mid|late|end)\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        repl_qualifier, text, flags=re.IGNORECASE,
    )

    # Optional: normalize "this month" / "next month"
    try:
        today = _today_dt().date()
        cur_abs = today.strftime("%B %Y")
        # this month -> Month YYYY (no past marker since it's current)
        text = re.sub(r"\bthis month\b", cur_abs, text, flags=re.IGNORECASE)
        # next month -> compute first day next month
        nm = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        nxt_abs = nm.strftime("%B %Y")
        text = re.sub(r"\bnext month\b", nxt_abs, text, flags=re.IGNORECASE)
    except Exception:
        pass

    return text


def _now_block() -> str:
    tz = os.getenv("APP_TZ", "America/Chicago")
    try:
        tzinfo = ZoneInfo(tz) if ZoneInfo else timezone.utc
    except Exception:
        tzinfo = timezone.utc
    now = datetime.now(tzinfo)
    iso = now.strftime("%Y-%m-%d (%A) %H:%M %Z")
    return (
        f"DATE CONTEXT: Today is {iso}. "
        "Treat dates in source documents as historical unless the CONTEXT explicitly states otherwise. "
        "Do not assume future dates; prefer relative phrasing (e.g., 'this week') or omit if unknown."
    )

# --- Post-processing helpers ---

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


def _fix_future_tense_with_past_dates(text: str) -> str:
    """
    If a sentence contains a '(past)' month marker but uses future/modal verbs like
    'will/ships/is going to', rephrase to past/plan wording to avoid temporal confusion.
    """
    lines = [l for l in text.splitlines()]
    out = []
    for line in lines:
        if "(past)" in line:
            line = re.sub(r"\bwill ship\b", "was planned to ship", line, flags=re.IGNORECASE)
            line = re.sub(r"\bwill\b", "was going to", line, flags=re.IGNORECASE)
            line = re.sub(r"\bships\b", "was planned to ship", line, flags=re.IGNORECASE)
            line = re.sub(r"\bis going to\b", "was going to", line, flags=re.IGNORECASE)
        out.append(line)
    return "\n".join(out)


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
    retrieval_context = _normalize_dates_in_context(retrieval_context)

    parts: List[str] = []

    # Current date/time awareness
    parts.append(_now_block())

    # Persona (system.md)
    if _BASE_SYSTEM:
        parts.append(_BASE_SYSTEM.strip())

    # Full BrainLift (always include for experiment) with date normalization
    if _BRAINLIFT:
        brainlift_injected = _normalize_dates_in_context(_BRAINLIFT)
        parts.append("---\nBRAINLIFT (full):\n" + brainlift_injected)

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
    text = _fix_future_tense_with_past_dates(text)
    return text


