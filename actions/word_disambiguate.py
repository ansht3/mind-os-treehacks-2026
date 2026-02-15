"""Use OpenAI to pick the best word from EMG class sequence candidates given page context."""

import asyncio

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL


async def suggest_noun_or_use_dict(
    class_sequence: list[str],
    page_context: str,
    previously_added_words: list[str],
    api_key: str | None = None,
) -> tuple[str | None, bool]:
    """Ask OpenAI: is this a proper noun? If YES, return (word, True) → Yes/No only.
    If NO, return (None, False) → fall back to dictionary + Yes/No/Retry.
    No validation. Trust OpenAI."""
    if not class_sequence:
        return None, False

    client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
    seq_str = " → ".join(class_sequence)
    prev = " ".join(previously_added_words) if previously_added_words else "(none)"

    prompt = f"""NEW WORD — user just pressed "end word". This sequence is for the word they are typing NOW.

CURRENT sequence (USE THIS, ignore any prior words): {seq_str}
Letter map: OPEN=A,O,U,H | CLOSE=E,I,Y | TIP=T,D,N,L,S,Z | BACK=K,G,J,C,R | LIPS=B,M,F,P,W

Page: {page_context[:1000]}
(Previous words in their prompt, for context only: {prev})

Derive the word ONLY from the CURRENT sequence above. Do NOT suggest a word from previous context.
Is this sequence a PROPER NOUN (company, brand, product, person)?
If YES → reply: NOUN:word  (the word that spells from this exact sequence)
If NO → reply: DICT

Reply exactly NOUN:word or DICT. Nothing else."""

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip().upper()
        if raw.startswith("NOUN:"):
            word = raw[5:].strip().lower()
            return word, True
        return None, False
    except Exception:
        return None, False




async def pick_best_word(
    candidates: list[str],
    page_context: str,
    previously_added_words: list[str],
    api_key: str | None = None,
) -> str | None:
    """Pick the best word from candidates given webpage context and previously added words.

    Uses a low-cost model (gpt-4o-mini) for cheap inference.
    Returns the best match, or None if no candidates.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
    prev = " ".join(previously_added_words) if previously_added_words else "(none yet)"
    prompt = f"""You are helping a user who is typing via thought-controlled EMG input. They selected a sequence of mouth-shape classes (OPEN, CLOSE, TIP, BACK, LIPS) which maps to multiple possible English words.

Page context (what's visible on the webpage):
{page_context[:1500]}

Previously added words in the prompt so far: {prev}

Possible words from the EMG sequence: {', '.join(candidates)}

Pick the single most likely word the user intended. Consider:
- The webpage context (search queries, forms, links, etc.)
- The previously added words (the full prompt they're building)
- Common English usage

Reply with ONLY the word, nothing else. Lowercase."""

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Validate it's one of our candidates
        for c in candidates:
            if c.lower() == raw:
                return c
        return candidates[0]
    except Exception:
        return candidates[0]
