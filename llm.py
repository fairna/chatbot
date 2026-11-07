"""LLM access via an OpenAI-compatible endpoint. Default: Google Gemini.
Env: GEMINI_API_KEY (or LLM_API_KEY), LLM_MODEL, LLM_FAST_MODEL, LLM_BASE_URL,
LLM_MIN_INTERVAL (seconds between calls, for free-tier rate limits),
LLM_REASONING_EFFORT ("none" disables Gemini 2.5 thinking; set to "" to omit),
LLM_TIMEOUT (seconds per request, default 45), LKY_DEBUG=0 to silence timing logs."""

import os
import streamlit as st
import time
from openai import (
    OpenAI,
    RateLimitError,
    BadRequestError,
    APIConnectionError,
    InternalServerError,
)

BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    or st.secrets.get("LLM_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("LLM_API_KEY")
)

MAIN_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gemini-2.5-flash-lite")
REASONING = os.getenv("LLM_REASONING_EFFORT", "none")
MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45"))
DEBUG = os.getenv("LKY_DEBUG", "1") == "1"
MAX_ATTEMPTS = 3

# max_retries=0: the OpenAI SDK retries 429/5xx by itself (2x, silently) on top of our
# own loop below, which multiplies the wait. We do the retrying, and we log it.
_client = (
    OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0, timeout=TIMEOUT)
    if API_KEY
    else None
)
_last = 0.0


def log(msg):
    if DEBUG:
        print(f"[lky] {msg}", flush=True)


def _kwargs(messages, system, model, max_tokens, temperature):
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    kwargs = dict(model=model or MAIN_MODEL, messages=msgs, max_tokens=max_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if REASONING:
        kwargs["reasoning_effort"] = REASONING
    return kwargs


def _create(kwargs):
    """Create the completion (or the stream, if kwargs['stream']) with visible retries."""
    global _last
    if _client is None:
        raise RuntimeError("API key missing. Set GEMINI_API_KEY (or LLM_API_KEY).")
    for attempt in range(MAX_ATTEMPTS):
        wait = MIN_INTERVAL - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            _last = time.time()
            return _client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "reasoning_effort" in kwargs:  # model does not accept it: retry without
                log(
                    f"{kwargs['model']} rejected reasoning_effort ({e}); retrying WITH "
                    "thinking enabled, which is much slower. Use a model that accepts it, "
                    "or set LLM_REASONING_EFFORT=low."
                )
                kwargs.pop("reasoning_effort")
                continue
            raise
        except RateLimitError as e:
            if "perday" in str(e).lower().replace("_", ""):
                raise RuntimeError(
                    f"Daily quota exhausted for {kwargs['model']}. Retrying will not help: "
                    "wait for the reset, switch LLM_MODEL, or use another API key."
                ) from e
            delay = min(5 * 2**attempt, 20)
            log(
                f"429 rate limit on {kwargs['model']}, waiting {delay}s "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS}): {str(e)[:200]}"
            )
            time.sleep(delay)
        except (APIConnectionError, InternalServerError) as e:  # includes timeouts
            delay = min(3 * 2**attempt, 15)
            log(
                f"{type(e).__name__} on {kwargs['model']}, waiting {delay}s "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            time.sleep(delay)
    raise RuntimeError(
        "LLM request failed after retries (rate limit or connection). "
        "See the [lky] lines in the terminal."
    )


def chat(messages, system=None, model=None, max_tokens=1500, temperature=None):
    """messages = [{'role': 'user'|'assistant', 'content': str}, ...]. Returns text."""
    kwargs = _kwargs(messages, system, model, max_tokens, temperature)
    t0 = time.time()
    r = _create(kwargs)
    log(f"{kwargs['model']} done in {time.time() - t0:.1f}s")
    return (r.choices[0].message.content or "").strip()


def chat_stream(messages, system=None, model=None, max_tokens=1500, temperature=None):
    """Same as chat() but yields text pieces as they arrive."""
    kwargs = _kwargs(messages, system, model, max_tokens, temperature)
    kwargs["stream"] = True
    t0 = time.time()
    stream = _create(kwargs)
    first = True
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            if first:
                log(f"{kwargs['model']} first token after {time.time() - t0:.1f}s")
                first = False
            yield piece
    log(f"{kwargs['model']} stream finished in {time.time() - t0:.1f}s")
