"""
Retrieval-Augmented Generation (RAG) layer for the music recommender.

Pipeline for one plain-English request:
  1. Parse    - Gemini turns the request into a structured preferences dict,
                using only genres/moods/tags that exist in the catalog.
  2. Retrieve - the existing rule-based scorer (recommend_songs) pulls the
                top candidate songs from data/songs.csv.
  3. Generate - Gemini picks and explains songs using ONLY the retrieved
                candidates.
  4. Guard    - any song Gemini names that was not retrieved is dropped.

If Gemini is unavailable (no API key, network/API error, blocked response),
each AI step falls back to a deterministic rule-based version so the app
still returns useful results.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from src.recommender import recommend_songs, DEFAULT_MODE
except ImportError:
    from recommender import recommend_songs, DEFAULT_MODE

logger = logging.getLogger("recommender.rag")

# A fast model that works on the Gemini API free tier. Override with the GEMINI_MODEL env var.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
REQUEST_TIMEOUT_MS = 30_000  # give up on a single Gemini call after 30 seconds
MAX_ATTEMPTS = 2             # one retry on temporary errors (429/5xx), then fall back
MAX_REQUEST_CHARS = 1000
NUM_CANDIDATES = 8   # how many songs the retriever hands to the model
NUM_PICKS = 5        # how many songs end up in the final answer


# ---------- Structured output schemas ----------

class ParsedPreferences(BaseModel):
    """What the parser step must return. Mirrors the user_prefs dict keys."""
    genre: Optional[str] = Field(description="One genre from the catalog list, or null if none fits")
    mood: Optional[str] = Field(description="One mood from the catalog list, or null if none fits")
    energy: float = Field(description="Target energy from 0.0 (very calm) to 1.0 (very intense)")
    likes_acoustic: bool
    likes_instrumental: bool
    likes_live: bool
    mood_tags: List[str] = Field(description="Zero or more tags from the catalog tag list")
    decade: Optional[int] = Field(description="Release decade like 1980 or 2010, or null")


class Pick(BaseModel):
    song_id: int
    reason: str


class GeneratedAnswer(BaseModel):
    """What the generation step must return."""
    summary: str
    picks: List[Pick]


# ---------- Result returned to the CLI ----------

@dataclass
class Recommendation:
    song: Dict
    retrieval_score: float
    reason: str


@dataclass
class RAGResult:
    request: str
    prefs: Dict
    candidates: List[Tuple[Dict, float, str]]
    recommendations: List[Recommendation]
    summary: str
    used_ai: bool                      # True only if both AI steps succeeded
    notes: List[str] = field(default_factory=list)  # guardrail / fallback messages


class RAGRecommender:
    def __init__(self, songs: List[Dict], client=None, mode: str = DEFAULT_MODE,
                 model: str = MODEL, use_ai: bool = True):
        self.songs = songs
        self.mode = mode
        self.model = model
        self.client_problem: Optional[str] = None  # why the AI is unavailable, shown once per request
        if not use_ai:
            self.client = None  # force the rule-based fallback (no API calls)
        elif client is not None:
            self.client = client
        else:
            self.client, self.client_problem = _make_client()
        self.vocab = _catalog_vocab(songs)

    # ----- public entry point -----

    def recommend(self, request: str) -> RAGResult:
        request = _validate_request(request)
        logger.info("Request: %r", request)
        notes: List[str] = []
        if self.client_problem:
            _note(notes, self.client_problem)

        prefs, parsed_by_ai = self._parse(request, notes)
        logger.info("Preferences (%s): %s", "gemini" if parsed_by_ai else "keyword fallback", prefs)

        candidates = recommend_songs(prefs, self.songs, k=NUM_CANDIDATES, mode=self.mode)
        logger.info("Retrieved %d candidates: %s", len(candidates),
                    [f"{s['id']}:{s['title']} ({score:.2f})" for s, score, _ in candidates])

        recs, summary, generated_by_ai = self._generate(request, prefs, candidates, notes)
        logger.info("Final picks: %s", [r.song["id"] for r in recs])

        return RAGResult(request, prefs, candidates, recs, summary,
                         used_ai=parsed_by_ai and generated_by_ai, notes=notes)

    # ----- step 1: parse -----

    def _parse(self, request: str, notes: List[str]) -> Tuple[Dict, bool]:
        system = (
            "You convert a listener's music request into search preferences for a song catalog.\n"
            f"Catalog genres: {', '.join(self.vocab['genres'])}\n"
            f"Catalog moods: {', '.join(self.vocab['moods'])}\n"
            f"Catalog mood tags: {', '.join(self.vocab['tags'])}\n"
            f"Catalog decades: {', '.join(str(d) for d in self.vocab['decades'])}\n"
            "Only use values from these lists; use null or an empty list when nothing fits. "
            "Set likes_instrumental when the listener wants no vocals or lyrics."
        )
        parsed = self._call_llm("parse", system, request, ParsedPreferences, notes)
        if parsed is None:
            return keyword_parse(request, self.vocab), False
        return self._clean_prefs(parsed, notes), True

    def _clean_prefs(self, parsed: ParsedPreferences, notes: List[str]) -> Dict:
        """Guardrail on the parser output: keep only values the catalog knows about."""
        prefs: Dict = {"energy": min(max(parsed.energy, 0.0), 1.0)}
        for key, allowed in (("genre", self.vocab["genres"]), ("mood", self.vocab["moods"])):
            value = getattr(parsed, key)
            if value is None:
                continue
            value = value.lower().strip()
            if value in allowed:
                prefs[key] = value
            else:
                _note(notes, f"Ignored unknown {key} '{value}' from parser")
        tags = [t.lower() for t in parsed.mood_tags if t.lower() in self.vocab["tags"]]
        if tags:
            prefs["mood_tags"] = tags
        if parsed.decade in self.vocab["decades"]:
            prefs["decade"] = parsed.decade
        for flag in ("likes_acoustic", "likes_instrumental", "likes_live"):
            if getattr(parsed, flag):
                prefs[flag] = True
        return prefs

    # ----- step 3: generate -----

    def _generate(self, request: str, prefs: Dict, candidates, notes: List[str]):
        if not candidates:
            return [], "No songs in the catalog to recommend.", False

        lines = []
        for song, score, reasons in candidates:
            lines.append(
                f"- id={song['id']} | \"{song['title']}\" by {song['artist']} | genre={song['genre']} "
                f"| mood={song['mood']} | energy={song['energy']} | tags={';'.join(song['mood_tags'])} "
                f"| decade={song['release_decade']} | instrumentalness={song['instrumentalness']} "
                f"| retrieval score={score:.2f} ({reasons})"
            )
        system = (
            "You are a music recommendation assistant. You may ONLY recommend songs from the "
            "retrieved candidate list you are given; never mention any other song. "
            f"Pick up to {NUM_PICKS} songs that best fit the listener's request, best first. "
            "The retrieval score is a rough rule-based match; you may reorder songs when the "
            "request's wording calls for it. Each reason should be one sentence that ties the "
            "song's attributes to the request. The summary is one or two sentences."
        )
        user = (f"Listener request: {request}\n\nRetrieved candidates:\n" + "\n".join(lines))

        answer = self._call_llm("generate", system, user, GeneratedAnswer, notes)
        if answer is None:
            return _fallback_recs(candidates), "Top matches from the rule-based scorer.", False

        recs = apply_guardrail(answer.picks, candidates, notes)
        if not recs:
            _note(notes, "No valid picks from Gemini; showing rule-based matches instead")
            return _fallback_recs(candidates), "Top matches from the rule-based scorer.", False
        return recs, answer.summary, True

    # ----- shared Gemini call with error handling -----

    def _call_llm(self, step: str, system: str, user: str, schema, notes: List[str]):
        """Return the parsed schema instance, or None if Gemini can't be used."""
        if self.client is None:
            logger.info("%s: no AI client available, using fallback", step)
            return None
        from google.genai import errors, types
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,  # keep answers consistent between runs
                ),
            )
        except errors.ClientError as e:  # 4xx: bad key, quota, bad request
            if e.code == 429:
                _note(notes, f"{step}: Gemini free-tier rate limit reached, using fallback")
            elif e.code in (400, 401, 403) and "key" in str(e).lower():
                _note(notes, f"{step}: Gemini rejected the API key, using fallback")
                self.client = None  # don't retry a bad key on the next step
            else:
                _note(notes, f"{step}: Gemini API error {e.code}, using fallback")
            return None
        except errors.ServerError as e:  # 5xx: Gemini is down or overloaded
            _note(notes, f"{step}: Gemini server error {e.code}, using fallback")
            return None
        except Exception:
            # Network problems and anything unexpected; full traceback goes to the log file
            logger.warning("%s: unexpected error calling Gemini", step, exc_info=True)
            _note(notes, f"{step}: could not get a response from Gemini, using fallback")
            return None

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) if feedback else None
            if reason:
                _note(notes, f"{step}: Gemini blocked the request ({reason}), using fallback")
            else:
                _note(notes, f"{step}: response did not match the expected format, using fallback")
            return None
        logger.debug("%s usage: %s", step, getattr(response, "usage_metadata", None))
        return parsed


# ---------- Guardrail ----------

def apply_guardrail(picks: List[Pick], candidates, notes: List[str]) -> List[Recommendation]:
    """Keep only picks that refer to retrieved songs; drop duplicates and extras."""
    by_id = {song["id"]: (song, score) for song, score, _ in candidates}
    recs, seen = [], set()
    for pick in picks:
        if pick.song_id not in by_id:
            _note(notes, f"Guardrail: dropped song id {pick.song_id} (not in retrieved set)")
            continue
        if pick.song_id in seen:
            continue
        seen.add(pick.song_id)
        song, score = by_id[pick.song_id]
        recs.append(Recommendation(song, score, pick.reason))
        if len(recs) == NUM_PICKS:
            break
    return recs


# ---------- Fallbacks (no AI) ----------

ENERGY_HIGH = ("energetic", "workout", "gym", "party", "hype", "pump", "intense", "fast", "dance", "running")
ENERGY_LOW = ("chill", "calm", "relax", "sleep", "study", "studying", "quiet", "slow", "peaceful", "soft")


def keyword_parse(request: str, vocab: Dict) -> Dict:
    """Rule-based parser used when Gemini is unavailable."""
    text = request.lower().replace("lo-fi", "lofi").replace("hip-hop", "hip hop")
    prefs: Dict = {}
    for key, values in (("genre", vocab["genres"]), ("mood", vocab["moods"])):
        # Treat hyphens as part of a word so "k-pop" doesn't match "pop"
        match = next((v for v in values if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text)), None)
        if match:
            prefs[key] = match
    words = set(re.findall(r"[a-z]+", text))
    if words & set(ENERGY_HIGH):
        prefs["energy"] = 0.85
    elif words & set(ENERGY_LOW):
        prefs["energy"] = 0.3
    else:
        prefs["energy"] = 0.5
    tags = [t for t in vocab["tags"] if t in words]
    if tags:
        prefs["mood_tags"] = tags
    decade = re.search(r"\b(?:19|20)?([0-9])0s\b", text)
    if decade:
        prefs["decade"] = _decade_from(decade.group(0), vocab["decades"])
        if prefs["decade"] is None:
            del prefs["decade"]
    if "acoustic" in words:
        prefs["likes_acoustic"] = True
    if "instrumental" in words or re.search(r"\bno (vocals|lyrics|singing)\b", text):
        prefs["likes_instrumental"] = True
    if "live" in words:
        prefs["likes_live"] = True
    return prefs


def _fallback_recs(candidates) -> List[Recommendation]:
    return [Recommendation(song, score, reasons) for song, score, reasons in candidates[:NUM_PICKS]]


# ---------- Helpers ----------

def _validate_request(request: str) -> str:
    if request is None or not request.strip():
        raise ValueError("Please describe what you want to listen to.")
    request = request.strip()
    if len(request) > MAX_REQUEST_CHARS:
        raise ValueError(f"Request is too long ({len(request)} characters, max {MAX_REQUEST_CHARS}).")
    return request


def _catalog_vocab(songs: List[Dict]) -> Dict:
    return {
        "genres": sorted({s["genre"] for s in songs}),
        "moods": sorted({s["mood"] for s in songs}),
        "tags": sorted({t for s in songs for t in s["mood_tags"]}),
        "decades": sorted({s["release_decade"] for s in songs}),
    }


def _decade_from(token: str, decades: List[int]) -> Optional[int]:
    digits = token.rstrip("s")
    if len(digits) == 4:
        value = int(digits)
    else:  # "80s" -> 1980, "10s" -> 2010
        value = (1900 if int(digits) >= 30 else 2000) + int(digits)
    return value if value in decades else None


def _note(notes: List[str], message: str) -> None:
    logger.warning(message)
    notes.append(message)


def _make_client():
    """Create a Gemini client. Returns (client, None) or (None, reason it's unavailable)."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return None, "No GEMINI_API_KEY set; using rule-based fallback"
    try:
        from google import genai
        from google.genai import types
        # Reads GEMINI_API_KEY / GOOGLE_API_KEY. The timeout and retry cap stop the app
        # from hanging when a model is overloaded.
        options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS,
                                    retry_options=types.HttpRetryOptions(attempts=MAX_ATTEMPTS))
        return genai.Client(http_options=options), None
    except Exception as e:  # missing package or bad configuration
        logger.warning("Gemini client unavailable", exc_info=True)
        return None, f"Gemini unavailable ({type(e).__name__}); using rule-based fallback"
