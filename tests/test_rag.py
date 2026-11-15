"""Tests for the RAG pipeline. A fake Gemini client is used, so no network or API key is needed."""

from types import SimpleNamespace

import pytest
from google.genai import errors

from src.recommender import load_songs
from src.rag import (
    RAGRecommender, ParsedPreferences, GeneratedAnswer, Pick,
    apply_guardrail, keyword_parse, _catalog_vocab, NUM_PICKS,
)

SONGS = load_songs("data/songs.csv")
SONG_IDS = {s["id"] for s in SONGS}


class FakeClient:
    """Mimics client.models.generate_content, returning canned outputs per schema."""

    def __init__(self, prefs=None, answer=None, error=None, block_reason=None):
        self.prefs, self.answer, self.error, self.block_reason = prefs, answer, error, block_reason
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.block_reason:
            return SimpleNamespace(parsed=None, prompt_feedback=SimpleNamespace(block_reason=self.block_reason))
        schema = kwargs["config"].response_schema
        output = self.prefs if schema is ParsedPreferences else self.answer
        if callable(output):  # let a test build the answer from the prompt
            output = output(kwargs)
        return SimpleNamespace(parsed=output, prompt_feedback=None)


def lofi_prefs(**overrides):
    base = dict(genre="lofi", mood="chill", energy=0.35, likes_acoustic=False,
                likes_instrumental=True, likes_live=False, mood_tags=["cozy"], decade=None)
    base.update(overrides)
    return ParsedPreferences(**base)


def answer_from_candidates(kwargs):
    """Pick the first two candidate ids that appear in the prompt."""
    import re
    ids = [int(i) for i in re.findall(r"id=(\d+)", kwargs["contents"])]
    return GeneratedAnswer(summary="Cozy study picks.",
                           picks=[Pick(song_id=i, reason="fits") for i in ids[:2]])


# ---------- full pipeline ----------

def test_rag_uses_gemini_output_and_retrieved_songs():
    client = FakeClient(prefs=lofi_prefs(), answer=answer_from_candidates)
    result = RAGRecommender(SONGS, client=client).recommend("chill lofi for studying, no vocals")

    assert result.used_ai
    assert result.prefs["genre"] == "lofi" and result.prefs["likes_instrumental"]
    assert result.summary == "Cozy study picks."
    candidate_ids = {s["id"] for s, _, _ in result.candidates}
    assert [r.song["id"] for r in result.recommendations] and \
        all(r.song["id"] in candidate_ids for r in result.recommendations)
    # Retrieval happened before generation: the top candidate is a lofi song
    assert result.candidates[0][0]["genre"] == "lofi"
    assert len(client.calls) == 2


def test_generation_prompt_contains_only_retrieved_songs():
    client = FakeClient(prefs=lofi_prefs(), answer=answer_from_candidates)
    result = RAGRecommender(SONGS, client=client).recommend("chill lofi")
    prompt = client.calls[1]["contents"]
    retrieved_titles = {s["title"] for s, _, _ in result.candidates}
    for song in SONGS:
        assert (song["title"] in prompt) == (song["title"] in retrieved_titles)


def test_guardrail_drops_hallucinated_song():
    first_id = None

    def answer(kwargs):
        nonlocal first_id
        import re
        first_id = int(re.search(r"id=(\d+)", kwargs["contents"]).group(1))
        return GeneratedAnswer(summary="x", picks=[Pick(song_id=9999, reason="made up"),
                                                   Pick(song_id=first_id, reason="real")])

    result = RAGRecommender(SONGS, client=FakeClient(prefs=lofi_prefs(), answer=answer)).recommend("lofi")
    assert [r.song["id"] for r in result.recommendations] == [first_id]
    assert any("9999" in n for n in result.notes)


def test_all_picks_invalid_falls_back_to_retrieval():
    bad = GeneratedAnswer(summary="x", picks=[Pick(song_id=9999, reason="made up")])
    result = RAGRecommender(SONGS, client=FakeClient(prefs=lofi_prefs(), answer=bad)).recommend("lofi")
    assert not result.used_ai
    assert len(result.recommendations) == NUM_PICKS
    assert [r.song["id"] for r in result.recommendations] == [s["id"] for s, _, _ in result.candidates[:NUM_PICKS]]


def test_parser_output_is_sanitized():
    prefs = lofi_prefs(genre="k-pop", mood="CHILL", energy=1.7, mood_tags=["cozy", "spooky"], decade=1950)
    client = FakeClient(prefs=prefs, answer=answer_from_candidates)
    result = RAGRecommender(SONGS, client=client).recommend("something")
    assert "genre" not in result.prefs             # unknown genre ignored
    assert result.prefs["mood"] == "chill"         # normalized case
    assert result.prefs["energy"] == 1.0           # clamped
    assert result.prefs["mood_tags"] == ["cozy"]   # unknown tag dropped
    assert "decade" not in result.prefs            # decade not in catalog


# ---------- error handling / fallback ----------

def _bad_key_error():
    return errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid.",
                                              "status": "INVALID_ARGUMENT"}})


@pytest.mark.parametrize("case", ["no_key", "server_error", "rate_limit", "bad_key", "network", "blocked"])
def test_falls_back_when_gemini_unavailable(case, monkeypatch):
    client = None
    if case == "no_key":
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    elif case == "server_error":
        client = FakeClient(error=errors.ServerError(503, {"error": {"message": "overloaded"}}))
    elif case == "rate_limit":
        client = FakeClient(error=errors.ClientError(429, {"error": {"message": "quota"}}))
    elif case == "bad_key":
        client = FakeClient(error=_bad_key_error())
    elif case == "network":
        client = FakeClient(error=ConnectionError("no internet"))
    else:
        client = FakeClient(block_reason="SAFETY")

    result = RAGRecommender(SONGS, client=client).recommend("chill lofi for studying")
    assert not result.used_ai
    assert result.prefs.get("genre") == "lofi"     # keyword parser still understood it
    assert len(result.recommendations) == NUM_PICKS
    assert result.notes                            # the user is told why


def test_bad_key_skips_second_call():
    client = FakeClient(error=_bad_key_error())
    result = RAGRecommender(SONGS, client=client).recommend("lofi")
    assert len(client.calls) == 1
    assert len(result.notes) == 1


def test_offline_mode_makes_no_calls():
    client = FakeClient(prefs=lofi_prefs(), answer=answer_from_candidates)
    result = RAGRecommender(SONGS, client=client, use_ai=False).recommend("lofi")
    assert client.calls == [] and not result.used_ai


@pytest.mark.parametrize("bad", ["", "   ", "x" * 1001])
def test_rejects_invalid_requests(bad):
    with pytest.raises(ValueError):
        RAGRecommender(SONGS, client=FakeClient()).recommend(bad)


# ---------- unit tests ----------

def test_keyword_parse():
    vocab = _catalog_vocab(SONGS)
    prefs = keyword_parse("Upbeat hip-hop for the gym, 2010s, no vocals", vocab)
    assert prefs["genre"] == "hip hop"
    assert prefs["energy"] == 0.85
    assert prefs["decade"] == 2010
    assert prefs["likes_instrumental"]


def test_keyword_parse_does_not_match_inside_hyphenated_genre():
    prefs = keyword_parse("upbeat k-pop for a dance party", _catalog_vocab(SONGS))
    assert "genre" not in prefs


def test_apply_guardrail_dedupes_and_caps():
    candidates = [(s, 1.0, "") for s in SONGS[:8]]
    picks = [Pick(song_id=SONGS[0]["id"], reason="a")] * 3 + \
            [Pick(song_id=s["id"], reason="b") for s in SONGS[1:8]]
    recs = apply_guardrail(picks, candidates, [])
    ids = [r.song["id"] for r in recs]
    assert len(ids) == len(set(ids)) == NUM_PICKS
