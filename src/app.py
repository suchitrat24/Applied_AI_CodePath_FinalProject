"""
Browser UI for the Music Recommender (Streamlit).

  streamlit run src/app.py

Uses the same RAG pipeline as the command line app (src/main.py).
"""

import os
import sys

import streamlit as st

# Make "src.*" imports work no matter which folder streamlit is started from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # so logs/ and data/ resolve the same way as the CLI

from src.main import setup_logging, SONGS_PATH, LOG_PATH  # noqa: E402
from src.recommender import load_songs, SCORING_MODES, DEFAULT_MODE  # noqa: E402
from src.rag import RAGRecommender  # noqa: E402

EXAMPLES = [
    "chill lofi for studying late at night, no vocals",
    "80s synthwave for a nostalgic night drive",
    "something sad and acoustic for a rainy day",
]


@st.cache_resource
def init_logging() -> bool:
    """Set up logging once per server (Streamlit reruns this script on every click)."""
    setup_logging()
    return True


@st.cache_data
def get_songs():
    return load_songs(SONGS_PATH)


st.set_page_config(page_title="Music Recommender", page_icon="🎵")
init_logging()

st.title("🎵 Music Recommender")
st.caption("Describe what you want to hear. Gemini interprets the request, the rule-based "
           "retriever finds matching songs in the catalog, and Gemini explains its picks.")

try:
    songs = get_songs()
except (OSError, KeyError, ValueError) as e:
    st.error(f"Could not load the song catalog from {SONGS_PATH}: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    modes = list(SCORING_MODES)
    mode = st.selectbox("Retriever scoring mode", modes, index=modes.index(DEFAULT_MODE))
    use_ai = st.toggle("Use Gemini", value=True,
                       help="Turn off to use only the rule-based fallback (no API calls).")
    if not os.environ.get("GEMINI_API_KEY"):
        st.info("No GEMINI_API_KEY set, so the rule-based fallback will be used.")
    st.caption(f"{len(songs)} songs in the catalog")

example = st.selectbox("Try an example (optional)", [""] + EXAMPLES)
with st.form("request_form"):
    request = st.text_area("What do you want to listen to?", value=example, height=80)
    submitted = st.form_submit_button("Recommend", type="primary")

if submitted:
    with st.spinner("Finding songs..."):
        try:
            result = RAGRecommender(songs, mode=mode, use_ai=use_ai).recommend(request)
        except ValueError as e:  # input guardrail
            st.error(f"Invalid request: {e}")
            st.stop()

    source = "Gemini (RAG)" if result.used_ai else "rule-based fallback"
    st.subheader("Recommendations")
    st.markdown(f"**Answer source:** {source}")
    st.write(result.summary)

    st.dataframe(
        [
            {
                "#": rank,
                "Title": r.song["title"],
                "Artist": r.song["artist"],
                "Genre": r.song["genre"],
                "Mood": r.song["mood"],
                "Match": round(r.retrieval_score, 2),
                "Why": r.reason,
            }
            for rank, r in enumerate(result.recommendations, start=1)
        ],
        hide_index=True,
        width="stretch",
    )

    for note in result.notes:
        st.warning(note)

    with st.expander("How the request was interpreted"):
        st.json(result.prefs)
        st.write(f"Retrieved {len(result.candidates)} candidate songs from the catalog.")
        st.dataframe(
            [
                {"Title": s["title"], "Artist": s["artist"], "Score": round(score, 2), "Reasons": why}
                for s, score, why in result.candidates
            ],
            hide_index=True,
            width="stretch",
        )
    st.caption(f"Details logged to {LOG_PATH}")
