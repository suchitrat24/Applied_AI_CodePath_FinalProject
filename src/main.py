"""
Command line runner for the Music Recommender.

  python -m src.main "chill music for studying, no vocals"   # RAG recommendation
  python -m src.main                                         # prompts for a request
  python -m src.main --demo [--mode mood_first]              # rule-based profile demo
"""

import argparse
import logging
import os
import textwrap

try:
    from src.recommender import load_songs, recommend_songs, SCORING_MODES, DEFAULT_MODE
    from src.rag import RAGRecommender
except ImportError:
    from recommender import load_songs, recommend_songs, SCORING_MODES, DEFAULT_MODE
    from rag import RAGRecommender

SONGS_PATH = "data/songs.csv"
LOG_PATH = "logs/app.log"


def setup_logging() -> None:
    """Log everything to logs/app.log; only errors reach the console (warnings are shown as notes)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    # Keep HTTP client chatter out of the log
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_table(headers, rows, widths) -> None:
    divider = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row(cells):
        # Wrap each cell to its column width, then print the wrapped lines side by side
        wrapped = [textwrap.wrap(str(c), w) or [""] for c, w in zip(cells, widths)]
        for i in range(max(len(lines) for lines in wrapped)):
            parts = [(lines[i] if i < len(lines) else "").ljust(w) for lines, w in zip(wrapped, widths)]
            print("| " + " | ".join(parts) + " |")

    print(divider)
    row(headers)
    print(divider.replace("-", "="))
    for cells in rows:
        row(cells)
        print(divider)


def print_recommendations(label: str, user_prefs: dict, songs: list, mode: str, k: int = 5) -> None:
    """Print the top k songs for a profile as an ASCII table with scores and reasons."""
    print(f"\n=== {label} ===")
    print("Profile: " + ", ".join(f"{key}={value}" for key, value in user_prefs.items()))
    rows = [
        [rank, song["title"], song["artist"], f"{score:.2f}", explanation]
        for rank, (song, score, explanation) in enumerate(recommend_songs(user_prefs, songs, k, mode), start=1)
    ]
    _print_table(["#", "Title", "Artist", "Score", "Reasons"], rows, [3, 22, 16, 6, 46])


def run_rag(request: str, songs: list, mode: str) -> None:
    """Answer a plain-English request with the RAG pipeline and print the result."""
    try:
        result = RAGRecommender(songs, mode=mode).recommend(request)
    except ValueError as e:  # input guardrail
        print(f"Invalid request: {e}")
        return

    print(f"\nRequest: {result.request}")
    print("Interpreted as: " + ", ".join(f"{k}={v}" for k, v in result.prefs.items()))
    print(f"Retrieved {len(result.candidates)} candidate songs from the catalog.")
    print(f"Answer source: {'Gemini (RAG)' if result.used_ai else 'rule-based fallback'}")
    print(f"\n{result.summary}")
    rows = [
        [rank, r.song["title"], r.song["artist"], f"{r.retrieval_score:.2f}", r.reason]
        for rank, r in enumerate(result.recommendations, start=1)
    ]
    _print_table(["#", "Title", "Artist", "Match", "Why"], rows, [3, 22, 16, 5, 50])
    for note in result.notes:
        print(f"note: {note}")
    print(f"(details logged to {LOG_PATH})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Music recommender with retrieval-augmented generation.")
    parser.add_argument("request", nargs="*", help="what you want to listen to, in plain English")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=list(SCORING_MODES),
                        help="retriever scoring mode")
    parser.add_argument("--demo", action="store_true",
                        help="run the built-in rule-based profiles instead of a request")
    args = parser.parse_args()

    setup_logging()
    try:
        songs = load_songs(SONGS_PATH)
    except (OSError, KeyError, ValueError) as e:
        logging.getLogger(__name__).error("Could not load %s: %s", SONGS_PATH, e)
        print(f"Could not load the song catalog from {SONGS_PATH}: {e}")
        return
    print(f"Loaded songs: {len(songs)}")
    print(f"Scoring mode: {args.mode}")

    if not args.demo:
        try:
            # Only prompt when no argument was given; an empty "" argument goes to validation
            request = " ".join(args.request) if args.request else input("What do you want to listen to? ")
        except (EOFError, KeyboardInterrupt):
            print("\nNo request given.")
            return
        run_rag(request, songs, args.mode)
        return
    mode = args.mode

    profiles = {
        # Normal profiles
        "High-Energy Pop": {"genre": "pop", "mood": "happy", "energy": 0.8},
        "Chill Lofi": {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True},
        "Deep Intense Rock": {"genre": "rock", "mood": "intense", "energy": 0.9},
        # Edge cases: try to trick the scoring
        "Sad but High Energy": {"genre": "folk", "mood": "sad", "energy": 0.9},
        "Unknown Genre": {"genre": "k-pop", "mood": "happy", "energy": 0.7},
        "Loud Acoustic Fan": {"genre": "metal", "mood": "angry", "energy": 0.95, "likes_acoustic": True},
        # Uses the advanced features
        "Nostalgic Night Drive": {
            "genre": "synthwave",
            "mood": "moody",
            "energy": 0.6,
            "popularity": 60,
            "decade": 1980,
            "mood_tags": ["nostalgic", "dreamy"],
            "likes_instrumental": True,
        },
    }

    for label, user_prefs in profiles.items():
        print_recommendations(label, user_prefs, songs, mode)


if __name__ == "__main__":
    main()
