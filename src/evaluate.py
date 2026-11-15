"""
Evaluation harness: runs the full RAG system on predefined requests and scores the results.

  python -m src.evaluate            # uses Gemini if GEMINI_API_KEY is set, else the fallback
  python -m src.evaluate --offline  # force the rule-based fallback (free, no API calls)

Every normal case is also checked against the system-wide rules:
  - every recommended song was in the retrieved set (grounding)
  - no duplicate songs, at most NUM_PICKS songs, at least one song
  - every song has a non-empty reason
Exit code is 0 only if every case passes.
"""

import argparse
import logging
import sys
from typing import Callable, List, Optional, Tuple

try:
    from src.recommender import load_songs
    from src.rag import RAGRecommender, RAGResult, NUM_PICKS
except ImportError:
    from recommender import load_songs
    from rag import RAGRecommender, RAGResult, NUM_PICKS

Check = Tuple[str, Callable[[RAGResult], bool]]


def top(result: RAGResult) -> dict:
    return result.recommendations[0].song


# (name, request, case-specific checks). checks=None means the request must be rejected.
CASES: List[Tuple[str, str, Optional[List[Check]]]] = [
    ("study lofi", "chill lofi for studying late at night, no vocals", [
        ("top pick is lofi", lambda r: top(r)["genre"] == "lofi"),
        ("top pick is instrumental", lambda r: top(r)["instrumentalness"] > 0.6),
    ]),
    ("gym rock", "high energy rock for the gym", [
        ("top pick is rock", lambda r: top(r)["genre"] == "rock"),
        ("top pick energy >= 0.7", lambda r: top(r)["energy"] >= 0.7),
    ]),
    ("night drive", "80s synthwave for a night drive", [
        ("top pick is synthwave", lambda r: top(r)["genre"] == "synthwave"),
    ]),
    ("happy pop", "happy upbeat pop songs", [
        ("top pick is pop", lambda r: top(r)["genre"] == "pop"),
        ("top pick is happy", lambda r: top(r)["mood"] == "happy"),
    ]),
    ("relaxing jazz", "some relaxing jazz for dinner", [
        ("top pick is jazz", lambda r: top(r)["genre"] == "jazz"),
    ]),
    ("sad acoustic", "sad acoustic songs", [
        ("a sad song is in the picks", lambda r: any(x.song["mood"] == "sad" for x in r.recommendations)),
    ]),
    ("genre not in catalog", "upbeat k-pop for a dance party", [
        ("unknown genre not used", lambda r: r.prefs.get("genre") not in ("k-pop", "kpop")),
    ]),
    ("empty request", "   ", None),
    ("too-long request", "music " * 300, None),
]


def system_checks(result: RAGResult) -> List[Check]:
    retrieved = {s["id"] for s, _, _ in result.candidates}
    ids = [r.song["id"] for r in result.recommendations]
    return [
        ("returns 1-%d songs" % NUM_PICKS, lambda r: 1 <= len(ids) <= NUM_PICKS),
        ("all picks were retrieved", lambda r: set(ids) <= retrieved),
        ("no duplicate picks", lambda r: len(ids) == len(set(ids))),
        ("every pick has a reason", lambda r: all(x.reason.strip() for x in r.recommendations)),
    ]


def run(offline: bool) -> bool:
    songs = load_songs("data/songs.csv")
    recommender = RAGRecommender(songs, use_ai=not offline)

    total_checks = passed_checks = passed_cases = ai_answers = 0
    print(f"{'Case':<22} {'Result':<6} {'Source':<9} Details")
    print("-" * 78)

    for name, request, checks in CASES:
        failures: List[str] = []
        source = "-"
        if checks is None:  # must be rejected by input validation
            total_checks += 1
            try:
                recommender.recommend(request)
                failures.append("request was not rejected")
            except ValueError:
                passed_checks += 1
        else:
            try:
                result = recommender.recommend(request)
            except Exception as e:  # a crash fails the case but not the whole run
                failures.append(f"crashed: {type(e).__name__}: {e}")
                result = None
            if result is not None:
                source = "gemini" if result.used_ai else "fallback"
                ai_answers += result.used_ai
                for label, check in system_checks(result) + checks:
                    total_checks += 1
                    try:
                        ok = bool(check(result))
                    except Exception:
                        ok = False
                    if ok:
                        passed_checks += 1
                    else:
                        failures.append(label)

        status = "PASS" if not failures else "FAIL"
        passed_cases += not failures
        details = "; ".join(failures) if failures else (request.strip()[:40] or "(blank)")
        print(f"{name:<22} {status:<6} {source:<9} {details}")

    answered = sum(1 for _, _, c in CASES if c is not None)
    print("-" * 78)
    print(f"Cases passed:  {passed_cases}/{len(CASES)}")
    print(f"Checks passed: {passed_checks}/{total_checks} ({100 * passed_checks / total_checks:.0f}%)")
    print(f"Answered by Gemini: {ai_answers}/{answered}"
          + ("  (offline mode)" if offline else ""))
    return passed_cases == len(CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the recommender on predefined requests.")
    parser.add_argument("--offline", action="store_true", help="use the rule-based fallback only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)  # keep the report readable
    sys.exit(0 if run(args.offline) else 1)


if __name__ == "__main__":
    main()
