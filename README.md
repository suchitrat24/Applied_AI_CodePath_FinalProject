# Music Recommender with RAG

## Music Recommender Simulation (Modules 1-3)

My original project was the **Music Recommender Simulation**. It loaded a small catalog of songs from a CSV file and scored each one against a hard-coded user profile (favorite genre, mood, target energy, and a few extras like acoustic or instrumental). It returned the top 5 songs with a reason for each score. It had three scoring modes (`genre_first`, `mood_first`, `energy_focused`) and a diversity penalty so the same artist or genre didn't fill the whole list. The catch was that you had to write the profile as a Python dictionary, so you couldn't just ask it for music.

## Title and Summary

For the final project I turned that simulation into a recommender you can talk to in plain English. You type something like *"chill lofi for studying late at night, no vocals"* and the app recommends songs from its catalog, with a sentence about why each one fits.

I think this matters because people don't think about music in terms of `energy=0.3` or `genre=lofi`. They think "I need something for studying" or "something sad for a rainy day." I used AI to connect the way people actually talk to the structured data my recommender already understood, without letting the AI make up songs that aren't in the catalog.

The AI feature I chose is **Retrieval-Augmented Generation (RAG)**. Gemini doesn't answer from its own memory of music. It only answers using songs my retriever pulls from `data/songs.csv`.

## Architecture Overview

The full diagram is in [diagrams/architecture.mmd](diagrams/architecture.mmd). Here's how a request moves through the system:

1. **Input:** the user types a request on the command line (`src/main.py`).
2. **Preference Parser (Gemini):** Gemini reads the request and turns it into structured preferences like genre, mood, energy, mood tags, decade, and instrumental. I give it the exact list of genres, moods and tags that exist in my catalog, so it can only pick real values.
3. **Retriever:** my original scoring code from Modules 1-3 (`score_song` / `recommend_songs`) ranks all the songs against those preferences and keeps the top 8 candidates.
4. **Recommendation Generator (Gemini):** Gemini gets the original request plus *only* those 8 retrieved songs. It picks up to 5 and writes a reason for each.
5. **Guardrail:** my code checks every song Gemini picked. If a song ID wasn't in the retrieved set, the code drops it and logs a warning. If nothing valid is left, the app shows the retriever's top matches instead.
6. **Output:** the ranked songs with explanations, plus a line showing whether the answer came from Gemini or the fallback.

Two things run alongside the main flow:

- **Fallback:** if Gemini can't be reached (no API key, a network error, or a refusal), a keyword-based parser and the rule-based scorer take over, so you still get recommendations.
- **Logging:** every step writes to `logs/app.log`.

People and tests check the AI in two places. Automated tests (pytest) check the retriever, the guardrail and the fallbacks. A person reads the output and can review the log to see exactly what Gemini was given and what it returned.

## Setup Instructions

You need Python 3.10 or newer.

1. **Clone the repo and go into it:**
   ```bash
   git clone https://github.com/suchitrat24/Applied_AI_CodePath_FinalProject.git
   cd Applied_AI_CodePath_FinalProject
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate it:**
   ```bash
   source .venv/bin/activate        # macOS / Linux
   source .venv/Scripts/activate    # Git Bash on Windows
   .\.venv\Scripts\Activate.ps1     # Windows PowerShell
   ```
4. **Install the dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Add a free Gemini API key** (optional, but needed for the AI part). Go to https://aistudio.google.com/apikey, sign in with a Google account, and click **Create API key**. No credit card is needed. Then:
   ```bash
   export GEMINI_API_KEY="your-key"          # macOS / Linux / Git Bash
   $env:GEMINI_API_KEY = "your-key"          # Windows PowerShell
   ```
   This sets the key for the current terminal only, so it never ends up in the code. Without a key, the app still runs, using the rule-based fallback.
6. **Run it:**
   ```bash
   python -m src.main "chill lofi for studying late at night, no vocals"
   python -m src.main                         # asks you to type a request
   python -m src.main "80s synthwave" --mode mood_first
   python -m src.main --demo                  # the original Module 1-3 profile demo
   ```
7. **Run the tests and the evaluation:**
   ```bash
   python -m pytest                    # unit tests, offline
   python -m src.evaluate --offline    # evaluation on 9 predefined requests, no API calls
   python -m src.evaluate              # same evaluation, using Gemini (needs the API key)
   ```

## Sample Interactions

These are real runs with Gemini (`gemini-3.1-flash-lite`). After each one, I compare it with what the rule-based fallback gives for the same request (no API key), which shows what the AI adds. Long reasons are shortened slightly to fit.

**Example 1: a clear request**

```
$ python -m src.main "chill lofi for studying late at night, no vocals"
Interpreted as: energy=0.2, genre=lofi, mood=chill, mood_tags=['calm', 'cozy'], likes_instrumental=True
Retrieved 8 candidate songs from the catalog.
Answer source: Gemini (RAG)

These instrumental lofi and ambient tracks are perfect for maintaining focus during late-night
study sessions. They provide a calm, cozy atmosphere without any distracting vocals.

| # | Title                | Artist         | Match | Why                                                                              |
|---|----------------------|----------------|-------|----------------------------------------------------------------------------------|
| 1 | Library Rain         | Paper Lanterns | 4.65  | The cozy and melancholic atmosphere ... ideal companion for late-night studying. |
| 2 | Midnight Coding      | LoRoom         | 4.08  | This dreamy lofi track offers a perfect chill vibe to help you stay focused.     |
| 3 | Focus Flow           | LoRoom         | 2.40  | Designed specifically for focus, this instrumental track provides a calm rhythm. |
| 4 | Spacewalk Thoughts   | Orbit Bloom    | 2.42  | The spacey, ambient nature creates a quiet, distraction-free environment.        |
| 5 | Moonlit Sonata Redux | Clara Vane     | 1.78  | This peaceful classical piece offers a very low-energy, instrumental soundscape. |
```

*Compared to the fallback:* the fallback found the same top two songs, because the request uses my catalog's exact words. Gemini went further. It understood "late at night" as lower energy (0.2 instead of 0.3), added the "calm" and "cozy" tags, and moved Focus Flow above Spacewalk Thoughts because it's made for studying, even though its retrieval score is slightly lower. Its reasons are written for the listener instead of being a list of score parts like "genre match (+2.0)".

**Example 2: a request that needs judgment**

```
$ python -m src.main "80s synthwave for a nostalgic night drive"
Interpreted as: energy=0.5, genre=synthwave, mood=nostalgic, mood_tags=['nostalgic'], decade=1980
Retrieved 8 candidate songs from the catalog.
Answer source: Gemini (RAG)

For your nostalgic night drive, I have selected the perfect 80s synthwave track that captures
the dreamy atmosphere you are looking for.

| # | Title            | Artist    | Match | Why                                                                                              |
|---|------------------|-----------|-------|--------------------------------------------------------------------------------------------------|
| 1 | Night Drive Loop | Neon Echo | 3.55  | This track perfectly captures the 80s synthwave aesthetic with its nostalgic, dreamy soundscape. |
```

*Compared to the fallback:* the fallback returned 5 songs, and #2 was a country song (Dusty Porch Swing) that only matched because its mood is "nostalgic." Gemini looked at the same 8 retrieved songs and returned just **one**, the only real 80s synthwave track in my catalog, instead of padding the list with poor matches. I didn't expect this. I asked for "up to 5" songs, and it decided one good answer was better than five weak ones.

**Example 3: a vague request**

```
$ python -m src.main "something sad and acoustic for a rainy day"
Interpreted as: energy=0.2, genre=folk, mood=sad, mood_tags=['melancholic', 'cozy'], likes_acoustic=True
Retrieved 8 candidate songs from the catalog.
Answer source: Gemini (RAG)

These selections offer a melancholic and acoustic atmosphere perfect for a rainy day. They focus
on slow, reflective sounds that capture a sense of solitude and calm.

| # | Title                | Artist         | Match | Why                                                                                   |
|---|----------------------|----------------|-------|---------------------------------------------------------------------------------------|
| 1 | Empty Train Station  | Hollow Pines   | 4.70  | This folk track perfectly captures a sad and lonely mood that pairs with the rain.    |
| 2 | Library Rain         | Paper Lanterns | 1.95  | The melancholic tone and lofi acoustic qualities create a cozy environment indoors.   |
| 3 | Moonlit Sonata Redux | Clara Vane     | 1.78  | This classical piece provides a peaceful, deeply melancholic backdrop for a rainy day. |
| 4 | Coffee Shop Stories  | Slow Stereo    | 1.63  | Its relaxed jazz instrumentation and nostalgic feel offer a comforting experience.    |
| 5 | Spacewalk Thoughts   | Orbit Bloom    | 1.42  | The dreamy, chill ambient soundscape creates a reflective space for watching rain.    |
```

*Compared to the fallback:* the fallback didn't understand "rainy day," so it guessed a medium energy (0.5) and ranked an upbeat country song second. Gemini turned "rainy day" into low energy (0.2) and the "melancholic" tag, and guessed that folk fits "sad and acoustic." That pushed Empty Train Station's score from 2.30 to 4.70 and filled the rest of the list with slow, melancholic songs.

## Guardrails in Action

Each example shows the input, what the system does, and the result.

**1. Input validation: a blank request**

- **Input:** `python -m src.main "   "`
- **Behavior:** the request is checked before any AI call is made. It's empty, so it's rejected.
- **Result:**
  ```
  Invalid request: Please describe what you want to listen to.
  ```
  Requests over 1000 characters are rejected the same way.

**2. Output guardrail: Gemini recommends a song that wasn't retrieved**

- **Input:** "chill lofi for studying". For this test, the fake Gemini client from my test suite answers with two picks: song ID 99 (which doesn't exist) and Library Rain (ID 4, which was retrieved).
- **Behavior:** the guardrail compares every pick with the retrieved set. ID 99 isn't in it, so it's dropped and logged.
- **Result:**
  ```
  Final picks: [(4, 'Library Rain')]
  note: Guardrail: dropped song id 99 (not in retrieved set)
  ```
  If *every* pick were invalid, the app would show the retriever's top 5 instead.

**3. Parser cleanup: values that aren't in the catalog**

- **Input:** "upbeat k-pop". Here the fake parser returns `genre="k-pop"`, `mood="Happy"`, `energy=1.4`, and `mood_tags=["euphoric", "bubbly"]`.
- **Behavior:** the unknown genre is dropped, the mood's capitalization is fixed, energy is clamped to the 0-1 range, and the unknown tag is removed. Retrieval only sees values my catalog really has.
- **Result:**
  ```
  prefs: {'energy': 1.0, 'mood': 'happy', 'mood_tags': ['euphoric'], ...}
  note: Ignored unknown genre 'k-pop' from parser
  ```

**4. Fallback: no API key**

- **Input:** `python -m src.main "chill lofi for studying late at night, no vocals"` with no `GEMINI_API_KEY` set.
- **Behavior:** the app notices there's no key before making any AI call, so it uses the keyword parser and the rule-based scorer instead. An invalid key works much the same way: Gemini rejects the first call, and the app skips the second one.
- **Result:** you still get 5 recommendations (see Example 1 above), plus:
  ```
  Answer source: rule-based fallback
  note: No GEMINI_API_KEY set; using rule-based fallback
  ```

## Evaluation Harness

`python -m src.evaluate` runs the whole system on 9 predefined requests and prints a pass/fail report. Each normal request is checked against system-wide rules:

- every pick was retrieved
- no duplicates
- 1-5 songs returned
- every pick has a reason

Each request also has its own checks, like "the top pick for 'high energy rock for the gym' is rock with energy ≥ 0.7". Two cases (a blank request and a too-long one) must be rejected.

Result with Gemini:

```
$ python -m src.evaluate
Case                   Result Source    Details
------------------------------------------------------------------------------
study lofi             PASS   gemini    chill lofi for studying late at night, n
gym rock               PASS   gemini    high energy rock for the gym
night drive            PASS   gemini    80s synthwave for a night drive
happy pop              PASS   gemini    happy upbeat pop songs
relaxing jazz          PASS   gemini    some relaxing jazz for dinner
sad acoustic           PASS   gemini    sad acoustic songs
genre not in catalog   PASS   gemini    upbeat k-pop for a dance party
empty request          PASS   -         (blank)
too-long request       PASS   -         music music music music music music musi
------------------------------------------------------------------------------
Cases passed:  9/9
Checks passed: 40/40 (100%)
Answered by Gemini: 7/7
```

Offline result (rule-based fallback, no API key):

```
$ python -m src.evaluate --offline
Case                   Result Source    Details
------------------------------------------------------------------------------
study lofi             PASS   fallback  chill lofi for studying late at night, n
gym rock               PASS   fallback  high energy rock for the gym
night drive            PASS   fallback  80s synthwave for a night drive
happy pop              PASS   fallback  happy upbeat pop songs
relaxing jazz          PASS   fallback  some relaxing jazz for dinner
sad acoustic           PASS   fallback  sad acoustic songs
genre not in catalog   PASS   fallback  upbeat k-pop for a dance party
empty request          PASS   -         (blank)
too-long request       PASS   -         music music music music music music musi
------------------------------------------------------------------------------
Cases passed:  9/9
Checks passed: 40/40 (100%)
Answered by Gemini: 0/7  (offline mode)
```

The script exits with code 1 if any case fails, so it could run automatically in a CI pipeline.

## Design Decisions

- **I reused my original scorer as the retriever.** I already had a scorer that ranks songs and explains why, so I didn't add embeddings or a vector database. With only 18 songs, a vector database would add setup and dependencies without any real benefit. The trade-off is that retrieval is only as good as my hand-tuned weights. A bigger catalog would probably need semantic search.
- **I used Gemini at both ends.** Gemini reads the request (parser) and writes the answer (generator), but the retrieval in the middle is deterministic code I can test. That keeps the AI grounded and makes the system easier to debug: I can look at the log and see whether a bad result came from parsing, retrieval, or generation.
- **Structured outputs instead of free text.** Both Gemini calls return JSON that has to match a schema (Pydantic models), so I never have to parse messy text. Gemini refers to songs by ID, which makes the guardrail check simple and reliable.
- **I never fully trust the AI's output.** Even with structured outputs, I clean the parser's results (unknown genres are ignored, energy is clamped to 0-1) and check every generated pick against the retrieved set. It's a little extra code, but it means a made-up song can't reach the user.
- **There's always a fallback.** I wanted anyone (including a grader without an API key) to be able to clone the repo and run it. The trade-off is that the fallback is noticeably worse at understanding vague requests, so the output always says which path produced the answer.
- **Gemini instead of Claude.** I first built the AI calls with Claude, but Claude's API requires paid credits. I switched to Google's Gemini (`gemini-3.1-flash-lite`) because it has a free tier, so anyone (including me) can run the AI part without paying. You can choose a different Gemini model with the `GEMINI_MODEL` environment variable. Because my retriever, guardrail and fallback are separate from the AI calls, only one function (`_call_llm`) had to change. The trade-off is that the free tier has rate limits, but the app falls back gracefully when it hits them.
- **Low temperature.** I set the temperature to 0.2 so the same request gives similar answers each time, which makes the system more consistent and easier to test.

## Testing Summary

I have **21 automated tests**, and all of them pass (`python -m pytest`). The RAG tests use a fake Gemini client, so they don't need an API key or internet access, and they give the same result every time. I also built the evaluation harness above, which runs the real system end to end on 9 requests (9/9 passing offline).

**What the tests cover:**

- The full pipeline uses Gemini's output, and every recommended song comes from the retrieved set.
- The prompt sent to the generator contains only retrieved songs and no others.
- The guardrail drops a made-up song ID, and falls back to retrieval if every pick is invalid.
- Bad parser output gets cleaned up (unknown genre, wrong capitalization, energy out of range, unknown tag, unknown decade).
- The fallback works when there's no API key, an invalid key, a rate limit, a server error, a network error, or a blocked response.
- Empty and too-long requests are rejected.
- The keyword parser, the guardrail's duplicate removal and its 5-song limit, plus the original Module 1-3 tests.

**What worked:** the guardrail and fallback design. When I ran the app for the first time without a key, it still returned sensible recommendations instead of crashing.

**What didn't work at first:**

- When I first ran it with a real Gemini key, the model I had picked (`gemini-2.5-flash`) returned "404: no longer available to new users." The newer model Google suggested (`gemini-3.8-flash`) was overloaded ("503: high demand"), and the app **hung for over three minutes** because the SDK kept retrying with no time limit. The fallback never got a chance to run. I added a 30-second timeout and capped retries at 2, then switched to `gemini-3.1-flash-lite`, which answers in a few seconds. Lesson: a fallback only helps if the app gives up on the AI in time.
- In my first version (built with Claude), running without a key printed two long tracebacks. The SDK raised a `TypeError` for missing credentials, not an authentication error, so my code treated it as an "unexpected error." That taught me to check for a missing key up front. The Gemini version does this before making any call, shows one clear message, and keeps full tracebacks in the log file only.
- The sample runs above also showed that the keyword fallback misreads requests that don't use catalog words ("nostalgic" matched the wrong song, and "rainy day" didn't change the energy).
- While writing the guardrail examples, I found that the fallback read "k-pop" as "pop", because the hyphen counted as a word break. I fixed the pattern and added a test for it. I also found that `python -m src.main ""` opened the input prompt instead of rejecting the empty request, so I fixed that too.

**What I learned:** mocking the AI client was the key to testing an AI feature. I couldn't write tests like "Gemini returns exactly this text," but I *could* test everything my code does with Gemini's output, and that's where the guardrails live.

## Reflection

This project taught me that the AI model is only one piece of an AI system, and probably not the hardest one. Most of my time went into the parts around it: deciding what data the model sees, checking what it gives back, and deciding what happens when it fails. RAG changed how I think about trusting AI. Instead of hoping the model knows the right answer, I control what it's allowed to answer *from*, and then I verify it stuck to that.

I also learned that breaking a problem into steps (parse, then retrieve, then generate, then check) makes it much easier to find where something goes wrong. When a recommendation looked off, the log showed me which step caused it. Having a rule-based version of each step made the AI version easier to judge, because I could compare the two directly.
