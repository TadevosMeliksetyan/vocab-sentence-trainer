"""Core logic: word storage, embeddings, word selection, Gemini calls.

No input() / print() here, so the same code can be used by a web UI (app.py)
or by a notebook.
"""
import json
import os

import numpy as np

DATA_FILE = "words.json"          # {"word": {"definition": "...", "unknown_count": 0}}
LLM_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RULES = """You help me learn English vocabulary.
You get JSON with "new_word" (one word/phrase with its definition) and
"review_words" (other words with definitions, for review).
Write ONE natural English sentence.

Priority:
1. The sentence is mainly about the new word. Use it in the meaning given by its
   definition, in a concrete everyday situation (a person doing something), so a learner
   can guess its meaning from context. Do not copy the definition.
2. Also use the review words, but only as supporting parts. If a review word would make
   the sentence unnatural or unclear, skip it. Never sacrifice the new word's clarity.

Other rules:
- Keep each used word in its given form (only verb tense may change).
- Use an idiom only where a native speaker would really say it.
- Everyday, correct English, one sentence, not too long.
- Output only the sentence."""


def load_data(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_data(data, path):
    # write to a temp file first, so a crash can never leave a half-written words.json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class Vocab:
    def __init__(self, path=DATA_FILE):
        # heavy imports live here, so `import vocab` stays fast
        from google import genai
        from sentence_transformers import SentenceTransformer

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")
        self.client = genai.Client(api_key=key)
        self.st = SentenceTransformer(EMBED_MODEL)
        self.path = path
        if not os.path.exists(path):
            print(f"WARNING: {os.path.abspath(path)} not found, starting with an empty word list.")
        self.data = load_data(path)
        self.mtime = self._file_mtime()      # remember the file version we loaded
        self.rebuild()

    # ---- the single source of truth is self.data; words and vecs are derived from it
    def rebuild(self):
        self.words = list(self.data)
        texts = [f"{w}: {self.data[w]['definition']}" for w in self.words]
        self.vecs = self.st.encode(texts, normalize_embeddings=True)

    def _file_mtime(self):
        return os.path.getmtime(self.path) if os.path.exists(self.path) else None

    def save(self):
        # if the file changed since we loaded it, our copy in memory is outdated:
        # saving now would overwrite the newer file, so refuse instead
        if self._file_mtime() != self.mtime:
            raise RuntimeError(f"{self.path} was changed outside the app. "
                               "Stop the app (Ctrl+C) and start it again; nothing was overwritten.")
        save_data(self.data, self.path)
        self.mtime = self._file_mtime()

    # ---- Gemini calls
    def _ask(self, prompt, system=None):
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system) if system else None
        r = self.client.models.generate_content(model=LLM_MODEL, contents=prompt, config=config)
        return (r.text or "").strip()

    def define(self, word):
        return self._ask(f'Give a short, simple definition (max 15 words) of "{word}" '
                         "in its most common meaning. Output only the definition.")

    def make_sentence(self, word, review):
        payload = {
            "new_word": {word: self.data[word]["definition"]},
            "review_words": {w: self.data[w]["definition"] for w in review},
        }
        return self._ask(json.dumps(payload, ensure_ascii=False), system=RULES)

    # ---- word storage
    def add_word(self, word, definition):
        """Add a word, or update its definition (keeps unknown_count)."""
        old = self.data.get(word, {})
        self.data[word] = {"definition": definition,
                           "unknown_count": old.get("unknown_count", 0)}
        self.save()
        self.rebuild()

    def mark_unknown(self, words):
        for w in words:
            self.data[w]["unknown_count"] += 1
        self.save()

    # ---- selection
    def pick_review(self, word, k=2, skip=3, pool=30):
        """Pick k review words related to `word`.

        - rank all other words by similarity to `word`
        - skip the `skip` closest ones (they are often near-synonyms -> bad sentences)
        - from the next `pool` words, pick k at random, where a word is more likely to be
          picked the more often I marked it as unknown (weight = 1 + unknown_count)
        """
        i = self.words.index(word)
        scores = self.vecs @ self.vecs[i]
        scores[i] = -np.inf                              # never pick the word itself
        order = np.argsort(-scores)[skip:skip + pool]
        order = order[np.isfinite(scores[order])]
        if len(order) == 0:
            return []
        weights = np.array([1 + self.data[self.words[j]]["unknown_count"] for j in order],
                           dtype=float)
        chosen = np.random.choice(order, size=min(k, len(order)), replace=False,
                                  p=weights / weights.sum())
        return [self.words[j] for j in chosen]
