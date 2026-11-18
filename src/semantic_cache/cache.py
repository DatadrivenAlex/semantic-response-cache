"""SQLite-backed lexical similarity response cache; no model or network is involved."""
from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter


def tokens(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def cosine(a: Counter, b: Counter) -> float:
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return sum(v * b.get(k, 0) for k, v in a.items()) / (norm_a * norm_b)


class ResponseCache:
    def __init__(self, path: str = ":memory:", clock=time.time):
        self.db = sqlite3.connect(path)
        self.clock = clock
        self.db.execute("""CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY, scope TEXT NOT NULL, model TEXT NOT NULL,
            prompt TEXT NOT NULL, answer TEXT NOT NULL, expires REAL NOT NULL)
        """)

    def put(self, *, scope: str, model: str, prompt: str, answer: str, ttl: float):
        if not scope or not model or not tokens(prompt) or ttl <= 0 or not math.isfinite(ttl):
            raise ValueError("scope, model, nonempty prompt and finite positive ttl required")
        expires = self.clock() + ttl
        with self.db:
            self.db.execute(
                "INSERT INTO entries(scope, model, prompt, answer, expires) VALUES(?,?,?,?,?)",
                (scope, model, prompt, answer, expires),
            )

    def lookup(self, *, scope: str, model: str, prompt: str, threshold: float = 0.95):
        if not scope or not model or not 0 < threshold <= 1:
            raise ValueError("scope, model and threshold in (0,1] required")
        query = tokens(prompt)
        if not query:
            return None
        rows = self.db.execute(
            "SELECT id, prompt, answer FROM entries WHERE scope=? AND model=? AND expires>?",
            (scope, model, self.clock()),
        )
        # Ties go to newest entry, making the result deterministic.
        candidates = [(cosine(query, tokens(p)), i, answer) for i, p, answer in rows]
        if not candidates:
            return None
        score, _, answer = max(candidates)
        return {"answer": answer, "score": score} if score >= threshold else None

    def invalidate(self, *, scope: str, model: str) -> int:
        with self.db:
            result = self.db.execute("DELETE FROM entries WHERE scope=? AND model=?", (scope, model))
        return result.rowcount

    def purge_expired(self) -> int:
        with self.db:
            result = self.db.execute("DELETE FROM entries WHERE expires<=?", (self.clock(),))
        return result.rowcount

    def close(self):
        self.db.close()
