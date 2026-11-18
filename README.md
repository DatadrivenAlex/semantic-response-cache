# Semantic Response Cache Lab

A small offline demonstration of caching AI-style responses by similarity. A SQLite store compares normalized word-frequency vectors using cosine similarity, with explicit tenant-like scope, model/version key, TTL and invalidation. No embedding model, AI provider or network is required.

```bash
python -m pip install -e ".[dev]"
python -m semantic_cache.demo
pytest -q
ruff check .
```

## How it works

A query searches only unexpired entries with the exact scope and model key. The highest cosine score above the chosen threshold is returned; ties favor the newest entry. Expired rows can be purged and a scope/model pair invalidated.

## Important limitations

This baseline is lexical, not a semantic embedding model: word order, negation, numeric changes and context can produce unsafe false hits. It is not suitable for sensitive, personalized or high-stakes responses without further controls. Scope is a caller-provided label, not authentication or authorization. Data, including prompts and responses, is stored in plaintext in SQLite; use synthetic data only. There is no provider integration, encryption, distributed concurrency or production tenant isolation.
