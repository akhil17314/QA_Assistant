"""Provide required settings so tests can import app modules without a local .env file."""

import os

os.environ.setdefault("LLM_MODEL", "test-llm-model")
os.environ.setdefault("EMBEDDING_MODEL", "test-embedding-model")
