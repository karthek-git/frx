"""Text embedders for RFQ clustering / novelty detection.

  GeminiEmbedder : Google embeddings API (needs GEMINI_API_KEY or GOOGLE_API_KEY, in env or a .env file)
  TfidfEmbedder  : local scikit-learn fallback, no network

Both expose fit(texts) / transform(texts) -> dense, L2-normalised numpy array, plus a `.name` for the UI.
Gemini vectors are cached on disk (keyed by model + text hash) so a demo re-run needs no network.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
CACHE_FILE = ROOT / ".embed_cache.json"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"


class EmbedError(RuntimeError):
    pass


def load_env():
    """Tiny .env loader (rulemine/.env, then repo-root .env). Never overrides real environment variables."""
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except OSError:
            pass


def _normalise(m):
    m = np.asarray(m, dtype=float)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class TfidfEmbedder:
    name = "TF-IDF (local scikit-learn)"

    def __init__(self):
        # letters only: numbers and dimensions vary per email and carry no meaning for grouping
        self.vec = TfidfVectorizer(token_pattern=r"[A-Za-z][A-Za-z]+", stop_words="english", sublinear_tf=True)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return _normalise(self.vec.transform(texts).toarray())


class GeminiEmbedder:
    def __init__(self, api_key, model=None, dims=768, batch=50):
        self.keys = [api_key] if isinstance(api_key, str) else list(api_key)   # tried in order if one is rejected
        self.ki = 0
        self.model = model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        self.dims, self.batch = dims, batch
        self.name = f"{self.model} (Google)"
        self.cache = self._load_cache()

    @staticmethod
    def _load_cache():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}

    def _save_cache(self):
        try:
            CACHE_FILE.write_text(json.dumps(self.cache))
        except OSError:
            pass

    def _k(self, text):
        return hashlib.sha256(f"{self.model}|{self.dims}|{text}".encode()).hexdigest()

    def fit(self, texts):
        return self

    def _request(self, texts):
        v2 = "embedding-2" in self.model
        reqs = []
        for t in texts:
            r = {"model": f"models/{self.model}", "output_dimensionality": self.dims}
            if v2:   # gemini-embedding-2 has no taskType; the task is a text prefix
                r["content"] = {"parts": [{"text": f"task: clustering | query: {t}"}]}
            else:
                r["content"] = {"parts": [{"text": t}]}
                r["taskType"] = "CLUSTERING"
            reqs.append(r)
        body = json.dumps({"requests": reqs}).encode()
        attempt = 0
        while attempt < 4:
            req = urllib.request.Request(
                API.format(model=self.model), data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.keys[self.ki]})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read())
                return [e["values"] for e in data["embeddings"]]
            except urllib.error.HTTPError as ex:
                detail = ""
                try:   # Google's error body explains the cause (it does not echo the key)
                    detail = json.loads(ex.read()).get("error", {}).get("message", "")[:240]
                except Exception:
                    pass
                if ex.code == 400 and "API key" in detail and self.ki + 1 < len(self.keys):
                    self.ki += 1          # this key was rejected; try the next configured one
                    continue
                if ex.code in (429, 500, 503) and attempt < 3:
                    # 429 on the free tier is usually the per-minute limit (~100 texts/min): wait for the window to reset
                    wait = 65 if ex.code == 429 else 2 ** attempt * 2
                    print(f"[embed] HTTP {ex.code}, waiting {wait}s (retry {attempt + 1}/3)", flush=True)
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise EmbedError(f"Google embeddings API returned HTTP {ex.code}: {detail}".rstrip(": ")) from None
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as ex:
                raise EmbedError(f"Google embeddings request failed: {type(ex).__name__}") from None
        raise EmbedError("Google embeddings API: retries exhausted")

    def transform(self, texts):
        keys = [self._k(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        for s in range(0, len(missing), self.batch):
            idx = missing[s:s + self.batch]
            vecs = self._request([texts[i] for i in idx])
            if len(vecs) != len(idx):
                raise EmbedError("Google embeddings API returned an unexpected number of vectors")
            for i, v in zip(idx, vecs):
                self.cache[keys[i]] = v
            self._save_cache()
            print(f"[embed] {len(texts) - len(missing) + s + len(idx)}/{len(texts)} vectors ready", flush=True)
        return _normalise([self.cache[k] for k in keys])


def get_embedder():
    """Gemini if a key is configured, else the local fallback."""
    load_env()
    keys = []
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        k = os.environ.get(var, "").strip().strip("'\"").strip()   # tolerate stray whitespace / quotes
        if k and k not in keys:
            keys.append(k)
    return GeminiEmbedder(keys) if keys else TfidfEmbedder()


if __name__ == "__main__":
    # Warm the cache once:  ../.pyi/bin/python embed.py   (resumable; the server then starts instantly)
    import gen
    emb = get_embedder()
    print("backend:", emb.name, flush=True)
    emb.transform([e["email"] for e in gen.make_events()])
    print("done", flush=True)
