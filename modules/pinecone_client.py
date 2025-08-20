# modules/pinecone_client.py
from typing import Optional, List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import re, unicodedata, hashlib

# ----- ID sanitizer (keeps the hash; good for vector IDs) -----
def ascii_id(s: str, maxlen: int = 200) -> str:
    raw = s or "doc"
    base = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    base = re.sub(r"[^\w\-.]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_.-") or "doc"
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    out = f"{base[:maxlen]}_{h}"
    return out[:512]

# ----- NAMESPACE sanitizer (NO hash; stays exactly 'pa' etc.) -----
def ascii_ns(s: str) -> str:
    raw = s or "pa"
    base = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    base = re.sub(r"[^\w\-.]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_.-") or "pa"
    return base[:64]

class PineconeClient:
    def __init__(
        self,
        index_name: str,
        creds: Dict[str, Any],
        dimension: int,
        metric: str = "cosine",
        namespace: Optional[str] = "pa",
    ):
        self.index_name = index_name
        self.namespace = ascii_ns(namespace or "pa")   # <-- stable (no hash)
        self.dimension = int(dimension)
        self.metric = metric

        api_key = creds.get("api_key")
        cloud = creds.get("cloud", "aws")
        region = creds.get("region", "us-east-1")
        host = creds.get("host")

        if not api_key:
            raise RuntimeError("Pinecone API key missing. Did you load your .env?")

        pc = Pinecone(api_key=api_key)

        # Ensure index exists
        existing = {i["name"] for i in pc.list_indexes()}
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = pc.Index(index_name, host=host) if host else pc.Index(index_name)

        # Optional sanity: skip brittle describe() if your client acts up
        try:
            desc = getattr(self.index, "describe_index_stats", None)
            if callable(desc):
                d = desc()
                idx_dim = d.get("dimension") if isinstance(d, dict) else getattr(d, "dimension", None)
                if idx_dim and int(idx_dim) != self.dimension:
                    raise RuntimeError(
                        f"Configured dim {self.dimension} != index dim {idx_dim} for '{self.index_name}'."
                    )
        except Exception as e:
            print(f"[pinecone] describe_index_stats() unavailable/failed: {e} (continuing).")

    def upsert(self, chunks: List[str], vectors: List[List[float]], metadata: Dict[str, Any]):
        """
        Upsert chunk embeddings.
        - Uses the client's namespace ONLY (ignores metadata['namespace'] to avoid drift)
        - Stores chunk text in metadata['text'] (truncated)
        """
        ns = self.namespace                         # <-- force stable namespace
        base = dict(metadata)
        base.pop("namespace", None)                 # <-- ignore caller-provided namespace

        # Stable, ASCII-safe IDs per document + chunk
        src = base.get("title", base.get("filename", "doc"))
        rid_base = ascii_id(src)

        payload = []
        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            meta = base.copy()
            meta["text"] = (text or "")[:8000]
            rid = f"{rid_base}_{i:04d}"
            payload.append({"id": rid, "values": vec, "metadata": meta})

        self.index.upsert(vectors=payload, namespace=ns)
