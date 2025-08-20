import argparse
import yaml
import json
import sys

from modules.utils import load_environment

from modules.loader import load_files
from modules.chunker import chunk_text
from modules.embedder_pubmedbert import PubMedBERTEmbedder
from modules.embedder_openai import OpenAIEmbedder
from modules.pinecone_client import PineconeClient
import sys
print(f"[ingest] file={__file__} python={sys.executable}")


def get_embedder(name: str):
    n = (name or "").strip().lower()
    if n in ("openai-3-large", "text-embedding-3-large", "openai-large"):
        return OpenAIEmbedder(model="text-embedding-3-large", dim=3072)
    if n in ("openai-3-small", "text-embedding-3-small", "openai-small"):
        return OpenAIEmbedder(model="text-embedding-3-small", dim=1536)
    if n in ("pubmedbert", "pubmed-bert", "microsoft/biomednlp-pubmedbert-base-uncased-abstract"):
        return PubMedBERTEmbedder()
    raise ValueError(f"Unknown embedding_model: {name}")

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="UIG ingestion pipeline")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--config-json", help="Inline JSON config (overrides --config)")
    parser.add_argument("--profile", choices=["sciai", "pa", "dionbot"],
                        help="Shortcut to configs/config/{profile}.yml")
    parser.add_argument("--dry-run", action="store_true", help="Run without Pinecone upsert")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # --- Load config ---
    if args.config_json:
        config = json.loads(args.config_json)
        print("[ingest] Using inline JSON config")
    else:
        if args.profile and not args.config:
            args.config = f"config/{args.profile}.yml"
        if not args.config:
            parser.error("Provide --config, --config-json, or --profile")
        print(f"[ingest] Using config: {args.config}")
        with open(args.config) as f:
            config = yaml.safe_load(f)
            load_environment(config["env_file"])
            embedder = get_embedder(config["embedding_model"])
            print(f"[ingest] Embedder: {config['embedding_model']}")


    # now continue with environment load, embedder selection, etc...


    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/sciai.yml",
                   help="Path to YAML config (default: config/sciai.yml)")
    p.add_argument("--dry-run", action="store_true", help="Run everything except Pinecone upsert")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    print(f"[ingest] Using config: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    creds = load_environment(config["env_file"])
    if args.verbose:
        print(f"[ingest] Loaded env. Index: {config['pinecone_index']} · Dim: {config.get('dimension')}")

    embedder = get_embedder(config["embedding_model"])
    print(f"[ingest] Embedder: {config['embedding_model']}")

    pinecone = PineconeClient(
        index_name=config["pinecone_index"],
        creds=creds,
        dimension=int(config["dimension"]),
        metric="cosine",
        namespace=config.get("namespace"),
    )

    stream = config["input_streams"][0]
    path = stream["path"]
    exts = stream["content_types"]
    print(f"[ingest] Loading from {path} types={exts}")

    docs = load_files(path, exts)
    if not docs:
        print(f"[ingest] No documents found in: {path}")
        sys.exit(0)

    # Process only first doc for this smoke test
    doc = docs[0]
    print(f"[ingest] Processing: {doc['metadata']['filename']}")
    chunks = chunk_text(doc["text"], config["chunk_size"], config["chunk_overlap"])
    print(f"[ingest] Chunks: {len(chunks)}")

    vectors = embedder.embed_chunks(chunks)
    vec_dim = len(vectors[0]) if vectors else 0
    cfg_dim = int(config["dimension"])
    print(f"[ingest] Vector dim={vec_dim} · Config dim={cfg_dim}")
    if vec_dim != cfg_dim:
     raise RuntimeError(
        f"Embedding dimension {vec_dim} != configured {cfg_dim}. "
        f"Model={config['embedding_model']}"
    )

    if args.dry_run:
        print("[ingest] DRY-RUN: skipping upsert")
        sys.exit(0)

    pinecone.upsert(chunks, vectors, {**config["metadata"], **doc["metadata"]})
    print(f"[ingest] Uploaded {len(chunks)} chunks → index={config['pinecone_index']}")

if __name__ == "__main__":
    main()
