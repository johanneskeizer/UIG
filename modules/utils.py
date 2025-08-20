# modules/utils.py
import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment(env_file: str):
    """
    Load a .env file robustly:
    - Accepts mixed-case filenames (.env.sciai vs .env.SCIAI)
    - Resolves relative to project root and this file's directory
    - Supports Pinecone v2 (ENVIRONMENT) and v3 (HOST)
    """
    # Candidates to try
    here = Path(__file__).resolve().parent          # .../modules
    root = here.parent                              # project root
    names = {env_file, env_file.lower(), env_file.upper()}
    candidates = []
    for name in names:
        candidates += [
            Path(name),                # as given, relative to CWD
            root / name,               # at project root
            here / name,               # next to modules/
        ]

    loaded = False
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            loaded = True
            break

    if not loaded:
        raise FileNotFoundError(f"Could not find env file among: {', '.join(str(c) for c in candidates)}")

    # Accept common variants for env var names
    api_key = (
        os.getenv("PINECONE_API_KEY")
        or os.getenv("PINECONE_APIKEY")
        or os.getenv("PINECONE_KEY")
    )
    env = (
        os.getenv("PINECONE_ENVIRONMENT")
        or os.getenv("PINECONE_ENV")
    )
    host = os.getenv("PINECONE_HOST")  # v3 style

    if not api_key:
        raise ValueError(f"Missing PINECONE_API_KEY in {env_file}")

    # Return a small creds dict so callers can support v2/v3
    return {"api_key": api_key, "environment": env, "host": host}

