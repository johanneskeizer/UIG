import argparse
from ingest import Ingestor

def main():
    parser = argparse.ArgumentParser(description="Universal Pinecone Ingestor")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    Ingestor.from_config(args.config).run()

if __name__ == "__main__":
    main()
