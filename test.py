
import os
from dotenv import load_dotenv
from pinecone import Pinecone
load_dotenv(".env.sciai")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
print("Indexes:", [i["name"] for i in pc.list_indexes()])

