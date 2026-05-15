import chromadb

# ==========================================
# CONNECT
# ==========================================

client = chromadb.PersistentClient(
    path="../chroma_db"
)

# ==========================================
# LIST COLLECTIONS
# ==========================================

collections = client.list_collections()

print("\n================================")
print("CHROMADB COLLECTION STATS")
print("================================")

for collection in collections:

    name = collection.name

    col = client.get_collection(name)

    count = col.count()

    print(f"\nCollection: {name}")

    print(f"Chunks Stored: {count}")