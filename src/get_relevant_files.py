import os
import torch
import json
import faiss
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
LOCAL_RETRIEVAL_DIR = str(SCRIPT_DIR / "../models/embedding")
VECTOR_DB_PATH = str(SCRIPT_DIR / "../vectorDB")
INDEX_FILE = os.path.join(VECTOR_DB_PATH, "file_index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_PATH, "index_metadata.json")

index = None
metadata_list = []
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    emb_processor = CLIPProcessor.from_pretrained(LOCAL_RETRIEVAL_DIR)
    emb_model = CLIPModel.from_pretrained(LOCAL_RETRIEVAL_DIR).to(device)
    return emb_processor, emb_model

def text_to_vector(text: str, emb_model, emb_processor) -> np.ndarray:
    inputs = emb_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = emb_model.get_text_features(**inputs)
    # Normalize to unit vector (L2 norm)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().squeeze()

def load_or_create_vector_db():
    global index, metadata_list
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print("Loading existing vector database...")
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r') as f:
                metadata_list = json.load(f)
                
            # Add consistency check
            if index.ntotal != len(metadata_list):
                print(f"Warning: Index count ({index.ntotal}) doesn't match metadata count ({len(metadata_list)}). Recreating index.")
                create_new_vector_db()
            else:
                print(f"Loaded vector DB with {index.ntotal} entries")
                
        except Exception as e:
            print(f"Error loading vector DB: {e}")
            create_new_vector_db()
    else:
        print("Creating new vector database...")
        create_new_vector_db()
        
    return index, metadata_list

def create_new_vector_db():
    """Create a new empty vector database"""
    global index, metadata_list
    dimension = 512  # CLIP embedding dimension
    index = faiss.IndexFlatL2(dimension)
    metadata_list = []
    save_vector_db()
    print(f"Created new vector DB with dimension {dimension}")

def save_vector_db():
    """Save the vector database to disk"""
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"Saved vector DB with {index.ntotal} entries")
    except Exception as e:
        print(f"Error saving vector DB: {e}")

def search_similar(query_embedding: np.ndarray, k: int = 5):
    """
    Search for similar vectors in the database
    
    Args:
        query_embedding: Embedding vector to compare against
        k: Number of results to return
        
    Returns:
        list: Metadata of top k matches
    """
    global index, metadata_list
    
    if index.ntotal == 0:
        return []
    
    # Adjust k to not exceed available entries
    actual_k = min(k, index.ntotal)
    
    # Prepare query vector
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query_embedding.astype('float32'), actual_k)
    
    # Get metadata for results
    results = []
    for i in indices[0]:
        if i >= 0 and i < len(metadata_list):
            results.append(metadata_list[i]['file_path'])
    
    return results

def semantic_search_engine(search_query: str):
    load_or_create_vector_db()
    emb_processor, emb_model = load_models()
    search_emb = text_to_vector(search_query, emb_model, emb_processor)
    relevant_files_paths = search_similar(search_emb)
    return relevant_files_paths

# ==== Example usage ====
if __name__ == "__main__":
    search_query = "sea with blue sky"
    results = semantic_search_engine(search_query)
    print("Search results:", results)