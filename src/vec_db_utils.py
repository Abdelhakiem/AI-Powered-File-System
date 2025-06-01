import os
import json
import faiss
import numpy as np
from pathlib import Path

INDEX_FILE = os.path.join(VECTOR_DB_PATH, "file_index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_PATH, "index_metadata.json")

# Initialize global variables
index = None
metadata_list = []

def load_or_create_vector_db():
    global index, metadata_list
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    # Check if database exists
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print("Loading existing vector database...")
        try:
            # Load FAISS index
            index = faiss.read_index(INDEX_FILE)
            
            # Load metadata
            with open(METADATA_FILE, 'r') as f:
                metadata_list = json.load(f)
                
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
        # Save FAISS index
        faiss.write_index(index, INDEX_FILE)
        
        # Save metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata_list, f, indent=2)
            
        print(f"Saved vector DB with {index.ntotal} entries")
    except Exception as e:
        print(f"Error saving vector DB: {e}")

def store_embedding(embedding: np.ndarray, meta_data: dict):
    """
    Store embedding and metadata in vector database
    
    Args:
        embedding: Embedding vector (1D numpy array)
        meta_data: Metadata dictionary
    """
    global index, metadata_list
    
    # Ensure embedding is in correct format
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    
    # Add to index
    index.add(embedding.astype('float32'))
    
    # Add to metadata
    metadata_list.append(meta_data)
    
    # Save to disk
    save_vector_db()
    
    print(f"Stored embedding for: {meta_data['file_name']}")

from typing import List, Dict, Tuple

# Update the store_Vdb function
def store_Vdb(embedding: np.ndarray, meta_data: dict):
    """Store in vector database (actual implementation)"""
    global index, metadata_list
    
    # Load DB if not already loaded
    if index is None:
        index, metadata_list = load_or_create_vector_db()
    
    # Store the embedding
    store_embedding(embedding, meta_data)
    
    return True
    
def retrieve_all_embeddings():
    """
    Retrieve all stored embeddings and metadata
    
    Returns:
        tuple: (embeddings, metadata_list)
    """
    global index, metadata_list
    
    if index.ntotal == 0:
        return np.array([]), []
    
    # Retrieve all embeddings
    all_embeddings = index.reconstruct_n(0, index.ntotal)
    
    return all_embeddings, metadata_list

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
    
    # Prepare query vector
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Get metadata for results
    results = []
    for i in indices[0]:
        if i >= 0:  # FAISS returns -1 for invalid indices
            results.append(metadata_list[i])
    
    return results



# New function for searching files
def search_files(query: str, k: int = 5) -> List[Dict]:
    """
    Search files based on text query
    
    Args:
        query: Text query to search for
        k: Number of results to return
        
    Returns:
        List of metadata dictionaries for matching files
    """
    # Load models and DB
    _, _, _, emb_processor, emb_model = load_models()
    if index is None:
        index, metadata_list = load_or_create_vector_db()
    
    # Convert query to embedding
    query_embedding = text_to_vector(query, emb_model, emb_processor)
    
    # Search vector DB
    results = search_similar(query_embedding, k)
    
    # Format results
    for result in results:
        result['stored_file'] = os.path.join(
            FILE_STORING_PATH, 
            f"{result['file_id']}{Path(result['file_path']).suffix}"
        )
    
    return results

    
