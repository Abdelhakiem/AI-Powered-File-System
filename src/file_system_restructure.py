import faiss
import os
import json
from sklearn.preprocessing import normalize
import hdbscan
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

IMG_TXT_PATH              = str(SCRIPT_DIR / "../models/img_to_text/")
SUMMARIZER_PATH           = str(SCRIPT_DIR / "../models/summarizer/")
LOCAL_CAPTION_DIR         = str(SCRIPT_DIR / "../models/img_caption")
LOCAL_RETRIEVAL_DIR       = str(SCRIPT_DIR / "../models/embedding")
FILE_STORING_PATH         = str(SCRIPT_DIR / "../files_DB")
FILE_SYSTEM_PATH          = str(SCRIPT_DIR / "../File_System_Simulation")
VECTOR_DB_PATH            = str(SCRIPT_DIR / "../vectorDB")

INDEX_FILE    = os.path.join(VECTOR_DB_PATH, "file_index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_PATH, "index_metadata.json")

KEYWORD_EXTRACTION_MODEL_PATH = str(SCRIPT_DIR / "../models/kword_extraction")
NEW_ROOT                     = str(SCRIPT_DIR / "../File_System_Restructured")

index = None
metadata_list = []



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
            else:
                print(f"Loaded vector DB with {index.ntotal} entries")
        except Exception as e:
            print(f"Error loading vector DB: {e}")
    else:
        print("Creating new vector database...")
    return index, metadata_list

def retrieve_all_embeddings():
    global index
    if index.ntotal == 0:
        return np.array([]), []
    # Retrieve all embeddings
    all_embeddings = index.reconstruct_n(0, index.ntotal)
    
    return all_embeddings
def save_vector_db(index, metadata_list):
    # 1) Write FAISS index
    faiss.write_index(index, INDEX_FILE)

    # 2) Dump metadata_list to JSON
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved index to {INDEX_FILE} and metadata ({len(metadata_list)} entries) to {METADATA_FILE}.")

import umap.umap_ as umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

def restructure_file_system(metadata_list, file_system_path):
    for meta in metadata_list:
        src_fullpath = meta["storage_path"]
        new_relpath = meta["file_path"]            # e.g. "swimming/swimming.txt"
        dest_fullpath = os.path.join(file_system_path, new_relpath)

        # 1) Ensure destination directory exists
        dest_folder = os.path.dirname(dest_fullpath)
        os.makedirs(dest_folder, exist_ok=True)

        # 2) Move the file
        if os.path.exists(src_fullpath):
            shutil.copy2(src_fullpath, dest_fullpath)
        else:
            print(f"[WARNING] Source not found: {src_fullpath}")


def cluster_files():
    load_or_create_vector_db()
    all_embeddings = retrieve_all_embeddings()
    um = umap.UMAP(n_components=4,min_dist=0, metric='cosine',  random_state=42)
    reduced = um.fit_transform(all_embeddings)  # shape (6, 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,     # minimum cluster size
        min_samples=1,                  # aggressive clusterin                
        cluster_selection_method='leaf' # more fine-grained clusters
    )
    labels = clusterer.fit_predict(reduced)
    # get file_names per cluster: -> groupd {cluster_number: list_file_names}
    grouped={}
    for i,l in enumerate(labels):
        if l in grouped:
            grouped[l]=grouped[l]+','+metadata_list[i]['file_name']
        else:
            grouped[l]=metadata_list[i]['file_name']
    docs=np.array(list(grouped.values()))
    local_embedding_model = SentenceTransformer(KEYWORD_EXTRACTION_MODEL_PATH)
    # Initialize KeyBERT with the locally loaded embedding model
    kw_model = KeyBERT(model=local_embedding_model)
    keyphrases = kw_model.extract_keywords(docs, top_n=1,)
    title_dict={}
    for ky, kphrase in zip(grouped.keys(),keyphrases):
        title_dict[ky]=kphrase[0][0]
    for label,mt_data in zip(labels,metadata_list):
        mt_data['file_path']=f"{title_dict[label]}/{mt_data['file_path'].split('/')[-1]}"
        # Call the mover using the updated_meta from before:
    restructure_file_system(metadata_list, NEW_ROOT)
    save_vector_db(index, metadata_list)

if __name__ == "__main__":
    cluster_files()
