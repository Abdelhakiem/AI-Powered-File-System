from PIL import Image
from transformers  import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import mimetypes
import hashlib
from pathlib import Path
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
import numpy as np
import torch
import shutil
import os
import json
import faiss
import numpy as np
from pathlib import Path


import os
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()

# Convert all paths to absolute paths relative to script location
IMG_TXT_PATH = str(SCRIPT_DIR / "../models/img_to_text/")
SUMMARIZER_PATH = str(SCRIPT_DIR / "../models/summarizer/")
LOCAL_CAPTION_DIR = str(SCRIPT_DIR / "../models/img_caption")
LOCAL_RETRIEVAL_DIR = str(SCRIPT_DIR / "../models/embedding")
FILE_STORING_PATH = str(SCRIPT_DIR / "../files_DB")
FILE_SYSTEM_PATH = str(SCRIPT_DIR / "../File_System_Simulation")
VECTOR_DB_PATH = str(SCRIPT_DIR / "../vectorDB")

INDEX_FILE = os.path.join(VECTOR_DB_PATH, "file_index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_PATH, "index_metadata.json")

index = None
metadata_list = []

# Choose device (GPU if available):
device = "cuda" if torch.cuda.is_available() else "cpu"



def load_models():

    # summarizer:
    summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_PATH)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PATH)
    summarizer = pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer)

    # image captioning
    caption_processor = BlipProcessor.from_pretrained(LOCAL_CAPTION_DIR)
    caption_model     = BlipForConditionalGeneration.from_pretrained(LOCAL_CAPTION_DIR).to(device)

    # embedding model:
    emb_processor = CLIPProcessor.from_pretrained(LOCAL_RETRIEVAL_DIR)
    emb_model     = CLIPModel.from_pretrained(LOCAL_RETRIEVAL_DIR).to(device)
    
    return summarizer, caption_processor, caption_model, emb_processor, emb_model

def text_to_vector(text: str, emb_model, emb_processor) -> np.ndarray:
    inputs = emb_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = emb_model.get_text_features(**inputs)
    # Normalize to unit vector (L2 norm)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().squeeze()

# 2. Image to normalized vector (using CLIP)
def image_to_vector(image: Image.Image, emb_model, emb_processor) -> np.ndarray:
    """
    Convert image to normalized L2 vector using CLIP
    
    Args:
        image: PIL Image object
        
    Returns:
        Normalized embedding vector (numpy array)
    """
    inputs = emb_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = emb_model.get_image_features(**inputs)
    # Normalize to unit vector (L2 norm)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().squeeze()

# 3. Image to caption (using BLIP)
def image_to_caption(image: Image.Image,caption_processor, caption_model , max_length: int = 30) -> str:
    """
    Generate caption from image using BLIP
    
    Args:
        image: PIL Image object
        max_length: Maximum caption length (default 30)
        
    Returns:
        Generated caption string
    """
    inputs = caption_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = caption_model.generate(**inputs, max_length=max_length)
    caption = caption_processor.decode(output[0], skip_special_tokens=True)
    return caption


def summarize_text_file(summarizer, text):
    return summarizer(text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']


def check_file_type(file_path) -> str:
    file_path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("text/"):
            return "text"
    return "unknown"

def generate_file_id(file_path) -> str:
    file_path = Path(file_path)
    path_hash = hashlib.md5(str(file_path.resolve()).encode()).hexdigest()[:8]
    return f"{file_path.stem}_{path_hash}"


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
    


def store_file(file_id: str, original_file_path: str, file_type: str):
    # Create storage directory if it doesn't exist
    os.makedirs(FILE_STORING_PATH, exist_ok=True)
    
    # Get the original file extension
    original_path = Path(original_file_path)
    file_extension = original_path.suffix.lower()
    
    # Handle files without extensions
    if not file_extension:
        if file_type == 'image':
            file_extension = '.jpg'  # Default image format
        else:
            file_extension = '.txt'  # Default text format
    
    # Create storage path with file_id + original extension
    storage_path = os.path.join(FILE_STORING_PATH, f"{file_id}{file_extension}")
    
    # Copy the file to storage
    try:
        if file_type == 'image':
            # For images, we use the processed version to maintain RGB format
            with Image.open(original_file_path) as img:
                img = img.convert('RGB')
                img.save(storage_path)
            print(f"Stored image: {storage_path}")
        else:
            # For text files, copy the original content
            shutil.copy2(original_file_path, storage_path)
            print(f"Stored text: {storage_path}")
    except Exception as e:
        print(f"Error storing file {original_file_path}: {e}")
        return None
    
    return storage_path
        


def add_file_pipeline( file_path):
    load_or_create_vector_db()
    loading_file_path = os.path.join(FILE_SYSTEM_PATH, file_path)
    # 0. Load models: 
    summarizer, caption_processor, caption_model, emb_processor, emb_model = load_models()
    
    # 1. check file type: image, text, unknown
    file_type = check_file_type(file_path)
    file_name = Path(file_path).name
    file_id = generate_file_id(file_path)

    # Process based on file type
    if file_type == 'image':
        try:
            with Image.open(loading_file_path) as img:
                img = img.convert('RGB')
                content = image_to_caption(img, caption_processor, caption_model)
                emb_vec = text_to_vector(content, emb_model, emb_processor)
        except Exception as e:
            print(f"Error processing image {loading_file_path}: {e}")
            return None
            
    elif file_type == 'text':
        try:
            with open(loading_file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
                content = summarize_text_file(summarizer, full_text)
                emb_vec = text_to_vector(content, emb_model, emb_processor)
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return None
            
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return None
    

    storage_path = store_file(file_id, original_file_path =loading_file_path , file_type =file_type)

    metadata = {
        'file_id': file_id,
        'file_name': file_name,
        'file_path': str(Path(file_path).resolve()),
        'file_type': file_type,
        'content': content,
        'storage_path' : storage_path
    }
    store_Vdb(emb_vec, metadata)
    return metadata



def list_files_recursive(root_path):
    """
    Return a list of all files under `root_path`, with each path given
    relative to `root_path` itself.

    Parameters
    ----------
    root_path : str
        The directory from which to start the recursive search.

    Returns
    -------
    List[str]
        A list of file paths (as strings), each relative to `root_path`.
        e.g. if root_path = "/home/user/project", and there is a file
        "/home/user/project/src/main.py", this list will contain "src/main.py".
    """
    result = []
    # os.walk traverses dirpath, dirnames, filenames in a top-down manner
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            # Compute the path relative to the root_path
            rel_path = os.path.relpath(full_path, start=root_path)
            # Normalize to use forward slashes even on Windows (optional)
            result.append(rel_path)
    return result

def process_all_files(root_path):
    rel_file_paths = list_files_recursive(root_path)
    for rel_file_path in rel_file_paths:
        print('Processing file: ', rel_file_path)
        add_file_pipeline( rel_file_path)



# ==== Example usage ====
if __name__ == "__main__":
    process_all_files('../File_System_Simulation')
