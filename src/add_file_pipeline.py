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
import os
import shutil


IMG_TXT_PATH = "../models/img_to_text/"
SUMMARIZER_PATH = "../models/summarizer/"
# Paths where you previously saved the BLIP models & processors:
LOCAL_CAPTION_DIR = "../models/img_caption"
LOCAL_RETRIEVAL_DIR    = "../models/embedding"

FILE_STORING_PATH ='../files_DB'
FILE_SYSTEM_PATH  = '../File_System_Simulation'

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
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


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


def store_Vdb(V_db,emd,meta_data):
    pass

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
    loading_file_path = os.path.join(FILE_SYSTEM_PATH, file_path)
    file = None
    # 0. Load models: 
    summarizer, caption_processor, caption_model, emb_processor, emb_model = load_models()
    
    # 1. check file type: image, text, unknown
    file_type = check_file_type(file_path)
    file_name = Path(file_path).name
    file_id = generate_file_id(file_path)

    # Process based on file type
    if file_type == 'image':
        try:
            with Image.open(file_path) as img:
                global file
                file = img
                img = img.convert('RGB')
                content = image_to_caption(img, caption_processor, caption_model)
                emb_vec = image_to_vector(img, emb_model, emb_processor)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None
            
    elif file_type == 'text':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                global file
                file = f

                full_text = f.read()
                content = summarize_text_file(summarizer, full_text)
                emb_vec = text_to_vector(full_text, emb_model, emb_processor)
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return None
            
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return None
    

    # TODO: storing file in DB + file_id
    storage_path = store_file(file_id, original_file_path =loading_file_path , file_type)

    # TODO: store: text_embedding,metadata
    metadata = {
        'file_id': file_id,
        'file_name': file_name,
        'file_path': str(Path(file_path).resolve()),
        'file_type': file_type,
        'content': content,
        'storage_path' : storage_path
    }
    
    


# ==== Example usage ====
if __name__ == "__main__":
    f_path="/teamspace/studios/this_studio/AI-Powered-File-System/tests/test_data.py"
    print(f"Type :{check_file_type(f_path)}")
    print(f"Gen_ID : {generate_file_id(f_path)}")
