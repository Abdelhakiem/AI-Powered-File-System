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


IMG_TXT_PATH = "../models/img_to_text/"
SUMMARIZER_PATH = "../models/summarizer/"
# Paths where you previously saved the BLIP models & processors:
LOCAL_CAPTION_DIR = "../models/img_caption"
LOCAL_RETRIEVAL_DIR    = "../models/embedding"

FILE_STORING_PATH ='../files_DB'
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

def store_file (file_id, file, file_type):
    pass

def add_file_pipeline(file, file_path):
    # 0. Load models: 
    summarizer, caption_processor, caption_model, emb_processor, emb_model = load_models()
    
    # 1. check file type: image, text, unknown
    file_type = check_file_type(file_path)
    file_name = Path(file_path).name
    file_id = generate_file_id(file_path)

    # Process based on file type
    if file_type == 'image':
        content = image_to_caption(file, caption_processor, caption_model)
        emb_vec = image_to_vector(file, emb_model, emb_processor)
            
    elif file_type == 'text':
        content = summarize_text_file(summarizer, file)
        emb_vec = text_to_vector(file, emb_model, emb_processor)
            
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return None
    
    # Prepare metadata
    metadata = {
        'file_id': file_id,
        'file_name': file_name,
        'file_path': str(Path(file_path).resolve()),
        'file_type': file_type,
        'content': content
    }

    # TODO: storing file in DB + file_id
    store_file(file_id, file, file_type)

    # TODO: store: text_embedding,file_name, file_path, file_id
    
    pass


# ==== Example usage ====
if __name__ == "__main__":
    f_path="/teamspace/studios/this_studio/AI-Powered-File-System/tests/test_data.py"
    print(f"Type :{check_file_type(f_path)}")
    print(f"Gen_ID : {generate_file_id(f_path)}")
