from PIL import Image
from transformers  import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import mimetypes
import hashlib
from pathlib import Path
import faiss


IMG_TXT_PATH = "../models/img_to_text/"

def load_models():
    img_txt_model = BlipForConditionalGeneration.from_pretrained(IMG_TXT_PATH)
    img_txt_processor = BlipProcessor.from_pretrained(IMG_TXT_PATH)
    return img_txt_model, img_txt_processor


def img_to_text(img, model, processor )->str:
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def is_image(file_path):
    pass

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

    

def add_file_pipeline(file_path):
    file_path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = file_path.suffix.lower()

    img_txt_model, img_txt_processor = load_models()
    # Prepare metadata
    file_id = file_path.stem  # or generate a UUID
    meta = {
        "file_id": file_id,
        "file_name": file_path.name,
        "file_path": str(file_path),
    }
    # IMAGE
    if mime_type and mime_type.startswith("image"):
        img = Image.open(file_path).convert("RGB")
        text = img_to_text(img , model = img_txt_model, processor= img_txt_processor)

    # TEXT
    elif extension in {".txt", ".md", ".csv"}:
        raw = file_path.read_text(encoding="utf-8")
        #text = text_to_summary(raw, summarizer)

    else:
        raise ValueError(f"Unsupported file type: {extension}")
    

    # TODO: text to vector embedding

    # TODO: storing file in DB + file_id

    # TODO: store: text_embedding,file_name, file_path, file_id
    store_Vdb(V_db,emd,meta_data)
    pass


# ==== Example usage ====
if __name__ == "__main__":
    f_path="/teamspace/studios/this_studio/AI-Powered-File-System/tests/test_data.py"
    print(f"Type :{check_file_type(f_path)}")
    print(f"Gen_ID : {generate_file_id(f_path)}")