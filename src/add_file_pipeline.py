from PIL import Image
from transformers  import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import mimetypes


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
        # text = text_to_summary(raw, summarizer)

    else:
        raise ValueError(f"Unsupported file type: {extension}")
    

    # TODO: text to vector embedding

    # TODO: storing file in DB + file_id

    # TODO: store: text_embedding,file_name, file_path, file_id
    pass


# ==== Example usage ====
if __name__ == "__main__":
    pass
