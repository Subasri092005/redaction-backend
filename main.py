import os
import logging
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uuid
import json
import datetime
import time
from PIL import Image, ImageDraw
import pdfplumber
import spacy
import traceback
import re
import shutil
import requests
from PyPDF2 import PdfWriter, PdfReader
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://redaction-frontend-pink.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

nlp = spacy.load("en_core_web_sm")

# Redaction entity sets
FULL_ENTITIES = ["PERSON", "GPE", "ORG", "DATE", "CARDINAL", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "LANGUAGE"]
PARTIAL_ENTITIES = ["PERSON", "CARDINAL", "LOC", "NORP"]

def get_entity_types(redaction_level, consent_level, custom_types=None):
    if consent_level == "none":
        return FULL_ENTITIES
    elif consent_level == "limited":
        return PARTIAL_ENTITIES
    elif consent_level == "full":
        if redaction_level == "custom" and custom_types:
            return custom_types
        else:
            return []  # No redaction
    else:
        # fallback to redaction level
        if redaction_level == "full":
            return FULL_ENTITIES
        elif redaction_level == "partial":
            return PARTIAL_ENTITIES
        elif redaction_level == "custom" and custom_types:
            return custom_types
        else:
            return FULL_ENTITIES

def extract_text_blocks(image_path):
    api_key = os.getenv("OCR_SPACE_API_KEY")
    if not api_key:
        raise ValueError("OCR_SPACE_API_KEY not set")

    with open(image_path, 'rb') as f:
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': f},
            data={'apikey': api_key, 'language': 'eng', 'isOverlayRequired': 'true'}
        )
    result = response.json()
    lines = []
    if 'ParsedResults' in result:
        for res in result['ParsedResults']:
            if 'TextOverlay' in res and 'Lines' in res['TextOverlay']:
                for line in res['TextOverlay']['Lines']:
                    line_text = ' '.join(w['WordText'] for w in line['Words'])
                    words = [
                        {
                            "text": w['WordText'],
                            "left": w['Left'],
                            "top": w['Top'],
                            "width": w['Width'],
                            "height": w['Height']
                        } for w in line['Words']
                    ]
                    lines.append({"text": line_text, "words": words})
    return lines

def redact_text(text, redaction_level="full", consent_level="none", custom_types=None):
    doc = nlp(text)
    redacted = text
    entity_types = get_entity_types(redaction_level, consent_level, custom_types)
    for ent in doc.ents:
        if ent.label_ in entity_types:
            redacted = redacted.replace(ent.text, "[REDACTED]")
    # Always redact phone, email, address, and ID numbers
    redacted = re.sub(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}", "[REDACTED]", redacted)
    redacted = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED]", redacted)
    redacted = re.sub(r"\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Square|Sq|Place|Pl|Terrace|Ter)", "[REDACTED]", redacted)
    redacted = re.sub(r"\b\d{8,12}\b", "[REDACTED]", redacted)
    return redacted

def redact_image(file_path, redaction_level="full", consent_level="none", custom_types=None):
    start_time = time.time()
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    blocks = extract_text_blocks(file_path)
    draw = ImageDraw.Draw(image)
    entity_types = get_entity_types(redaction_level, consent_level, custom_types)
    for block in blocks:
        doc = nlp(block["text"])
        for ent in doc.ents:
            if ent.label_ in entity_types:
                start_char = ent.start_char
                end_char = ent.end_char
                curr_pos = 0
                min_left, min_top, max_right, max_bottom = float('inf'), float('inf'), 0, 0
                for word in block["words"]:
                    word_len = len(word["text"])
                    if curr_pos <= end_char and curr_pos + word_len >= start_char:
                        min_left = min(min_left, word["left"])
                        min_top = min(min_top, word["top"])
                        max_right = max(max_right, word["left"] + word["width"])
                        max_bottom = max(max_bottom, word["top"] + word["height"])
                    curr_pos += word_len + 1
                if min_left < float('inf'):
                    bounds = [
                        (min_left, min_top),
                        (max_right, min_top),
                        (max_right, max_bottom),
                        (min_left, max_bottom)
                    ]
                    draw.polygon(bounds, fill="black")
    processing_time = time.time() - start_time
    return image, processing_time


def redact_pdf(file_path, redaction_level="full", consent_level="none", custom_types=None):
    output_pdf = PdfWriter()
    start_time = time.time()
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            pil_img = page.to_image(resolution=200).original.convert('RGB')
            temp_img_path = f"temp_page_{uuid.uuid4()}.png"
            pil_img.save(temp_img_path)
            blocks = extract_text_blocks(temp_img_path)
            draw = ImageDraw.Draw(pil_img)
            entity_types = get_entity_types(redaction_level, consent_level, custom_types)
            for block in blocks:
                doc = nlp(block["text"])
                for ent in doc.ents:
                    if ent.label_ in entity_types:
                        start_char = ent.start_char
                        end_char = ent.end_char
                        curr_pos = 0
                        min_left, min_top, max_right, max_bottom = float('inf'), float('inf'), 0, 0
                        for word in block["words"]:
                            word_len = len(word["text"])
                            if curr_pos <= end_char and curr_pos + word_len >= start_char:
                                min_left = min(min_left, word["left"])
                                min_top = min(min_top, word["top"])
                                max_right = max(max_right, word["left"] + word["width"])
                                max_bottom = max(max_bottom, word["top"] + word["height"])
                            curr_pos += word_len + 1
                        if min_left < float('inf'):
                            bounds = [
                                (min_left, min_top),
                                (max_right, min_top),
                                (max_right, max_bottom),
                                (min_left, max_bottom)
                            ]
                            draw.polygon(bounds, fill="black")
            os.remove(temp_img_path)
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            try:
                pil_img.save(temp_pdf, format='PDF')
                temp_pdf.close()
                temp_reader = PdfReader(temp_pdf.name)
                output_pdf.add_page(temp_reader.pages[0])
            finally:
                os.unlink(temp_pdf.name)
    processing_time = time.time() - start_time
    return output_pdf, processing_time


# Save metadata
def save_redaction_metadata(unique_id, original_path, redacted_path, processing_time):
    meta_path = os.path.join(os.path.dirname(__file__), 'redacted_files', 'history.json')
    entry = {
        'id': unique_id,
        'original': os.path.basename(original_path),
        'file': os.path.basename(redacted_path),
        'timestamp': datetime.datetime.now().isoformat(),
        'processing_time': processing_time
    }
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

@app.get("/")
def root():
    return {"status": "Apollo Health backend is live"}

@app.get('/history')
def get_redaction_history():
    meta_path = os.path.join(os.path.dirname(__file__), 'redacted_files', 'history.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    return data

@app.get('/download/{file_id}')
def download_redacted_file(file_id: str):
    redacted_dir = os.path.join(os.path.dirname(__file__), 'redacted_files')
    meta_path = os.path.join(redacted_dir, 'history.json')
    if not os.path.exists(meta_path):
        return JSONResponse(status_code=404, content={"error": "No history found"})
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        if entry['id'] == file_id:
            file_path = os.path.join(redacted_dir, entry['file'])
            if os.path.exists(file_path):
                return FileResponse(file_path, media_type='application/pdf', filename=entry['file'])
            else:
                return JSONResponse(status_code=404, content={"error": "File not found"})
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get('/preview/{file_id}')
def preview_file(file_id: str, type: str = Query('redacted')):
    if type not in ['original', 'redacted']:
        return JSONResponse(status_code=400, content={"error": "Invalid type"})
    redacted_dir = os.path.join(os.path.dirname(__file__), 'redacted_files')
    meta_path = os.path.join(redacted_dir, 'history.json')
    if not os.path.exists(meta_path):
        return JSONResponse(status_code=404, content={"error": "No history found"})
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        if entry['id'] == file_id:
            key = 'original' if type == 'original' else 'file'
            file_path = os.path.join(redacted_dir, entry[key])
            if os.path.exists(file_path):
                return FileResponse(file_path, media_type='application/pdf', headers={"Content-Disposition": "inline"})
            else:
                return JSONResponse(status_code=404, content={"error": "File not found"})
    return JSONResponse(status_code=404, content={"error": "File not found"})

from fastapi import Form

@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    redaction_level: str = Form("full"),
    consent_level: str = Form("none"),
    custom_types: str = Form(None)
):
    file_location = f"temp_{file.filename}"
    redacted_dir = os.path.join(os.path.dirname(__file__), 'redacted_files')
    os.makedirs(redacted_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    ext = file.filename.lower().split('.')[-1]
    original_path = os.path.join(redacted_dir, f'original_{unique_id}.pdf')
    redacted_path = os.path.join(redacted_dir, f'redacted_{unique_id}.pdf')
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        custom_types_list = None
        if custom_types:
            try:
                custom_types_list = json.loads(custom_types)
            except Exception:
                custom_types_list = [s.strip() for s in custom_types.split(",") if s.strip()]
        logging.info(f"File received: {file.filename}")
        logging.info(f"Redaction level: {redaction_level}")
        logging.info(f"Consent level: {consent_level}")
        logging.info(f"Custom types: {custom_types_list}")
        if ext in ["png", "jpg", "jpeg", "bmp", "gif", "webp"]:
            Image.open(file_location).save(original_path, format="PDF")
            redacted_image, processing_time = redact_image(file_location, redaction_level, consent_level, custom_types_list)
            redacted_image.save(redacted_path, format="PDF")
        elif ext == "pdf":
            shutil.copy(file_location, original_path)
            redacted_pdf, processing_time = redact_pdf(file_location, redaction_level, consent_level, custom_types_list)
            with open(redacted_path, 'wb') as f:
                redacted_pdf.write(f)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
        save_redaction_metadata(unique_id, original_path, redacted_path, processing_time, consent_level)
        return JSONResponse({
            "id": unique_id,
            "message": "Redacted successfully",
            "processing_time": processing_time,
            "redaction_level": redaction_level,
            "consent_level": consent_level,
            "custom_types": custom_types_list
        })
    except Exception as e:
        logging.error(f"Processing error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
