from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter
import uuid
import json
import datetime
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from PIL import Image, ImageDraw
import pdfplumber
import spacy
import traceback
import re
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

nlp = spacy.load("en_core_web_sm")


# Redaction entity sets
FULL_ENTITIES = ["PERSON", "GPE", "ORG", "DATE", "CARDINAL", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "LANGUAGE"]
PARTIAL_ENTITIES = ["PERSON", "CARDINAL", "LOC", "NORP"]  # direct identifiers, can adjust as needed

def redact_text(text, redaction_level="full", custom_types=None):
    doc = nlp(text)
    redacted = text
    # Choose which entity types to redact
    if redaction_level == "full":
        entity_types = FULL_ENTITIES
    elif redaction_level == "partial":
        entity_types = PARTIAL_ENTITIES
    elif redaction_level == "custom" and custom_types:
        entity_types = custom_types
    else:
        entity_types = FULL_ENTITIES
    for ent in doc.ents:
        if ent.label_ in entity_types:
            redacted = redacted.replace(ent.text, "[REDACTED]")
    # Always redact phone, email, address, and ID numbers for all levels
    redacted = re.sub(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}", "[REDACTED]", redacted)
    redacted = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED]", redacted)
    redacted = re.sub(r"\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Square|Sq|Place|Pl|Terrace|Ter)", "[REDACTED]", redacted)
    redacted = re.sub(r"\b\d{8,12}\b", "[REDACTED]", redacted)
    return redacted

# Redact entities in images by blacking out detected text regions
def redact_image(file_path, redaction_level="full", custom_types=None):
    start_time = time.time()
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    draw = ImageDraw.Draw(image)
    # Choose which entity types to redact
    if redaction_level == "full":
        entity_types = FULL_ENTITIES
    elif redaction_level == "partial":
        entity_types = PARTIAL_ENTITIES
    elif redaction_level == "custom" and custom_types:
        entity_types = custom_types
    else:
        entity_types = FULL_ENTITIES
    for i in range(n_boxes):
        text = data['text'][i]
        if text.strip() == "":
            continue
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in entity_types:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                draw.rectangle([x, y, x + w, y + h], fill="black")
    # Save to redacted_files directory with unique name
    redacted_dir = os.path.join(os.path.dirname(__file__), 'redacted_files')
    os.makedirs(redacted_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    out_path = os.path.join(redacted_dir, f'redacted_{unique_id}.pdf')
    image.save(out_path, format="PDF")
    processing_time = time.time() - start_time
    # Save metadata
    save_redaction_metadata(unique_id, out_path, processing_time)
    output = open(out_path, 'rb')
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=redacted_{unique_id}.pdf"}
    )

# Redact entities in PDF by rendering each page as image, blacking out, and reassembling
def redact_pdf(file_path, redaction_level="full", custom_types=None):
    import pdfplumber
    from PIL import Image
    from PyPDF2 import PdfWriter
    import tempfile
    output_pdf = PdfWriter()
    start_time = time.time()
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            pil_img = page.to_image(resolution=200).original.convert('RGB')
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['level'])
            draw = ImageDraw.Draw(pil_img)
            # Choose which entity types to redact
            if redaction_level == "full":
                entity_types = FULL_ENTITIES
            elif redaction_level == "partial":
                entity_types = PARTIAL_ENTITIES
            elif redaction_level == "custom" and custom_types:
                entity_types = custom_types
            else:
                entity_types = FULL_ENTITIES
            for i in range(n_boxes):
                text = data['text'][i]
                if text.strip() == "":
                    continue
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in entity_types:
                        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                        draw.rectangle([x, y, x + w, y + h], fill="black")
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            try:
                pil_img.save(temp_pdf, format='PDF')
                temp_pdf.close()
                from PyPDF2 import PdfReader
                temp_reader = PdfReader(temp_pdf.name)
                output_pdf.add_page(temp_reader.pages[0])
            finally:
                os.unlink(temp_pdf.name)
    # Save to redacted_files directory with unique name
    redacted_dir = os.path.join(os.path.dirname(__file__), 'redacted_files')
    os.makedirs(redacted_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    out_path = os.path.join(redacted_dir, f'redacted_{unique_id}.pdf')
    with open(out_path, 'wb') as f:
        output_pdf.write(f)
    processing_time = time.time() - start_time
    save_redaction_metadata(unique_id, out_path, processing_time)
    output = open(out_path, 'rb')
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=redacted_{unique_id}.pdf"}
    )

# Save metadata for each redacted file, now with processing_time
def save_redaction_metadata(unique_id, file_path, processing_time):
    meta_path = os.path.join(os.path.dirname(__file__), 'redacted_files', 'history.json')
    entry = {
        'id': unique_id,
        'file': os.path.basename(file_path),
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

# Endpoint to get redaction history

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/history')
def get_redaction_history():
    meta_path = os.path.join(os.path.dirname(__file__), 'redacted_files', 'history.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    return data

# Endpoint to download a redacted file by id
from fastapi.responses import FileResponse
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

def generate_pdf_stream(text: str) -> StreamingResponse:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = [Paragraph(line, styles["Normal"]) for line in text.split('\n')]
    doc.build(flowables)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=redacted_output.pdf"}
    )

from fastapi import Form

@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    redaction_level: str = Form("full"),
    custom_types: str = Form(None)
):
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        ext = file.filename.lower().split('.')[-1]
        # Parse custom_types if provided
        custom_types_list = None
        if custom_types:
            try:
                custom_types_list = json.loads(custom_types)
            except Exception:
                custom_types_list = [s.strip() for s in custom_types.split(",") if s.strip()]

        if ext in ["png", "jpg", "jpeg", "bmp", "gif", "webp"]:
            return redact_image(file_location, redaction_level, custom_types_list)
        elif ext == "pdf":
            return redact_pdf(file_location, redaction_level, custom_types_list)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    except Exception as e:
        print("Processing error:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

    finally:
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
        except Exception as cleanup_err:
            print("Cleanup error:", cleanup_err)
import os