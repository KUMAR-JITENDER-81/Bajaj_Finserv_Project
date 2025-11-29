
# app.py
import io, os, re, zipfile, tempfile
from typing import Optional, List
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import requests
from PIL import Image, ImageSequence
import pytesseract
import numpy as np
import cv2

# Optional PDF support
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

app = FastAPI(title="Bill Extraction API (Deployed)")

class RequestModel(BaseModel):
    document: str

# regex patterns
AMOUNT_RE = re.compile(r'(?<!\d)(\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?)(?!\d)')
CURRENCY_RE = re.compile(r'([\₹$€£]|INR|Rs\.?)', re.IGNORECASE)

# ---- helpers ----
def download_bytes_from_url(url: str, timeout: int = 30) -> bytes:
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided.")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme == "":
        url = "https://" + url
        parsed = urlparse(url)
    if parsed.scheme not in ("http","https"):
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {parsed.scheme}")
    if parsed.hostname is None or "_" in parsed.hostname or " " in parsed.hostname:
        raise HTTPException(status_code=400, detail=f"URL hostname invalid: {parsed.hostname}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {str(e)}")
    return resp.content

def is_zip_bytes(b: bytes) -> bool:
    return b[:4] == b'PK\x03\x04'

def is_pdf_bytes(b: bytes) -> bool:
    return b[:4] == b'%PDF'

def pil_to_cv(img_pil: Image.Image):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def preprocess_image_cv(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15)
    return th

def run_tesseract_with_boxes(img_pil: Image.Image):
    img_cv = pil_to_cv(img_pil)
    proc = preprocess_image_cv(img_cv)
    pil = Image.fromarray(proc)
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    words = []
    n = len(data['level'])
    for i in range(n):
        text = str(data['text'][i]).strip()
        if text == "":
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        words.append({
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': conf
        })
    return words

def group_words_to_lines(words, y_tol=10):
    lines = []
    words_sorted = sorted(words, key=lambda w: (w['top'], w['left']))
    for w in words_sorted:
        placed = False
        cy = w['top'] + w['height']//2
        for line in lines:
            if abs(cy - line['avg_y']) <= y_tol:
                line['words'].append(w)
                line['avg_y'] = int(sum([ww['top'] + ww['height']//2 for ww in line['words']]) / len(line['words']))
                placed = True
                break
        if not placed:
            lines.append({'avg_y': cy, 'words': [w]})
    out = []
    for l in lines:
        wl = sorted(l['words'], key=lambda x: x['left'])
        text = ' '.join([w['text'] for w in wl])
        left = min(w['left'] for w in wl)
        top = min(w['top'] for w in wl)
        right = max(w['left'] + w['width'] for w in wl)
        bottom = max(w['top'] + w['height'] for w in wl)
        out.append({'text': text, 'left': left, 'top': top, 'right': right, 'bottom': bottom, 'words': wl})
    return out

def parse_amount_from_text(text):
    matches = list(AMOUNT_RE.finditer(text.replace(',', '')))
    if not matches:
        return None
    last = matches[-1].group(1)
    try:
        return float(last)
    except:
        return None

def detect_totals_and_items(lines):
    page_items = []
    subtotals = []
    totals = []
    for ln in lines:
        t = ln['text'].strip()
        tl = t.lower()
        amt = parse_amount_from_text(t)
        if 'sub total' in tl or 'subtotal' in tl or 'sub-total' in tl:
            if amt is not None:
                subtotals.append({'text': t, 'amount': amt, 'bbox': (ln['left'], ln['top'], ln['right'], ln['bottom'])})
            continue
        if re.search(r'\b(total|grand total|net amount|amount payable)\b', tl):
            if amt is not None:
                totals.append({'text': t, 'amount': amt, 'bbox': (ln['left'], ln['top'], ln['right'], ln['bottom'])})
            continue
        if amt is not None and not any(k in tl for k in ['tax','gst','discount','vat','invoice','bill','balance','due','amount payable']):
            qty = None
            rate = None
            m = re.search(r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)', t)
            if m:
                try:
                    qty = float(m.group(1)); rate = float(m.group(2))
                except:
                    pass
            else:
                nums = [float(m.group(1)) for m in AMOUNT_RE.finditer(t.replace(',', ''))]
                if len(nums) >= 2:
                    rate = nums[-2]
            name = re.sub(r'[\d,]*\.\d{1,2}\s*$', '', t).strip()
            page_items.append({
                'item_name': name,
                'item_amount': amt,
                'item_rate': rate if rate is not None else None,
                'item_quantity': qty if qty is not None else None,
                'bbox': (ln['left'], ln['top'], ln['right'], ln['bottom']),
                'raw_text': t
            })
    return page_items, subtotals, totals

def dedupe_items(items, iou_threshold=0.5, text_sim_threshold=0.85):
    keep = []
    from difflib import SequenceMatcher
    def iou(a,b):
        leftA,topA,rightA,bottomA = a
        leftB,topB,rightB,bottomB = b
        interLeft = max(leftA,leftB); interTop = max(topA,topB)
        interRight = min(rightA,rightB); interBottom = min(bottomA,bottomB)
        if interRight < interLeft or interBottom < interTop:
            return 0.0
        inter = (interRight-interLeft)*(interBottom-interTop)
        areaA = (rightA-leftA)*(bottomA-topA); areaB = (rightB-leftB)*(bottomB-topB)
        union = areaA + areaB - inter
        if union <= 0: return 0.0
        return inter/union
    for it in items:
        dup = False
        for k in keep:
            if k['item_amount'] is not None and it['item_amount'] is not None:
                if abs(k['item_amount'] - it['item_amount']) < 0.001 and iou(k['bbox'], it['bbox']) > iou_threshold:
                    dup = True; break
            sim = SequenceMatcher(None, k['item_name'], it['item_name']).ratio()
            if sim > text_sim_threshold and iou(k['bbox'], it['bbox']) > 0.1:
                dup = True; break
        if not dup:
            keep.append(it)
    return keep

# image/pdf/zip -> pages
def extract_pages_from_bytes(content: bytes):
    pages = []
    if is_pdf_bytes(content):
        if not PDF2IMAGE_AVAILABLE:
            raise HTTPException(status_code=400, detail="PDF received but pdf2image/poppler not available.")
        pil_pages = convert_from_bytes(content)
        for p in pil_pages:
            pages.append(p.convert("RGB"))
        return pages
    else:
        try:
            pil = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to open image/pdf bytes: {str(e)}")
        # multi-frame support (TIFF/GIF)
        frames = []
        try:
            for f in ImageSequence.Iterator(pil):
                frames.append(f.convert("RGB"))
        except Exception:
            frames = [pil.convert("RGB")]
        return frames

def extract_from_zip_bytes(content: bytes):
    pages = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.pdf')):
                b = z.read(name)
                subpages = extract_pages_from_bytes(b)
                pages.extend(subpages)
    return pages

def process_image_page(pil_img, doc_id, page_no):
    words = run_tesseract_with_boxes(pil_img)
    lines = group_words_to_lines(words, y_tol=10)
    page_items, subtotals, totals = detect_totals_and_items(lines)
    deduped = dedupe_items(page_items)
    formatted = []
    for it in deduped:
        formatted.append({
            "item_name": it['item_name'],
            "item_amount": float(it['item_amount']) if it['item_amount'] is not None else None,
            "item_rate": float(it['item_rate']) if it['item_rate'] is not None else None,
            "item_quantity": float(it['item_quantity']) if it['item_quantity'] is not None else None
        })
    page_entry = {"page_no": f"{doc_id}_p{page_no}", "page_type": "Bill Detail", "bill_items": formatted}
    return page_entry, deduped, subtotals, totals

# middleware to print raw request body for debugging (optional)
@app.middleware("http")
async def log_request_body(request: Request, call_next):
    body = await request.body()
    try:
        snippet = body.decode("utf-8", errors="replace")
    except:
        snippet = str(body)
    print("RAW REQUEST BODY (first 2000 chars):")
    print(snippet[:2000])
    return await call_next(request)

# ---- endpoint ----
@app.post("/extract-bill-data")
async def extract_bill_data(req: Optional[RequestModel] = Body(None), file: Optional[UploadFile] = File(None)):
    # token usage placeholders (0 if no LLM used)
    token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # Acquire bytes
    content_bytes = None
    src_desc = "unknown"
    if file is not None:
        content_bytes = await file.read()
        src_desc = getattr(file, "filename", "uploaded_file")
    else:
        if req is None or not getattr(req, "document", None):
            raise HTTPException(status_code=400, detail="No document URL provided and no file uploaded.")
        content_bytes = download_bytes_from_url(req.document)
        src_desc = req.document

    # Extract pages to PIL images
    pages = []
    try:
        if is_zip_bytes(content_bytes):
            pages = extract_from_zip_bytes(content_bytes)
        else:
            pages = extract_pages_from_bytes(content_bytes)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded/downloaded file: {str(e)}")

    if not pages:
        raise HTTPException(status_code=400, detail="No pages found to process.")

    pagewise_line_items = []
    all_items = []
    all_subtotals = []
    all_totals = []

    for idx, pil_page in enumerate(pages, start=1):
        doc_id = os.path.basename(src_desc).replace('.', '_')
        page_entry, deduped, subt, tot = process_image_page(pil_page, doc_id, idx)
        pagewise_line_items.append(page_entry)
        all_items.extend(deduped)
        all_subtotals.extend(subt)
        all_totals.extend(tot)

    # global dedupe
    final_items = dedupe_items(all_items, iou_threshold=0.4, text_sim_threshold=0.8)
    computed_final_total = sum(it['item_amount'] for it in final_items if it['item_amount'] is not None)

    response = {
        "is_success": True,
        "token_usage": token_usage,
        "data": {
            "pagewise_line_items": pagewise_line_items,
            "total_item_count": int(len(final_items)),
            "computed_final_total": float(round(computed_final_total, 2)),
            "detected_subtotals": all_subtotals,
            "detected_totals": all_totals
        }
    }
    return JSONResponse(status_code=200, content=response)
