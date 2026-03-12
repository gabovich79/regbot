import os
import re
import fitz  # pymupdf
from docx import Document
import httpx
from bs4 import BeautifulSoup
from config import DOCUMENTS_DIR

os.makedirs(DOCUMENTS_DIR, exist_ok=True)


from services.token_utils import estimate_tokens


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return clean_text("\n".join(pages))


def extract_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(paragraphs))


def extract_pdf_bytes(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return clean_text("\n".join(pages))


def extract_docx_bytes(content: bytes) -> str:
    import io
    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(paragraphs))


async def fetch_url_text(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        if "pdf" in content_type:
            return extract_pdf_bytes(response.content)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return clean_text(text)


def extract_gdrive_file_id(url: str) -> str | None:
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def fetch_gdrive_text(url: str) -> str:
    file_id = extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError("לא ניתן לחלץ File ID מהקישור של Google Drive")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(download_url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        if "pdf" in content_type:
            return extract_pdf_bytes(response.content)
        elif "document" in content_type or "docx" in content_type:
            return extract_docx_bytes(response.content)
        else:
            return clean_text(response.text)


def save_document_text(doc_id: int, text: str) -> str:
    path = os.path.join(DOCUMENTS_DIR, f"{doc_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def load_document_text(text_path: str) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def delete_document_file(text_path: str):
    if os.path.exists(text_path):
        os.remove(text_path)
