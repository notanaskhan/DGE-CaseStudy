
import os, subprocess, shutil

def is_tool(name: str) -> bool:
    return shutil.which(name) is not None

def ocr_pdf(filepath: str, out_txt_path: str):
    result = {"ok": False, "engine": None, "text_path": out_txt_path, "warning": None}
    try:
        if is_tool("ocrmypdf"):
            ocr_out_pdf = out_txt_path.replace(".txt", ".ocr.pdf")
            subprocess.run(["ocrmypdf", "--force-ocr", filepath, ocr_out_pdf],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if is_tool("pdftotext"):
                subprocess.run(["pdftotext", ocr_out_pdf, out_txt_path],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(out_txt_path, "r", errors="ignore") as f:
                    txt = f.read()
                result.update({"ok": True, "engine": "ocrmypdf+pdftotext", "chars": len(txt)})
                return result
            else:
                result["warning"] = "pdftotext not found; OCRed PDF produced but text not extracted."
                result["ok"] = True
                result["engine"] = "ocrmypdf"
                return result

        if is_tool("pdftotext"):
            subprocess.run(["pdftotext", filepath, out_txt_path], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_txt_path, "r", errors="ignore") as f:
                txt = f.read()
            result.update({"ok": True, "engine": "pdftotext", "chars": len(txt)})
            return result

        try:
            from pdfminer.high_level import extract_text
            txt = extract_text(filepath) or ""
            with open(out_txt_path, "w") as f:
                f.write(txt)
            if txt.strip():
                result.update({"ok": True, "engine": "pdfminer", "chars": len(txt)})
            else:
                result["warning"] = "No text layer found and no OCR tools installed."
            return result
        except Exception as ie:
            result["warning"] = f"pdfminer fallback failed: {ie}"
            return result
    except Exception as e:
        result["warning"] = f"OCR error: {e}"
        return result

def ocr_image(filepath: str, out_txt_path: str):
    result = {"ok": False, "engine": None, "text_path": out_txt_path, "warning": None}
    try:
        try:
            import pytesseract
            from PIL import Image
        except Exception as ie:
            result["warning"] = "pytesseract/Pillow not installed. Skipping image OCR."
            return result
        text = pytesseract.image_to_string(Image.open(filepath))
        with open(out_txt_path, "w") as f:
            f.write(text)
        result.update({"ok": True, "engine": "pytesseract", "chars": len(text)})
        return result
    except Exception as e:
        result["warning"] = f"Image OCR error: {e}"
        return result

def run_ocr(filepath: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(filepath)
    out_txt = os.path.join(out_dir, f"{base}.txt")
    ext = os.path.splitext(base)[1].lower()

    if ext in [".csv", ".xls", ".xlsx"]:
        return {"ok": True, "engine": None, "text_path": None, "warning": "OCR not applicable for spreadsheets."}

    if ext == ".pdf":
        return ocr_pdf(filepath, out_txt)
    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        return ocr_image(filepath, out_txt)
    return {"ok": False, "warning": f"Unsupported for OCR: {ext}", "text_path": out_txt}
