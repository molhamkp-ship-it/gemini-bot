# server.py
import os
import io
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import uvicorn

# ---------- إعدادات ----------
app = FastAPI(title="Gemini Image Editor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# اقرأ النموذج من متغير البيئة، مع قيمة افتراضية آمنة
MODEL = os.getenv("MODEL", "gemini-2.5-flash-image")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# ---------- دالة معالجة الصورة ----------
def edit_image_with_gemini(image_bytes: bytes, prompt: str) -> bytes:
    """إرسال الصورة والتعليمات إلى Gemini API وإرجاع الصورة المعدلة"""
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
                ]
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    try:
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inline_data" in part:
                img_b64 = part["inline_data"]["data"]
                return base64.b64decode(img_b64)
        raise Exception("لم يتم العثور على صورة في الرد.")
    except (KeyError, IndexError) as e:
        raise Exception(f"خطأ في تنسيق الرد: {data}")

# ---------- نقطة النهاية API ----------
@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = "قم بتحسين هذه الصورة الشخصية: أزل الخلفية واجعلها بيضاء، حسّن الإضاءة، ونعم البشرة بلطف مع الحفاظ على الملامح الطبيعية، ثم قص الصورة من أعلى الرأس إلى منتصف الصدر."
):
    try:
        image_bytes = await file.read()
        try:
            Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=422, detail="الملف المرفوع ليس صورة صالحة.")
        
        edited_bytes = edit_image_with_gemini(image_bytes, prompt)
        return Response(content=edited_bytes, media_type="image/jpeg")
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="انتهت المهلة أثناء الاتصال بـ Gemini API.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"خطأ في الاتصال بـ Gemini API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- نقطة للتحقق من الصحة ----------
@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL}

# ---------- تشغيل الخادم ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
