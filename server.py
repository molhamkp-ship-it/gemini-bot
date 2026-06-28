# server.py
import os
import io
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import uvicorn
import time

# ---------- إعدادات ----------
app = FastAPI(title="Gemini Image Editor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# قائمة النماذج الاحتياطية (مرتبة حسب الأولوية)
FALLBACK_MODELS = [
    "gemini-2.5-flash-image",  # الخيار الأول: نموذج تحرير الصور المثالي
    "gemini-2.0-flash",        # الخيار الثاني: نموذج سريع ومتعدد الوسائط
    "gemini-1.5-flash",        # الخيار الثالث: نموذج مستقر وموثوق
    "gemini-2.0-flash-lite",   # الخيار الرابع: نموذج خفيف للملاذ الأخير
]

# ---------- دالة معالجة الصورة مع آلية الاحتياطي ----------
def edit_image_with_gemini(image_bytes: bytes, prompt: str) -> bytes:
    """إرسال الصورة والتعليمات إلى Gemini API مع آلية احتياطي."""
    last_exception = None
    
    for model in FALLBACK_MODELS:
        try:
            print(f"🔄 محاولة استخدام النموذج: {model}")
            
            # تحويل الصورة إلى Base64
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            
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
            
            # إرسال الطلب مع مهلة 60 ثانية
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            # التحقق من نجاح الطلب
            if response.status_code == 200:
                data = response.json()
                
                # استخراج الصورة من الرد
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    for part in parts:
                        if "inline_data" in part:
                            img_b64 = part["inline_data"]["data"]
                            print(f"✅ نجح النموذج: {model}")
                            return base64.b64decode(img_b64)
                    # إذا وصلنا هنا، الرد ليس به صورة
                    print(f"⚠️ النموذج {model} لم يعدّ صورة، جرب النموذج التالي...")
                    continue
                    
                except (KeyError, IndexError) as e:
                    print(f"⚠️ النموذج {model} أعاد تنسيقاً غير متوقع، جرب النموذج التالي...")
                    continue
            
            # معالجة أخطاء محددة
            elif response.status_code == 429:
                print(f"⚠️ النموذج {model} تجاوز الحد اليومي (429)، جرب النموذج التالي...")
                continue
                
            elif response.status_code == 404:
                print(f"⚠️ النموذج {model} غير موجود (404)، جرب النموذج التالي...")
                continue
                
            else:
                # أي خطأ آخر (500، 403، إلخ)
                print(f"⚠️ النموذج {model} فشل برمز {response.status_code}: {response.text[:100]}...")
                continue
                
        except requests.exceptions.Timeout:
            print(f"⚠️ النموذج {model} انتهت مهلة الاتصال، جرب النموذج التالي...")
            continue
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ النموذج {model} فشل في الاتصال: {str(e)[:100]}...")
            continue
            
        except Exception as e:
            print(f"⚠️ النموذج {model} فشل بسبب خطأ غير متوقع: {str(e)[:100]}...")
            continue
    
    # إذا فشلت جميع النماذج
    raise Exception(f"❌ فشلت جميع النماذج الاحتياطية. آخر خطأ: {last_exception}")

# ---------- نقطة النهاية API ----------
@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = "قم بتحسين هذه الصورة الشخصية: أزل الخلفية واجعلها بيضاء، حسّن الإضاءة، ونعم البشرة بلطف مع الحفاظ على الملامح الطبيعية، ثم قص الصورة من أعلى الرأس إلى منتصف الصدر."
):
    """استقبال صورة وإرجاعها بعد التعديل حسب التعليمات"""
    try:
        # قراءة الصورة
        image_bytes = await file.read()
        # التحقق من أنها صورة صالحة
        try:
            Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=422, detail="الملف المرفوع ليس صورة صالحة.")
        
        # معالجة مع آلية الاحتياطي
        edited_bytes = edit_image_with_gemini(image_bytes, prompt)
        
        # إرجاع الصورة
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
    return {"status": "ok", "primary_model": FALLBACK_MODELS[0], "fallback_models": FALLBACK_MODELS[1:]}

# ---------- تشغيل الخادم ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
