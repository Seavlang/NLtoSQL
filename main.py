from typing import Optional
import httpx
from fastapi import HTTPException, Query, FastAPI
from pydantic import BaseModel

# ---- Models ----
class TranslationRequest(BaseModel):
    text: str  # source text (Korean for these endpoints)

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str                    # KR -> EN result
    back_translated_text: Optional[str] = None  # EN -> KR result when roundtrip=True

app = FastAPI()
# ---- Helpers ----
async def ollama_generate(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    return (data.get("response") or "").strip()


def prompt_translate(src_lang: str, tgt_lang: str, text: str) -> str:
    return (
        f"Translate the following text from {src_lang} to {tgt_lang}. "
        "Only return the translated sentence without any prefix or explanation.\n\n"
        f"Text: {text}\n"
        "Translation:"
    )



# ---- Endpoints ----
@app.post("/gemma-kr-to-eng", response_model=TranslationResponse, summary="Gemma3: Korean → English (optional round-trip)")
async def translate_text_gemma(request: TranslationRequest, roundtrip: bool = Query(False, description="Also translate EN → KR")):
    model = "zongwei/gemma3-translator:1b"
    # KR -> EN
    en = await ollama_generate(model, prompt_translate("Korean", "English", request.text))
    # EN -> KR (optional)
    back_kr = None
    if roundtrip:
        back_kr = await ollama_generate(model, prompt_translate("English", "Korean", en))
    return TranslationResponse(original_text=request.text, translated_text=en, back_translated_text=back_kr)


@app.post("/llama3B-kr-to-eng", response_model=TranslationResponse, summary="Llama 3B: Korean → English (optional round-trip)")
async def translate_text_llama3b(request: TranslationRequest, roundtrip: bool = Query(False, description="Also translate EN → KR")):
    model = "timHan/llama3.2korean3B4QKM"
    en = await ollama_generate(model, prompt_translate("Korean", "English", request.text))
    back_kr = None
    if roundtrip:
        back_kr = await ollama_generate(model, prompt_translate("English", "Korean", en))
    return TranslationResponse(original_text=request.text, translated_text=en, back_translated_text=back_kr)

@app.post("/llama8B-kr-to-eng", response_model=TranslationResponse, summary="Llama 8B: Korean → English (optional round-trip)")
async def translate_text_llama8b(request: TranslationRequest, roundtrip: bool = Query(False, description="Also translate EN → KR")):
    model = "timHan/llama3korean8B4QKM"
    en = await ollama_generate(model, prompt_translate("Korean", "English", request.text))
    back_kr = None
    if roundtrip:
        back_kr = await ollama_generate(model, prompt_translate("English", "Korean", en))
    return TranslationResponse(original_text=request.text, translated_text=en, back_translated_text=back_kr)

