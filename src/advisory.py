"""India NAQI (PM2.5 sub-index) + profile-specific guidance in English and Tamil,
with an optional GenAI layer that rewrites the advice from the computed facts only."""
import json
import os
import time

import requests

# CPCB NAQI breakpoints for 24-h PM2.5 (ug/m3) -> AQI
BREAKS = [(0, 30, 0, 50, "Good"), (31, 60, 51, 100, "Satisfactory"), (61, 90, 101, 200, "Moderate"),
          (91, 120, 201, 300, "Poor"), (121, 250, 301, 400, "Very Poor"), (251, 500, 401, 500, "Severe")]
COLORS = {"Good": "#2e9d4f", "Satisfactory": "#9ccc3c", "Moderate": "#f2c12e",
          "Poor": "#ef8a2b", "Very Poor": "#d7332f", "Severe": "#7e1f1f"}

# Tamil names for the NAQI categories
CAT_TA = {"Good": "நல்லது", "Satisfactory": "திருப்திகரமானது", "Moderate": "மிதமானது",
          "Poor": "மோசமானது", "Very Poor": "மிகவும் மோசமானது", "Severe": "கடுமையானது"}


def pm25_to_aqi(c):
    c = max(0.0, float(c))
    for lo, hi, alo, ahi, _ in BREAKS:
        if c <= hi + 0.999:
            return round(alo + (ahi - alo) * (min(c, hi) - lo) / (hi - lo))
    return 500


def aqi_category(c):
    a = pm25_to_aqi(c)
    for _, _, alo, ahi, name in BREAKS:
        if a <= ahi:
            return name, a
    return "Severe", a


DAYS_TA = {"Mon": "திங்கள்", "Tue": "செவ்வாய்", "Wed": "புதன்", "Thu": "வியாழன்", "Fri": "வெள்ளி",
           "Sat": "சனி", "Sun": "ஞாயிறு"}


def localize_time(s, lang="en"):
    """'Sun 07:00' -> 'ஞாயிறு 07:00' in Tamil."""
    if lang != "ta" or not s:
        return s
    for en, ta in DAYS_TA.items():
        s = s.replace(en, ta)
    return s


def cat_label(cat, lang="en"):
    return CAT_TA.get(cat, cat) if lang == "ta" else cat


ADVICE = {
    "en": {
        "general": {"Good": "Enjoy outdoor activity.", "Satisfactory": "Fine for most people.",
                    "Moderate": "Reduce long, intense outdoor exertion.",
                    "Poor": "Limit time outdoors; prefer the cleaner route.",
                    "Very Poor": "Avoid outdoor exertion; travel only if needed.",
                    "Severe": "Stay indoors where possible."},
        "sensitive": {"Good": "Safe for outdoor plans.",
                      "Satisfactory": "Generally fine; carry reliever medication if prescribed.",
                      "Moderate": "Choose the cleanest route and shorter outdoor time; avoid peak hours.",
                      "Poor": "Postpone non-essential trips; if you must go, use the cleanest route and a well-fitted mask.",
                      "Very Poor": "Avoid going out; keep medication at hand and follow your doctor's plan.",
                      "Severe": "Stay indoors with windows closed; seek medical help if symptoms worsen."},
    },
    "ta": {
        "general": {"Good": "வெளிப்புற செயல்பாடுகளுக்கு ஏற்ற நேரம்.",
                    "Satisfactory": "பெரும்பாலானோருக்கு பாதுகாப்பானது.",
                    "Moderate": "நீண்ட நேரக் கடினமான வெளிப்புற உழைப்பைக் குறைக்கவும்.",
                    "Poor": "வெளியில் இருக்கும் நேரத்தைக் குறைக்கவும்; தூய்மையான வழியைத் தேர்ந்தெடுக்கவும்.",
                    "Very Poor": "வெளிப்புற உழைப்பைத் தவிர்க்கவும்; அவசியமானால் மட்டும் பயணிக்கவும்.",
                    "Severe": "முடிந்தவரை வீட்டிற்குள்ளேயே இருக்கவும்."},
        "sensitive": {"Good": "வெளிப்புறத் திட்டங்களுக்குப் பாதுகாப்பானது.",
                      "Satisfactory": "பொதுவாகப் பாதுகாப்பானது; மருத்துவர் பரிந்துரைத்த மருந்தை உடன் வைத்திருக்கவும்.",
                      "Moderate": "தூய்மையான வழியைத் தேர்ந்தெடுத்து, வெளியில் இருக்கும் நேரத்தைக் குறைக்கவும்; நெரிசல் நேரங்களைத் தவிர்க்கவும்.",
                      "Poor": "அவசியமில்லாத பயணங்களைத் தள்ளிப்போடவும்; செல்ல வேண்டுமெனில் தூய்மையான வழியையும் சரியாகப் பொருந்தும் முகக்கவசத்தையும் பயன்படுத்தவும்.",
                      "Very Poor": "வெளியே செல்வதைத் தவிர்க்கவும்; மருந்துகளைக் கைவசம் வைத்து, மருத்துவரின் அறிவுரையைப் பின்பற்றவும்.",
                      "Severe": "ஜன்னல்களை மூடி வீட்டிற்குள்ளேயே இருக்கவும்; அறிகுறிகள் மோசமானால் உடனே மருத்துவ உதவி பெறவும்."},
    },
}

TEXT = {
    "en": {"expected": "Expected AQI {aqi} ({cat})", "worst": " - worst likely case",
           "saving": "The cleaner route cuts estimated inhaled PM2.5 by {pct:.0f}%.",
           "best": "Lowest-exposure departure in the next 24 h: {t}."},
    "ta": {"expected": "எதிர்பார்க்கப்படும் AQI {aqi} ({cat})", "worst": " - மோசமான நிலைக் கணிப்பு",
           "saving": "தூய்மையான வழி, நீங்கள் சுவாசிக்கும் PM2.5 அளவை {pct:.0f}% குறைக்கிறது.",
           "best": "அடுத்த 24 மணி நேரத்தில் குறைந்த மாசுபாட்டுடன் புறப்பட சிறந்த நேரம்: {t}."},
}


def guidance(profile, pm_med, pm_p90, best_departure=None, saving_pct=None, lang="en"):
    """Rule-based advisory. Sensitive profiles are judged on the p90 (worst likely) value."""
    lang = lang if lang in ADVICE else "en"
    sensitive = profile in ("asthma", "elderly", "child")
    basis = pm_p90 if sensitive else pm_med
    cat, aqi = aqi_category(basis)
    T = TEXT[lang]
    msg = ADVICE[lang]["sensitive" if sensitive else "general"][cat]
    parts = [T["expected"].format(aqi=aqi, cat=cat_label(cat, lang)) + (T["worst"] if sensitive else "") + ". " + msg]
    if saving_pct is not None and saving_pct >= 5:
        parts.append(T["saving"].format(pct=saving_pct))
    if best_departure:
        parts.append(T["best"].format(t=localize_time(best_departure, lang)))
    return {"category": cat, "category_label": cat_label(cat, lang), "aqi": aqi, "color": COLORS[cat],
            "text": " ".join(parts), "lang": lang}


# ------------------------------------------------------------------ optional GenAI layer
#  Provider 1: any OpenAI-compatible server (llama.cpp / Ollama on LinuxONE, watsonx proxy ...)
#              -> set LLM_URL (+ LLM_MODEL, LLM_KEY)
#  Provider 2: a GGUF model run locally on LinuxONE with llama-cpp-python -> set LOCAL_GGUF=<path to .gguf>
#  Provider 3: a small local Hugging Face model via `transformers` -> set LOCAL_LLM=<model id>
#  Otherwise the rule-based text above is used, so the app always works.
MAX_TOKENS = {"travel": 90, "bulletin": 150}     # shorter answers = faster on CPU
_local = {"pipe": None, "err": None, "llama": None, "llama_err": None}


def genai_status():
    if os.getenv("LLM_URL"):
        return {"available": True, "provider": f"server {os.getenv('LLM_MODEL', 'local')}", "languages": ["en", "ta"]}
    if os.getenv("LOCAL_GGUF"):
        try:
            import llama_cpp  # noqa: F401
            ok = os.path.exists(os.getenv("LOCAL_GGUF"))
            return {"available": ok, "provider": f"llama.cpp on LinuxONE ({os.path.basename(os.getenv('LOCAL_GGUF'))})",
                    "languages": ["en"], **({} if ok else {"reason": "model file not found"})}
        except Exception as e:
            return {"available": False, "provider": None, "reason": f"llama_cpp not importable: {e}"}
    if os.getenv("LOCAL_LLM"):
        try:
            import transformers  # noqa: F401
            return {"available": True, "provider": f"local {os.getenv('LOCAL_LLM')}", "languages": ["en"]}
        except Exception as e:
            return {"available": False, "provider": None, "reason": f"transformers not importable: {e}"}
    return {"available": False, "provider": None, "reason": "no model configured (set LLM_URL or LOCAL_LLM)"}


def _prompt(context, lang, task="travel"):
    language = "Tamil" if lang == "ta" else "English"
    if task == "bulletin":
        return ("You are a public-health officer in Chennai, India. Using ONLY the facts below, write a short "
                f"situation bulletin in {language} for city health officials: 3-4 sentences covering where and when "
                "air quality will be worst in the next 24 hours, who is at risk, and the recommended actions. "
                "Do not invent numbers, places or times.\n"
                f"FACTS: {json.dumps(context, ensure_ascii=False)}")
    return ("You are a careful public-health assistant for Chennai, India. Using ONLY the facts below, write "
            f"2-3 short, friendly sentences of travel advice in {language} for the person described. "
            "Do not invent numbers, places or medical claims. Mention the best time to leave if it is given.\n"
            f"FACTS: {json.dumps(context, ensure_ascii=False)}")


def llm_explain(context, lang="en", task="travel"):
    """Returns {"text", "provider", "ms"} or None when no model is configured / generation fails."""
    t0 = time.perf_counter()
    url = os.getenv("LLM_URL")
    if url:
        try:
            r = requests.post(url.rstrip("/") + "/v1/chat/completions", timeout=60,
                              headers={"Authorization": f"Bearer {os.getenv('LLM_KEY', 'none')}"},
                              json={"model": os.getenv("LLM_MODEL", "local"), "temperature": 0.2, "max_tokens": 200,
                                    "messages": [{"role": "user", "content": _prompt(context, lang, task)}]})
            text = r.json()["choices"][0]["message"]["content"].strip()
            return {"text": text, "provider": os.getenv("LLM_MODEL", "server"), "ms": round(1000 * (time.perf_counter() - t0))}
        except Exception:
            return None
    gguf = os.getenv("LOCAL_GGUF")
    if gguf and lang == "en":                          # small local models are not reliable in Tamil
        try:
            if _local["llama"] is None and _local["llama_err"] is None:
                from llama_cpp import Llama
                _local["llama"] = Llama(model_path=gguf, n_ctx=1024, n_threads=os.cpu_count(), verbose=False)
            if _local["llama"] is None:
                return None
            out = _local["llama"].create_chat_completion(
                messages=[{"role": "user", "content": _prompt(context, lang, task)}], max_tokens=MAX_TOKENS[task], temperature=0.2)
            text = out["choices"][0]["message"]["content"].strip()
            return {"text": text, "provider": "llama.cpp on LinuxONE · " + os.path.basename(gguf),
                    "ms": round(1000 * (time.perf_counter() - t0))}
        except Exception as e:
            _local["llama_err"] = str(e)
            return None
    model_id = os.getenv("LOCAL_LLM")
    if model_id and lang == "en":                      # small local models are not reliable in Tamil
        try:
            if _local["pipe"] is None and _local["err"] is None:
                from transformers import pipeline
                _local["pipe"] = pipeline("text-generation", model=model_id)
            if _local["pipe"] is None:
                return None
            msgs = [{"role": "user", "content": _prompt(context, lang, task)}]
            out = _local["pipe"](msgs, max_new_tokens=120, do_sample=False)
            text = out[0]["generated_text"][-1]["content"].strip()
            return {"text": text, "provider": model_id, "ms": round(1000 * (time.perf_counter() - t0))}
        except Exception as e:
            _local["err"] = str(e)
            return None
    return None


def warm_up():
    """Load the local model once in the background so the first user request doesn't pay the load time."""
    if os.getenv("LOCAL_GGUF"):
        llm_explain({"note": "warm-up"}, "en")
