import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from metrics import wcag_like_contrast_ratio, edge_density, fleiss_kappa_from_long, avg_cohen_kappa_pairwise

import os, unicodedata
from uuid import uuid4
from datetime import date
from pathlib import Path
import json, random, time
from zipfile import ZipFile, is_zipfile
import shutil
import math  

import base64
ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)  # pon tu imagen en assets/hero_gen-edviz.png
HERO_IMG = ASSETS / "hero_gen-edviz.png"


import re
from difflib import SequenceMatcher
try:
    from rapidfuzz import fuzz as rf_fuzz   # opcional
except Exception:
    rf_fuzz = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="GEN-EDViz Monitor", layout="wide", initial_sidebar_state="collapsed")

# Cargar configuración LLM desde ENV y, solo si existe, desde secrets.toml
from pathlib import Path

def _safe_load_secrets_to_env():
    KEYS = ("LLM_ENABLE","LLM_PROVIDER","OPENAI_API_KEY","LLM_MODEL","OLLAMA_URL","OLLAMA_MODEL")

    # 1) Si ya están en variables de entorno, no hacemos nada más
    if all(os.getenv(k) for k in KEYS):
        return

    # 2) Sólo intentamos leer st.secrets si realmente hay un secrets.toml presente
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    proj_secrets = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if user_secrets.exists() or proj_secrets.exists():
        try:
            for k in KEYS:
                if k in st.secrets and not os.getenv(k):
                    os.environ[k] = str(st.secrets[k])
        except Exception:
            # Si algo falla, seguimos sin romper la app
            pass

_safe_load_secrets_to_env()



def rerun():
    # Compatible con versiones nuevas y antiguas
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _safe_path(v) -> str:
        """
        Devuelve '' si v es None / NaN / 'nan'.
        Si viene algo válido, devuelve str(v). No valida existencia en disco.
        """
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        s = str(v).strip()
        return "" if s.lower() == "nan" else s

# ==== HELPERS para generación automática de preguntas/distractores ====
def _norm_txt(s: str) -> str:
    s = (s or "")
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _ensure_unique_options(opts: list[str]) -> list[str]:
    out, seen = [], set()
    for o in opts:
        o2 = _norm_txt(o)
        if not o2:
            continue
        k = o2.lower()
        if k in seen:
            continue
        out.append(o2)
        seen.add(k)
    return out

def _tfidf_distractors(correct: str,
                       corpus_prompts: list[str],
                       corpus_ids: list[str] | None = None,
                       k: int = 3) -> list[str]:
    """
    Devuelve hasta k prompts de corpus similares al 'correct' (sin repetir el correcto).
    """
    correct_n = _norm_txt(correct)
    if not correct_n:
        return []
    # Construimos el corpus con el correcto como query
    base = [correct_n] + [ _norm_txt(x) for x in corpus_prompts ]
    try:
        X = TfidfVectorizer(min_df=1).fit_transform(base)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()
        order = sims.argsort()[::-1]
    except Exception:
        # fallback por si hay algún problema con TF-IDF
        order = list(range(len(corpus_prompts)))

    out, seen = [], set()
    for idx in order:
        cand = _norm_txt(corpus_prompts[idx])
        if not cand:
            continue
        if cand.lower() == correct_n.lower():
            continue
        if cand.lower() in seen:
            continue
        out.append(cand)
        seen.add(cand.lower())
        if len(out) >= k:
            break
    return out

def _mutate_prompt(p: str) -> str:
    """
    Pequeñas mutaciones verosímiles (para respaldo).
    Cambia términos comunes en ED o adjetivos sin destruir el sentido general.
    """
    p = _norm_txt(p)
    if not p:
        return ""
    # Cambios típicos en ED
    swaps = [
        (r"\bBFS\b", "DFS"), (r"\bDFS\b", "BFS"),
        (r"\bmin-heap\b", "max-heap"), (r"\bmax-heap\b", "min-heap"),
        (r"\bárbol\b", "grafo"), (r"\bgrafo\b", "árbol"),
        (r"\bcola(s)?\b", "pila"), (r"\bpila(s)?\b", "cola"),
        (r"\bdirigido\b", "no dirigido"), (r"\bno dirigido\b", "dirigido"),
    ]
    for pat, rep in swaps:
        if re.search(pat, p, flags=re.I):
            return re.sub(pat, rep, p, count=1, flags=re.I)

    # Cambios de “tono”
    alt = {
        "claro": "detallado",
        "simple": "complejo",
        "detallado": "simple",
        "esquemático": "realista",
        "realista": "esquemático",
    }
    for k, v in alt.items():
        if re.search(rf"\b{k}\b", p, flags=re.I):
            return re.sub(rf"\b{k}\b", v, p, count=1, flags=re.I)

    # Fallback: reordenar ligeramente una frase final (muy simple)
    return p + " con variaciones menores"

def _pick_concept_distractors(correct: str, k: int = 3) -> list[str]:
    """
    Elige conceptos similares del catálogo (mismo dominio) que no sean el correcto.
    Si hay rapidfuzz, prioriza por similitud; si no, usa SequenceMatcher.
    """
    pool = df_meta.get("concepto", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
    pool = [_norm_txt(x) for x in pool if _norm_txt(x)]
    corr = _norm_txt(correct).lower()
    pool = [x for x in pool if x.lower() != corr]
    if not pool:
        return []

    def _score(x: str) -> float:
        a, b = x.lower(), corr
        if rf_fuzz:
            return float(rf_fuzz.ratio(a, b))
        else:
            return SequenceMatcher(None, a, b).ratio() * 100.0

    scored = sorted(pool, key=_score, reverse=True)
    out, seen = [], set()
    for s in scored:
        if s.lower() in seen: 
            continue
        out.append(s)
        seen.add(s.lower())
        if len(out) >= k:
            break
    return out

# ➜ coloca este bloque junto a tus helpers de query params
def _qp_get():
    try:    return dict(st.query_params)
    except: return st.experimental_get_query_params()

def _qp_set(d: dict):
    try:
        st.query_params.clear()
        for k,v in d.items(): st.query_params[k] = v
    except:
        st.experimental_set_query_params(**d)

def _qp_del(k: str):
    try:
        if k in st.query_params: del st.query_params[k]
    except:
        pass

# ➜ adapta tus 3 helpers:
def _get_sid_from_url():
    q = _qp_get(); sid = q.get("sid")
    return sid if sid else None

def _set_sid_url(sid: str):
    q = _qp_get(); q["sid"] = sid; _qp_set(q)

def _clear_sid_url():
    q = _qp_get(); q.pop("sid", None); _qp_set(q)

# ==== FIN HELPERS ====


# ========= LLM CONFIG & HELPERS (opcional) ====================================
LLM_ENABLE   = os.getenv("LLM_ENABLE", "0") == "1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")      # "openai" | "ollama"
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")    # para openai
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def llm_generate_distractors(correct: str,
                             mode: str = "prompt",   # "prompt" | "concepto"
                             k: int = 3,
                             concept_hint: str = "",
                             topic: str = "") -> list[str]:
    """Devuelve hasta k distractores en español usando LLM. Si LLM está apagado o falla: []."""
    if not LLM_ENABLE:
        return []

    sys_msg = (
        "Eres un asistente para un docente de Estructuras de Datos. "
        "Devuelve opciones INCORRECTAS pero plausibles y en español. "
        "Entrega SOLO una lista de ítems, uno por línea, sin numeración, sin comillas."
    )
    if mode == "prompt":
        user_msg = (
            f"Tema: {topic or 'NA'}\n"
            f"Concepto central: {concept_hint or 'NA'}\n"
            f"Dado este prompt CORRECTO usado para generar la imagen, produce {k} prompts "
            f"alternativos que sean verosímiles pero incorrectos (mismo estilo/longitud):\n\n"
            f"CORRECTO:\n{correct}\n"
            "No repitas el significado del correcto."
        )
    else:
        user_msg = (
            f"Tema: {topic or 'Estructuras de Datos'}\n"
            f"Dado este concepto CORRECTO representado por la imagen, devuelve {k} nombres de conceptos "
            f"del mismo dominio que suelen confundirse pero sean incorrectos:\n\n"
            f"CORRECTO:\n{correct}\n"
            "Una opción por línea. No repitas el correcto."
        )

    try:
        if LLM_PROVIDER.lower() == "openai":
            try:
                from openai import OpenAI
            except Exception:
                return []
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=400,
            )
            text = (res.choices[0].message.content or "")
        elif LLM_PROVIDER.lower() == "ollama":
            import requests, json as _json
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"{sys_msg}\n\n{user_msg}",
                "stream": False,
            }
            r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate", json=payload, timeout=120)
            text = r.json().get("response", "")
        else:
            return []
    except Exception:
        return []

    # Parseo y limpieza
    out = []
    for line in text.splitlines():
        s = re.sub(r"^[\-\*\d\)\.]+\s*", "", line).strip()
        if s:
            out.append(s)

    correct_norm = _norm_txt(correct).lower()
    uniq, seen = [], set()
    for s in out:
        ss = _norm_txt(s)
        if not ss:
            continue
        if ss.lower() == correct_norm:
            continue
        if ss.lower() in seen:
            continue
        uniq.append(ss)
        seen.add(ss.lower())
        if len(uniq) >= k:
            break
    return uniq
# ========= FIN LLM CONFIG & HELPERS ===========================================

def llm_generate_mcq_items(prompt_docente: str,
                           tema: str,
                           n_items: int = 5,
                           usar_catalogo: bool = True,
                           catalog_rows: list[dict] | None = None) -> list[dict]:
    """
    Devuelve una lista de items con esquema:
      {"stem": str, "options": [4], "correct_idx": int, "image_id": str|""}
    Valida y limpia antes de retornar.
    """
    if not LLM_ENABLE:
        return []

    # 1) Prepara contexto opcional de imágenes curadas
    image_pool = []
    if usar_catalogo and catalog_rows:
        # Solo aceptadas del tema
        for r in catalog_rows:
            image_pool.append({"image_id": str(r["image_id"]), "concepto": str(r["concepto"])})

    sys_msg = (
        "Eres un generador de reactivos de opción múltiple para Estructuras de Datos. "
        "Devuelve EXCLUSIVAMENTE un JSON con una lista 'items'. Cada item tiene: "
        "stem (string), options (lista de 4 strings únicas), correct_idx (0..3), image_id (string o vacío). "
        "No incluyas explicaciones ni formato adicional."
    )

    if image_pool:
        pool_text = "\n".join([f"- {p['image_id']}: {p['concepto']}" for p in image_pool])
        user_msg = (
            f"Tema: {tema}\n"
            f"Genera {n_items} preguntas de opción múltiple en español (dificultad media), "
            f"alineadas a competencias del tema. Si puedes, asigna una imagen de esta lista "
            f"coherente con el stem (usa el image_id exacto):\n{pool_text}\n\n"
            f"Instrucción del docente: {prompt_docente}\n"
            "Responde solo con JSON válido: {\"items\": [...]}"
        )
    else:
        user_msg = (
            f"Tema: {tema}\n"
            f"Genera {n_items} preguntas de opción múltiple en español (dificultad media). "
            f"Instrucción del docente: {prompt_docente}\n"
            "Responde solo con JSON válido: {\"items\": [...]}"
        )

    # 2) Llamada al proveedor
    try:
        if LLM_PROVIDER.lower() == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":sys_msg},
                          {"role":"user","content":user_msg}],
                temperature=0.4,
                max_tokens=1200
            )
            text = (res.choices[0].message.content or "")
        else:
            import requests
            payload = {"model": OLLAMA_MODEL, "prompt": f"{sys_msg}\n\n{user_msg}", "stream": False}
            r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate", json=payload, timeout=120)
            text = r.json().get("response","")
    except Exception:
        return []

    # 3) Parseo robusto
    import re, json
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        items = data.get("items", [])
    except Exception:
        return []

    # 4) Validación y limpieza
    out = []
    for it in items:
        stem = _norm_txt(it.get("stem",""))
        opts = _ensure_unique_options(it.get("options", [])[:4])
        if len(opts) != 4 or not stem:
            continue
        try:
            cidx = int(it.get("correct_idx", 0))
        except Exception:
            cidx = 0
        cidx = max(0, min(3, cidx))
        img_id = _norm_txt(it.get("image_id",""))
        if image_pool and img_id and img_id not in [p["image_id"] for p in image_pool]:
            img_id = ""  # si la IA inventa uno, lo vaciamos
        out.append({"stem": stem, "options": opts, "correct_idx": cidx, "image_id": img_id})
    return out


# ==== GENERADOR AUTOMÁTICO DE PREGUNTAS DESDE CATÁLOGO (Bloque B) ====
def auto_generate_questions_from_catalog(
    df_meta: pd.DataFrame,
    mode: str = "concepto",         # "concepto" | "prompt"
    topic_filter: str | None = None,
    state_filter: str = "aceptar",   # "aceptar" | "pendiente" | "ajustar" | "descartar" | "(todos)"
    n_questions: int = 5,
    k_distractors: int = 3,
    strategy: str = "mix",           # "tfidf" | "mutacion" | "llm" | "mix"
    seed: int | None = None
) -> list[dict]:
    """
    Devuelve una lista de items tipo:
        {"stem": str, "options": [str,str,str,str], "correct_idx": int,
         "image_path": str, "image_id": str}
    mode="concepto": opciones son conceptos; stem: '¿Qué concepto representa la imagen?'
    mode="prompt"  : opciones son prompts;  stem: 'Selecciona el prompt correcto...'
    """
    rng = random.Random(seed or 0)

    # --- 1) Filtrar base ---
    view = df_meta.copy()
    # imagen existente en disco
    view["ruta"] = view["ruta"].astype(str).fillna("")
    view = view[view["ruta"].apply(lambda p: bool(_safe_path(p)))]

    if topic_filter and topic_filter not in ("", "(todos)"):
        view = view[view["tema"] == topic_filter]

    if state_filter and state_filter not in ("", "(todos)"):
        view = view[view["estado"] == state_filter]

    if mode == "prompt":
        # solo filas con prompt no vacío
        view = view[view["prompt"].fillna("").astype(str).str.strip() != ""]

    if view.empty:
        return []

    # --- 2) Preparar corpus para distractores ---
    # corpus de prompts (para TF-IDF o LLM en modo prompt)
    corpus_prompts = view["prompt"].fillna("").astype(str).tolist()
    # pool de conceptos (para modo concepto)
    pool_conceptos = view["concepto"].fillna("").astype(str).tolist()

    # --- 3) Sampling de imágenes ---
    # Tomamos un subconjunto de filas al azar (sin replacement) hasta n_questions
    idxs = list(view.index)
    rng.shuffle(idxs)
    idxs = idxs[:min(n_questions, len(idxs))]

    out_items = []

    for idx in idxs:
        row = view.loc[idx]
        img_path = _safe_path(row.get("ruta", ""))
        img_id   = _safe_path(row.get("image_id", ""))
        tema     = _norm_txt(row.get("tema", ""))
        concepto = _norm_txt(row.get("concepto", ""))
        prompt   = _norm_txt(row.get("prompt", ""))

        if mode == "concepto":
            correct = concepto if concepto else "concepto"
            # Distractores por similitud de texto en el pool + LLM opcional
            dists = _pick_concept_distractors(correct, k=k_distractors)
            if LLM_ENABLE and strategy in ("llm", "mix"):
                d_llm = llm_generate_distractors(correct, mode="concepto", k=k_distractors, concept_hint=concepto, topic=tema)
                dists += d_llm
            # fallback si falta: usa conceptos aleatorios del pool
            if len(dists) < k_distractors:
                fallback = [c for c in pool_conceptos if _norm_txt(c).lower() != _norm_txt(correct).lower()]
                rng.shuffle(fallback)
                dists += fallback[: (k_distractors - len(dists))]

            opts = _ensure_unique_options([correct] + dists)[: (1 + k_distractors)]
            # barajar y calcular índice correcto
            rng.shuffle(opts)
            correct_idx = opts.index(correct) if correct in opts else 0

            stem = f"¿Qué concepto representa la imagen mostrada? (Tema: {tema})"

        else:  # mode == "prompt"
            # Si no hay prompt, salta esta fila
            if not prompt:
                continue

            # piscina de distractores
            cand = []

            if strategy in ("tfidf", "mix"):
                # similares por TF-IDF
                cand += _tfidf_distractors(prompt, [p for p in corpus_prompts if _norm_txt(p)], k=k_distractors*2)

            if strategy in ("mutacion", "mix"):
                # varias mutaciones del prompt correcto
                mset = set()
                for _ in range(k_distractors*2):
                    m = _mutate_prompt(prompt)
                    m = _norm_txt(m)
                    if m and m.lower() != _norm_txt(prompt).lower():
                        mset.add(m)
                    if len(mset) >= k_distractors*2:
                        break
                cand += list(mset)

            if LLM_ENABLE and strategy in ("llm", "mix"):
                cand += llm_generate_distractors(prompt, mode="prompt", k=k_distractors, concept_hint=concepto, topic=tema)

            # limpieza + recorte
            cand = [c for c in _ensure_unique_options(cand) if _norm_txt(c).lower() != _norm_txt(prompt).lower()]
            if len(cand) < k_distractors:
                # fallback: mini variaciones
                while len(cand) < k_distractors:
                    cand.append(_mutate_prompt(prompt))
                cand = _ensure_unique_options(cand)

            opts = _ensure_unique_options([prompt] + cand)[: (1 + k_distractors)]
            rng.shuffle(opts)
            correct_idx = opts.index(prompt) if prompt in opts else 0

            stem = f"Selecciona el prompt correcto usado para generar esta imagen (Tema: {tema}, Concepto: {concepto})."

        item = {
            "stem": stem,
            "options": opts,
            "correct_idx": int(correct_idx),
            "image_path": img_path,
            "image_id": img_id
        }
        out_items.append(item)

    return out_items
# ==== FIN Bloque B ====


# Asegura carpetas
BASE = Path("data")
IMG_DIR = BASE / "imagenes"
BASE.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# ===== Config simple (PIN docente y código de clase) =====
CONFIG_PATH = BASE / "config.json"

DEFAULT_CONFIG = {
    "DOCENTE_PIN": "1234",        # cámbialo
    "CLASS_CODE": "EDATA-2025A"   # cámbialo
}

def slugify(s: str, maxlen: int = 24) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^A-Za-z0-9-_]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s[:maxlen] if len(s) > maxlen else s


def _img_b64(path: Path):
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None

def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        return DEFAULT_CONFIG
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
CFG = load_config()

# --- Session Manager (persistencia de login) ---
SESSIONS_PATH = BASE / "sessions.json"
def _sess_load():
    return (json.load(open(SESSIONS_PATH, "r", encoding="utf-8"))
            if SESSIONS_PATH.exists() else {})

def _sess_save(d):
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# --- Query params helpers (SOLO API nueva) ---
def _get_sid_from_url():
    # Devuelve None si no hay ?sid=
    sid = st.query_params.get("sid")
    return sid if sid else None

def _set_sid_url(sid: str):
    st.query_params["sid"] = sid

def _clear_sid_url():
    if "sid" in st.query_params:
        del st.query_params["sid"]
    else:
        st.query_params.clear()


def restore_auth_from_sid():
    """Restaura auth si hay ?sid= en la URL y no estamos logueados."""
    sid = _get_sid_from_url()
    if sid and not st.session_state.get("auth_ok", False):
        sessions = _sess_load()
        info = sessions.get(sid)
        if info and info.get("auth_ok"):
            st.session_state.update({
                "auth_ok": True,
                "auth_role": info.get("auth_role"),
                "auth_name": info.get("auth_name"),
                "class_code": info.get("class_code", ""),
                "sid": sid
            })
            return True
    return False

def persist_current_auth():
    """Guarda la sesión actual en disco y deja ?sid= en la URL."""
    sid = st.session_state.get("sid") or uuid4().hex
    st.session_state["sid"] = sid
    sessions = _sess_load()
    sessions[sid] = {
        "auth_ok": True,
        "auth_role": st.session_state.get("auth_role"),
        "auth_name": st.session_state.get("auth_name"),
        "class_code": st.session_state.get("class_code", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds")
    }
    _sess_save(sessions)
    _set_sid_url(sid)

def clear_persisted_auth():
    """Cierra sesión: borra de disco y limpia la URL."""
    sid = st.session_state.get("sid")
    sessions = _sess_load()
    if sid in sessions:
        del sessions[sid]
        _sess_save(sessions)
    for k in ["auth_ok", "auth_role", "auth_name", "class_code", "sid"]:
        st.session_state.pop(k, None)
    _clear_sid_url()

# ===== Estado de autenticación =====
# Estado base
for k, v in {"auth_ok": False, "auth_role": None, "auth_name": "", "class_code": "", "last_login_err": ""}.items():
    if k not in st.session_state: st.session_state[k] = v

# Restaura si hay ?sid= y no estamos logueados aún
restore_auth_from_sid()

if not st.session_state.auth_ok:
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .hero {
        padding: 28px 32px; border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #0b1324 100%);
        color: #e5e7eb; border: 1px solid #1f2937;
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .hero h1 { margin: 0 0 10px 0; font-size: 2.0rem; line-height: 1.1; }
    .hero p  { margin: 6px 0 0 0; color: #cbd5e1; }
    .card {
        border: 1px solid #1f2937; border-radius: 14px; padding: 16px;
        background: #0b1220; color: #e5e7eb;
    }
    .muted { color:#a1a1aa; font-size:.92rem }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.markdown("""
        <h1>GEN-EDViz</h1>
        <p>Plataforma integral para auditar y utilizar imágenes didácticas generadas con IA en la enseñanza de Estructuras de Datos.
Implementa una metodología completa que articula curación visual, evaluación colaborativa, análisis de aprendizaje, retroalimentación estudiantil y gestión de quizzes pre/post, alineada a los requerimientos de investigación educativa.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown("**Rúbrica 4×4**")
            st.markdown('<div class="card muted">Fidelidad, claridad, pertinencia y equidad/accesibilidad con consenso.</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown("**Métricas**")
            st.markdown('<div class="card muted">Contraste aproximado y densidad de bordes como apoyo, no reemplazo del criterio humano.</div>', unsafe_allow_html=True)
        with cc3:
            st.markdown("**Colaboración**")
            st.markdown('<div class="card muted">Quiz en vivo y pizarra de ideas para retroalimentación y cierre de clase.</div>', unsafe_allow_html=True)

    with c2:
        b64 = _img_b64(HERO_IMG)
        if b64:
            st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:16px;border:1px solid #1f2937;">', unsafe_allow_html=True)
        else:
            st.info("Coloca una imagen en assets/hero_gen-edviz.png para el banner.")

        st.write("")
        st.markdown("### Ingreso")

        tab_doc, tab_est = st.tabs(["Docente", "Estudiante"])

        with tab_doc:
            name_d = st.text_input("Nombre", key="lp_name_doc")
            code_d = st.text_input("Código de clase", value=st.session_state.class_code, key="lp_code_doc")
            pin_d  = st.text_input("PIN Docente", type="password", key="lp_pin")
            if st.button("Entrar como Docente", type="primary", use_container_width=True):
                if not name_d.strip():
                    st.warning("Escribe tu nombre.")
                elif pin_d.strip() != CFG.get("DOCENTE_PIN", ""):
                    st.error("PIN incorrecto.")
                else:
                    st.session_state.auth_ok = True
                    st.session_state.auth_role = "Docente"
                    st.session_state.auth_name = name_d.strip()
                    st.session_state.class_code = code_d.strip()
                    persist_current_auth()
                    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

        with tab_est:
            name_e = st.text_input("Nombre", key="lp_name_est")
            code_e = st.text_input("Código de clase", value=st.session_state.class_code, key="lp_code_est")
            if st.button("Entrar como Estudiante", type="primary", use_container_width=True):
                if not name_e.strip():
                    st.warning("Escribe tu nombre.")
                elif code_e.strip() != CFG.get("CLASS_CODE", ""):
                    st.error("Código de clase inválido.")
                else:
                    st.session_state.auth_ok = True
                    st.session_state.auth_role = "Estudiante"
                    st.session_state.auth_name = name_e.strip()
                    st.session_state.class_code = code_e.strip()
                    persist_current_auth()
                    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

    st.stop()                # <- corta la ejecución aquí (no sigue al menú)

is_docente = (st.session_state.auth_role == "Docente")
user_name = st.session_state.auth_name

# ===== Menú por rol =====
DOCENTE_SECTIONS = [
    "Subir y catalogar", "Galería", "Evaluar (rúbrica 4x4)",
    "Resumen y reportes", "Resultados de aprendizaje",
    "Quiz manual", "Encuesta de percepción",
    "Retroalimentación compartida", "Ajustes"
]

ESTUDIANTE_SECTIONS = [
    "Galería", "Evaluar (rúbrica 4x4)",
    "Quiz manual",
    "Retroalimentación compartida", "Encuesta de percepción","Ajustes"
]





st.sidebar.title("GEN-EDViz Monitor")
st.sidebar.caption(f"{user_name} · {st.session_state.auth_role}")
secciones = DOCENTE_SECTIONS if is_docente else ESTUDIANTE_SECTIONS
seccion = st.sidebar.radio("Secciones", secciones, index=0)

# Cerrar sesión (opcional)
if st.sidebar.button("Cerrar sesión"):
    clear_persisted_auth()
    (st.rerun if hasattr(st,"rerun") else st.experimental_rerun)()


# Rutas UNIFICADAS con Path (no volver a redefinir IMG_DIR como str)
DATA_DIR = BASE                      # Path("data")
IMG_DIR  = IMG_DIR                   # ya es BASE / "imagenes"
CSV_PATH = DATA_DIR / "evaluaciones.csv"
META_PATH = DATA_DIR / "catalogo.csv"
PERCEP_PATH = DATA_DIR / "percepcion.csv"

# Asegura directorios
DATA_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# --- LOG DE EVENTOS (events.csv) ---
EVENTS = DATA_DIR / "events.csv"
if not EVENTS.exists():
    EVENTS.write_text("ts,user,role,action,ref_id,extra_json\n", encoding="utf-8")

def log_event(action, ref_id: str = "", extra: dict | None = None):
    # convierte extra a JSON seguro (por si hay Paths o tipos numpy)
    def _jsonable(v):
        import numpy as np
        from pathlib import Path
        if isinstance(v, Path): return v.as_posix()
        if isinstance(v, (np.integer, np.floating)): return v.item()
        try:
            json.dumps(v, ensure_ascii=False); return v
        except Exception:
            return str(v)
    extra = {k: _jsonable(v) for k, v in (extra or {}).items()}
    row = ",".join([
        datetime.now().isoformat(timespec="seconds"),
        st.session_state.get("auth_name",""),
        st.session_state.get("auth_role",""),
        action,
        ref_id,
        json.dumps(extra, ensure_ascii=False)
    ]) + "\n"
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(row)


# Directorio para retroalimentación colaborativa
FEED_DIR = Path("data") / "feedback"
FEED_DIR.mkdir(parents=True, exist_ok=True)

# Directorio para quizzes (futuro)
QUIZ_DIR = Path("data") / "quizzes"
QUIZ_DIR.mkdir(parents=True, exist_ok=True)

def jload(p: Path, default=None):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def jsave(p: Path, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# === QUIZ: estado por usuario (autodirigido) ===
def _user_state_path(qdir: Path) -> Path:
    return qdir / "user_state.json"

def _load_user_state(qdir: Path) -> dict:
    stobj = jload(_user_state_path(qdir), default=None)
    if not stobj:
        stobj = {"users": {}}  # user -> {idx, finished, started_at, finished_at}
    return stobj

def _save_user_state(qdir: Path, stobj: dict):
    jsave(_user_state_path(qdir), stobj)

def _get_or_init_user_state(qdir: Path, user: str) -> dict:
    stobj = _load_user_state(qdir)
    if user not in stobj["users"]:
        stobj["users"][user] = {
            "idx": 0,
            "finished": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None
        }
        _save_user_state(qdir, stobj)
    return stobj["users"][user]

def _update_user_state(qdir: Path, user: str, **patch):
    stobj = _load_user_state(qdir)
    if user not in stobj["users"]:
        stobj["users"][user] = {
            "idx": 0,
            "finished": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None
        }
    stobj["users"][user].update(patch)
    _save_user_state(qdir, stobj)

def _upsert_answer(ans_path: Path, quiz_id: str, user: str, q_idx: int, choice_idx: int, correct: int):
    cols = ["quiz_id","user","q_idx","choice_idx","correct","timestamp"]
    df = pd.read_csv(ans_path) if ans_path.exists() and ans_path.stat().st_size > 0 else pd.DataFrame(columns=cols)
    mask = (df["quiz_id"] == quiz_id) & (df["user"] == user) & (df["q_idx"] == q_idx)
    if mask.any():
        df.loc[mask, ["choice_idx","correct","timestamp"]] = [choice_idx, correct, datetime.now().isoformat(timespec="seconds")]
    else:
        df = pd.concat([df, pd.DataFrame([{
            "quiz_id": quiz_id, "user": user, "q_idx": q_idx,
            "choice_idx": choice_idx, "correct": correct,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }])], ignore_index=True)
    df.to_csv(ans_path, index=False)
    return df

def _recompute_user_score(qdir: Path, quiz_id: str, user: str) -> int:
    ans_path = qdir / "answers.csv"
    score = 0
    if ans_path.exists() and ans_path.stat().st_size > 0:
        tmp = pd.read_csv(ans_path)
        score = int(tmp[(tmp["quiz_id"] == quiz_id) & (tmp["user"] == user)]["correct"].fillna(0).astype(int).sum())
    roster_path = qdir / "roster.csv"
    roster_df = pd.read_csv(roster_path) if roster_path.exists() and roster_path.stat().st_size > 0 else pd.DataFrame(columns=["user","joined_at","score"])
    if user in roster_df["user"].tolist():
        roster_df.loc[roster_df["user"] == user, "score"] = score
    else:
        roster_df = pd.concat([roster_df, pd.DataFrame([{
            "user": user, "joined_at": datetime.now().isoformat(timespec="seconds"), "score": score
        }])], ignore_index=True)
    roster_df.to_csv(roster_path, index=False)
    return score

# ==== HELPERS PARA QUIZZES (para análisis pre/post) ====

def list_quizzes():
    """
    Devuelve una lista de quizzes encontrados en QUIZ_DIR, cada uno como:
    {"quiz_id", "title", "quiz_role", "path"}
    """
    quizzes = []
    if QUIZ_DIR.exists():
        for sub in QUIZ_DIR.iterdir():
            if sub.is_dir():
                q = jload(sub / "quiz.json", default=None)
                if q:
                    quizzes.append({
                        "quiz_id": q.get("quiz_id", sub.name),
                        "title": q.get("title", q.get("quiz_id", sub.name)),
                        "quiz_role": q.get("quiz_role", "other"),
                        "path": sub
                    })
    return quizzes

def load_roster_for_quiz(qinfo: dict) -> pd.DataFrame:
    """
    Carga el roster (usuario + score) para un quiz.
    """
    qdir = qinfo["path"]
    roster_path = qdir / "roster.csv"
    if roster_path.exists() and roster_path.stat().st_size > 0:
        df = pd.read_csv(roster_path)
        if "user" not in df.columns:
            df["user"] = ""
        if "score" not in df.columns:
            df["score"] = 0
        df["user"] = df["user"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        return df[["user","score"]]
    else:
        # si no hay roster, devolvemos vacío
        return pd.DataFrame(columns=["user","score"])


# -------------------------------------------------------
# Utilidades
# -------------------------------------------------------
CRITERIA = [
    ("fidelidad", "¿Representa correctamente el concepto (estructuras, relaciones, pasos)?"),
    ("claridad", "¿Se lee y entiende bien (legibilidad, señalización, foco visual)?"),
    ("pertinencia", "¿Está alineada al objetivo/competencia del tema del curso?"),
    ("equidad", "¿Cumple criterios de accesibilidad y neutralidad cultural/estética?")
]

# Descriptores de nivel 1–4 por criterio (para mostrar al evaluador)
LEVEL_DESC = {
    "fidelidad": {
        1: "Errores graves de concepto (estructura mal representada o paso incorrecto).",
        2: "Errores/modificaciones moderadas; requiere muchas aclaraciones del docente.",
        3: "Correcta en lo esencial, con detalles mejorables.",
        4: "Representación precisa y coherente con el concepto."
    },
    "claridad": {
        1: "Muy confusa o recargada; difícil de leer.",
        2: "Elementos legibles pero con desorden o ruido visual.",
        3: "Generalmente clara, con algunos elementos mejorables.",
        4: "Muy clara, jerarquía visual evidente y foco bien definido."
    },
    "pertinencia": {
        1: "No se ajusta al objetivo del tema o induce a malentendidos.",
        2: "Parcialmente alineada; mezcla información irrelevante.",
        3: "Pertinente al objetivo, con detalles adicionales no críticos.",
        4: "Directamente alineada al objetivo y al resultado de aprendizaje."
    },
    "equidad": {
        1: "Contiene sesgos evidentes o problemas serios de accesibilidad.",
        2: "Muestra posibles sesgos o baja accesibilidad (contrastes, texto, etc.).",
        3: "En general neutra y relativamente accesible, con mejoras posibles.",
        4: "Respetuosa, diversa y con buena accesibilidad visual."
    }
}

# Opciones para la decisión global de uso
DECISION_USO = {
    "usar_sin_cambios": "Sí, la usaría tal cual en clase.",
    "usar_con_ajustes": "La usaría con ajustes o explicaciones adicionales.",
    "no_usar_en_clase": "No la usaría en un contexto de enseñanza real."
}

SEVERIDAD_GLOBAL = {
    "ninguno": "Sin problemas relevantes.",
    "menores": "Problemas menores, poco visibles.",
    "moderados": "Problemas moderados, visibles pero manejables.",
    "graves": "Problemas graves que afectan su uso pedagógico."
}

def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """
    Calcula alfa de Cronbach a partir de un DataFrame de ítems Likert
    (filas = personas, columnas = ítems numéricos).
    """
    if items_df is None or items_df.empty:
        return np.nan

    # Eliminar filas totalmente vacías
    items = items_df.dropna(how="all")
    if items.empty or items.shape[1] < 2:
        return np.nan

    k = items.shape[1]  # número de ítems
    var_items = items.var(axis=0, ddof=1)
    var_total = items.sum(axis=1).var(ddof=1)

    if var_total <= 0:
        return np.nan

    return (k / (k - 1.0)) * (1.0 - var_items.sum() / var_total)


def load_csv(path, cols=None):
    path = str(path)  # por si viene como Path
    if os.path.exists(path):
        df = pd.read_csv(path)
        if cols:
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
        return df
    return pd.DataFrame(columns=cols or [])

def save_csv(df, path):
    df.to_csv(str(path), index=False)


# -------------------------------------------------------
# Datos
# -------------------------------------------------------
evaluaciones_cols = [
    "image_id","tema","concepto","rater","criterio","puntaje","comentario",
    "decision_uso","severidad","comentario_global","timestamp"
]
df_eval = load_csv(CSV_PATH, evaluaciones_cols)


catalogo_cols = [
    "image_id","tema","concepto","herramienta","prompt","version","fecha",
    "autor","alt_text","ruta","wcag_ratio","edge_density","estado"
]
df_meta = load_csv(META_PATH, catalogo_cols)

percepcion_cols = [
    "user","role","tema",
    "claridad","utilidad","aprendizaje","motivacion","sesgos",
    "comentario","timestamp"
]
df_percep = load_csv(PERCEP_PATH, percepcion_cols)


# -------------------------------------------------------
# Subir y catalogar
# -------------------------------------------------------
if seccion == "Subir y catalogar":
    if not is_docente:
        st.warning("Solo el Docente puede subir y catalogar imágenes.")
        st.stop()
    st.header("Subir y catalogar imágenes")
    tema = st.text_input("Tema (p. ej., arboles, grafos, pilas)", "")
    concepto = st.text_input("Concepto (p. ej., inorden, BFS, push/pop)", "")
    herramienta = st.selectbox("Herramienta", ["Gemini", "DALL·E", "Stable Diffusion", "Midjourney"])
    prompt = st.text_area("Prompt usado")
    alt_text = st.text_area("Texto alternativo (accesibilidad)")

    archivo = st.file_uploader("Cargar imagen (.png/.jpg)", type=["png","jpg","jpeg"])
    autor = st.text_input("Autor/Equipo", "")
    version = st.text_input("Versión", "v1")

    # --- Guardar en catálogo (todo dentro del botón) ---
    if st.button(
        "Guardar en catálogo",
        type="primary",
        disabled=(archivo is None or tema.strip()=="" or concepto.strip()=="")
    ):
        # 1) Bytes e ID
        img_bytes = archivo.read()
        image_id = f"IMG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # 2) Nombre corto y seguro
        tema_s = slugify(tema)
        herr_s = slugify(herramienta)
        hoy_s  = date.today().isoformat()
        uid8   = str(uuid4())[:8]
        filename = f"IMG-{hoy_s}-{tema_s}-{herr_s}-{uid8}.png"
        ruta = IMG_DIR / filename  # Path

        # 3) Guardar binario
        with open(ruta, "wb") as f:
            f.write(img_bytes)

        # 4) Métricas objetivas
        img = Image.open(io.BytesIO(img_bytes))
        ratio = wcag_like_contrast_ratio(img)
        edens = edge_density(img)

        # 5) Metadatos
        row = {
            "image_id": image_id,
            "tema": tema,
            "concepto": concepto,
            "herramienta": herramienta,
            "prompt": prompt,
            "version": version,
            "fecha": str(datetime.now().date()),
            "autor": user_name,
            "alt_text": alt_text,
            "ruta": str(ruta.as_posix()),
            "wcag_ratio": round(ratio, 2),
            "edge_density": round(edens, 3),
            "estado": "pendiente"
        }

        # 6) Persistir
        df_meta = pd.concat([df_meta, pd.DataFrame([row])], ignore_index=True)
        save_csv(df_meta, META_PATH)

        st.success(
            f"Guardado {image_id} ({filename}). "
            f"Contraste≈{row['wcag_ratio']} | Edge dens={row['edge_density']}."
        )
        log_event("catalog.save", ref_id=image_id, extra=row)

    st.divider()

    # ------------ CARGA MASIVA (ZIP) ------------
    st.subheader("Carga masiva (ZIP)")
    zip_file = st.file_uploader("Subir ZIP con imágenes", type=["zip"], key="zip_bulk")
    auto_tema = st.text_input("Tema por defecto para ZIP (opcional)", "")
    auto_herr = st.selectbox("Herramienta por defecto para ZIP (opcional)", ["", "Gemini", "DALL·E", "Stable Diffusion", "Midjourney"])
    auto_version = st.text_input("Versión por defecto", "v1")

    st.caption("Sugerencia de nombre por archivo: tema_concepto_herramienta_version.png. Si falta algo, se usa el valor por defecto.")
    added = 0
    if st.button("Procesar ZIP", disabled=zip_file is None):
        try:
            tmp_zip = BASE / f"tmp_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
            with open(tmp_zip, "wb") as f:
                f.write(zip_file.read())
            if not is_zipfile(tmp_zip):
                st.error("El archivo no es un ZIP válido.")
            else:
                added = 0
                with ZipFile(tmp_zip, "r") as z:
                    for nm in z.namelist():
                        low = nm.lower()
                        if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
                            raw = z.read(nm)
                            # inferencia básica por nombre
                            base = os.path.basename(nm)
                            name_noext = os.path.splitext(base)[0]
                            parts = [p for p in re.split(r"[_-]+", name_noext) if p]
                            tema_i = auto_tema or (parts[0] if len(parts) > 0 else "tema")
                            concepto_i = (parts[1] if len(parts) > 1 else "concepto")
                            herr_i = auto_herr or (parts[2] if len(parts) > 2 else "Gemini")
                            ver_i = auto_version or (parts[3] if len(parts) > 3 else "v1")

                            tema_s = slugify(tema_i)
                            herr_s = slugify(herr_i)
                            uid8 = str(uuid4())[:8]
                            filename = f"IMG-{date.today().isoformat()}-{tema_s}-{herr_s}-{uid8}.png"
                            dest = Path(IMG_DIR) / filename
                            with open(dest, "wb") as f:
                                f.write(raw)

                            try:
                                img = Image.open(io.BytesIO(raw))
                                ratio = wcag_like_contrast_ratio(img)
                                edens = edge_density(img)
                            except Exception:
                                ratio, edens = np.nan, np.nan

                            row = {
                                "image_id": f"IMG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uid8}",
                                "tema": tema_i,
                                "concepto": concepto_i,
                                "herramienta": herr_i,
                                "prompt": "",
                                "version": ver_i,
                                "fecha": str(datetime.now().date()),
                                "autor": st.session_state.get("auth_name", ""),
                                "alt_text": "",
                                "ruta": dest.as_posix(),
                                "wcag_ratio": round(ratio, 2) if not pd.isna(ratio) else "",
                                "edge_density": round(edens, 3) if not pd.isna(edens) else "",
                                "estado": "pendiente"
                            }
                            df_meta = pd.concat([df_meta, pd.DataFrame([row])], ignore_index=True)
                            added += 1
                save_csv(df_meta, META_PATH)
                st.success(f"ZIP procesado. Imágenes añadidas: {added}")
                log_event("catalog.zip_done", extra={"added": added})
            try:
                tmp_zip.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error procesando ZIP: {e}")

# -------------------------------------------------------
# Galería con filtros, buscador y paginación
# -------------------------------------------------------
elif seccion == "Galería":
    st.header("Galería de imágenes")
    if df_meta.empty:
        st.info("No hay imágenes en el catálogo.")
    else:
        # --------- Filtros básicos ---------
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            f_tema = st.selectbox(
                "Tema",
                ["(todos)"] + sorted(df_meta["tema"].dropna().astype(str).unique().tolist())
            )
        with colf2:
            f_herr = st.selectbox(
                "Herramienta",
                ["(todas)"] + sorted(df_meta["herramienta"].dropna().astype(str).unique().tolist())
            )
        with colf3:
            f_estado = st.selectbox(
                "Estado",
                ["(todos)", "pendiente", "aceptar", "ajustar", "descartar"]
            )

        view = df_meta.copy()

        if f_tema != "(todos)":
            view = view[view["tema"] == f_tema]
        if f_herr != "(todas)":
            view = view[view["herramienta"] == f_herr]
        if f_estado != "(todos)":
            view = view[view["estado"] == f_estado]

        # --------- Buscador por tema (texto libre) ---------
        st.markdown("### Búsqueda")
        search_tema = st.text_input(
            "Buscar por tema (coincidencia parcial, sin distinción de mayúsculas/minúsculas)",
            value="",
            key="gal_search_tema"
        ).strip()

        if search_tema:
            view = view[
                view["tema"]
                .fillna("")
                .astype(str)
                .str.contains(search_tema, case=False, na=False)
            ]

        if view.empty:
            st.info("No hay imágenes que coincidan con los filtros / búsqueda.")
        else:
            # Orden y reset de índice para que la paginación sea limpia
            view = view.sort_values(["tema", "concepto", "fecha"], ascending=[True, True, True])
            view = view.reset_index(drop=True)

            # --------- Paginación ---------
            PAGE_SIZE = 6  # 6 imágenes por página (3 x 2)
            total_items = len(view)
            total_pages = max(1, math.ceil(total_items / PAGE_SIZE))

            # Estado de página actual en sesión
            if "gal_page" not in st.session_state:
                st.session_state.gal_page = 1

            page = st.session_state.gal_page
            if page < 1:
                page = 1
            if page > total_pages:
                page = total_pages
            st.session_state.gal_page = page

            start = (page - 1) * PAGE_SIZE
            end = start + PAGE_SIZE
            page_df = view.iloc[start:end]

            # Controles del paginador
            colp1, colp2, colp3 = st.columns([1, 2, 1])
            with colp1:
                btn_prev = st.button(
                    "◀ Anterior",
                    disabled=(page <= 1),
                    key="gal_btn_prev"
                )
            with colp2:
                st.markdown(
                    f"<div style='text-align:center'>Página <b>{page}</b> / {total_pages} · "
                    f"{total_items} imágenes</div>",
                    unsafe_allow_html=True
                )
            with colp3:
                btn_next = st.button(
                    "Siguiente ▶",
                    disabled=(page >= total_pages),
                    key="gal_btn_next"
                )

            if btn_prev:
                st.session_state.gal_page = page - 1
                (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

            if btn_next:
                st.session_state.gal_page = page + 1
                (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

            st.write("")  # pequeño espacio

            # --------- Grid de miniaturas (3 columnas x 2 filas máx) ---------
            ncols = 3
            rows = [page_df.iloc[i:i + ncols] for i in range(0, len(page_df), ncols)]

            for r in rows:
                cols = st.columns(ncols)
                for c, (_, row) in zip(cols, r.iterrows()):
                    with c:
                        st.image(row["ruta"], use_column_width=True)
                        st.caption(
                            f"{row['tema']} / {row['concepto']}\n"
                            f"{row['herramienta']} · {row['version']} · {row['estado']}"
                        )

# -------------------------------------------------------
# Evaluación (rúbrica 4x4)
# -------------------------------------------------------
elif seccion == "Evaluar (rúbrica 4x4)":
    st.header("Evaluación con rúbrica 4x4")
    if df_meta.empty:
        st.info("No hay imágenes en catálogo.")
    else:
        # ====== lista de temas SIN duplicados ======
        temas_unicos = sorted(
            df_meta["tema"].dropna().astype(str).unique().tolist()
        )
        if not temas_unicos:
            st.info("No hay temas registrados en el catálogo.")
            st.stop()

        col1, col2 = st.columns([2, 1])

        # ---------------- COLUMNA IZQUIERDA: selección y navegación de imágenes ----------------
        with col1:
            # select solo con temas únicos
            tema_sel = st.selectbox(
                "Selecciona imagen por tema",
                temas_unicos,
                key="eval_tema_sel"
            )

            # --- estado de navegación por tema ---
            if "eval_tema_prev" not in st.session_state:
                st.session_state.eval_tema_prev = tema_sel
            if "eval_img_idx" not in st.session_state:
                st.session_state.eval_img_idx = 0

            # si el usuario cambia de tema, reiniciamos al primer índice
            if tema_sel != st.session_state.eval_tema_prev:
                st.session_state.eval_tema_prev = tema_sel
                st.session_state.eval_img_idx = 0

            # subconjunto de imágenes del tema elegido
            subset = df_meta[df_meta["tema"] == tema_sel].reset_index(drop=True)
            if subset.empty:
                st.warning("No hay imágenes para ese tema.")
                st.stop()

            n_imgs = len(subset)

            # índice actual dentro del tema
            idx = st.session_state.eval_img_idx
            if idx < 0:
                idx = 0
            if idx > n_imgs - 1:
                idx = n_imgs - 1
            st.session_state.eval_img_idx = idx

            # fila actual (imagen seleccionada)
            meta = subset.iloc[idx]

            # --- layout: flecha izquierda · imagen · flecha derecha ---
            c_left, c_center, c_right = st.columns([1, 4, 1])

            with c_left:
                prev_btn = st.button(
                    "◀",
                    key="eval_prev_img",
                    disabled=(idx == 0)
                )

            with c_center:
                st.image(
                    meta["ruta"],
                    caption=f"{meta['tema']} / {meta['concepto']} "
                            f"[{meta['herramienta']}] (v={meta['version']})",
                    use_column_width=True
                )
                st.caption(f"Imagen {idx+1} de {n_imgs} para el tema «{tema_sel}»")

            with c_right:
                next_btn = st.button(
                    "▶",
                    key="eval_next_img",
                    disabled=(idx >= n_imgs - 1)
                )

            # actualizar índice al pulsar flechas
            if prev_btn:
                st.session_state.eval_img_idx = idx - 1
                rerun()
            if next_btn:
                st.session_state.eval_img_idx = idx + 1
                rerun()

            # info extra debajo de la imagen
            st.caption(
                f"Contraste≈{meta['wcag_ratio']} | Edge dens={meta['edge_density']} | "
                f"Alt: {str(meta['alt_text'])[:100]}..."
            )

        # ---------------- COLUMNA DERECHA: formulario de rúbrica ----------------
        with col2:
            # Nombre del evaluador (por defecto toma el de la sesión)
            rater = st.text_input(
                "Nombre del evaluador (docente/estudiante)",
                value=user_name or "",
                key="eval_rater"
            )

            comentarios = {}
            puntajes = {}

            st.markdown("### Rúbrica 4×4")

            with st.form("form_rubrica"):
                # ---------- CRITERIOS UNO POR UNO ----------
                for key, q in CRITERIA:
                    st.markdown(f"**{key.capitalize()}** — {q}")

                    # Descriptores de niveles 1–4
                    descs = LEVEL_DESC.get(key, {})
                    with st.expander("Ver niveles (1–4)", expanded=True):
                        for nivel in range(1, 5):
                            txt = descs.get(nivel, "")
                            if txt:
                                st.markdown(f"- **{nivel}**: {txt}")

                    # Slider 1–4
                    puntajes[key] = st.slider(
                        f"Puntaje para {key}",
                        min_value=1,
                        max_value=4,
                        value=3,
                        step=1,
                        key=f"sl_{key}"
                    )

                    # Texto corto del nivel elegido
                    if descs.get(puntajes[key]):
                        st.caption(
                            f"Nivel seleccionado ({puntajes[key]}): "
                            f"{descs[puntajes[key]]}"
                        )

                    comentarios[key] = st.text_area(
                        f"Comentario específico sobre {key}",
                        height=60,
                        key=f"tx_{key}"
                    )

                    st.markdown("---")

                # ---------- SÍNTESIS GLOBAL ----------
                st.markdown("### Síntesis global de la imagen")

                decision_uso_key = st.radio(
                    "¿Usarías esta imagen en clase?",
                    options=list(DECISION_USO.keys()),
                    format_func=lambda k: DECISION_USO[k],
                    index=1,  # por defecto: "usar_con_ajustes"
                    key="eval_decision_uso"
                )

                severidad_key = st.radio(
                    "Severidad global de problemas detectados",
                    options=list(SEVERIDAD_GLOBAL.keys()),
                    format_func=lambda k: SEVERIDAD_GLOBAL[k],
                    index=2,  # por defecto: "moderados"
                    key="eval_severidad"
                )

                comentario_global = st.text_area(
                    "Comentario global (síntesis, ajustes recomendados)",
                    height=80,
                    key="eval_comentario_global"
                )

                enviado = st.form_submit_button(
                    "Enviar evaluación",
                    disabled=(rater.strip() == "")
                )

            # ---------- GUARDAR EN CSV ----------
            if enviado:
                now = datetime.now().isoformat(timespec="seconds")
                new_rows = []

                for key, _ in CRITERIA:
                    new_rows.append({
                        "image_id": meta["image_id"],
                        "tema": meta["tema"],
                        "concepto": meta["concepto"],
                        "rater": rater.strip() or user_name,
                        "criterio": key,
                        "puntaje": puntajes[key],
                        "comentario": comentarios[key],
                        "decision_uso": decision_uso_key,
                        "severidad": severidad_key,
                        "comentario_global": comentario_global,
                        "timestamp": now
                    })

                df_eval = pd.concat(
                    [df_eval, pd.DataFrame(new_rows)],
                    ignore_index=True
                )
                save_csv(df_eval, CSV_PATH)

                log_event(
                    "rubric.submit",
                    ref_id=meta["image_id"],
                    extra={
                        "scores": puntajes,
                        "tema": meta["tema"],
                        "decision_uso": decision_uso_key,
                        "severidad": severidad_key
                    }
                )

                st.success("Evaluación registrada.")

        # ------- resumen de evaluaciones de ESA imagen (igual que tenías) -------
        df_img = df_eval[df_eval["image_id"] == meta["image_id"]]
        if df_img.empty:
            st.info("Aún no hay evaluaciones.")
        else:
            # (todo tu bloque de resumen / kappa / cambiar estado se mantiene igual)
            piv = df_img.pivot_table(index="criterio", values="puntaje", aggfunc="mean")
            st.write(piv)
            min_crit = piv["puntaje"].min()
            avg_all = piv["puntaje"].mean()
            sugerencia = "aceptar"
            if min_crit < 2:
                sugerencia = "descartar"
            elif min_crit < 3:
                sugerencia = "ajustar"
            st.success(
                f"Sugerencia: {sugerencia} (min criterio={min_crit:.2f}, "
                f"promedio={avg_all:.2f})"
            )

            kappas = {}
            for key, _ in CRITERIA:
                sub = df_img[df_img["criterio"] == key]
                wide = sub.pivot_table(index="rater", columns="timestamp",
                                       values="puntaje").T
                k_pair = avg_cohen_kappa_pairwise(wide) if wide.shape[1] >= 2 else np.nan
                kappas[key] = k_pair

            fk = fleiss_kappa_from_long(
                df_img, "rater", "timestamp", "puntaje"
            )

            c1, c2 = st.columns(2)
            with c1:
                st.write(
                    "Cohen κ promedio por criterio:",
                    pd.DataFrame(
                        {
                            "criterio": [k for k, _ in CRITERIA],
                            "kappa": [kappas[k] for k, _ in CRITERIA],
                        }
                    ),
                )
            with c2:
                st.write(
                    f"Fleiss κ global (todas las puntuaciones): {fk:.3f}"
                    if not pd.isna(fk)
                    else "Fleiss κ: insuficiente para calcular"
                )

            if {"decision_uso","severidad","comentario_global"}.issubset(df_img.columns):
                st.subheader("Decisiones globales registradas")
                resumen_global = (
                    df_img[["rater","decision_uso","severidad","comentario_global"]]
                    .drop_duplicates()
                )
                st.dataframe(resumen_global, use_container_width=True)

            if is_docente:
                nuevo_estado = st.selectbox(
                    "Actualizar estado en catálogo",
                    ["pendiente","aceptar","ajustar","descartar"],
                    index=["pendiente","aceptar","ajustar","descartar"].index(
                        meta["estado"]
                    )
                    if meta["estado"] in ["pendiente","aceptar","ajustar","descartar"]
                    else 0
                )
                if st.button("Guardar estado"):
                    df_meta.loc[
                        df_meta["image_id"] == meta["image_id"], "estado"
                    ] = nuevo_estado
                    save_csv(df_meta, META_PATH)
                    st.success("Estado actualizado.")
                    log_event(
                        "catalog.update_state",
                        ref_id=meta["image_id"],
                        extra={"estado": nuevo_estado},
                    )
            else:
                st.info("Solo el Docente puede cambiar el estado en el catálogo.")



# -------------------------------------------------------
# Resumen y reportes
# -------------------------------------------------------
elif seccion == "Resumen y reportes":
    st.header("Reportes por tema, herramienta y estado")
    if df_meta.empty:
        st.info("Sin datos.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imágenes en catálogo", len(df_meta))
        with col2:
            st.metric("Pendientes", int((df_meta["estado"]=="pendiente").sum()))
        with col3:
            st.metric("Aprobadas", int((df_meta["estado"]=="aceptar").sum()))

        st.subheader("Promedios de rúbrica por imagen")
        if df_eval.empty:
            st.info("Aún no hay evaluaciones.")
        else:
            piv = df_eval.pivot_table(index="image_id", columns="criterio", values="puntaje", aggfunc="mean").reset_index()
            merged = df_meta.merge(piv, on="image_id", how="left")
            st.dataframe(merged, use_container_width=True)

        st.subheader("Exportar CSV consolidado")
        if is_docente:
            if st.button("Exportar archivo"):
                merged = df_eval.merge(df_meta, on=["image_id","tema","concepto"], how="right")
                merged.to_csv(str((DATA_DIR / "consolidado_export.csv")), index=False)
                st.success("Generado data/consolidado_export.csv")
                st.subheader("Exportar banco curado (ZIP)")

    # Usa un key para evitar conflictos con otros botones
    crear_zip = st.button("Crear ZIP con aceptadas", key="btn_zip_aceptadas")

    if crear_zip:
        # 1) Filtrar aceptadas dentro del mismo bloque
        aceptadas = df_meta[df_meta["estado"] == "aceptar"].copy()

        if aceptadas.empty:
            st.info("No hay imágenes aceptadas.")
        else:
            # 2) Crear ZIP con metadatos + imágenes
            out_zip = BASE / f"banco_curado_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
            with ZipFile(out_zip, "w") as z:
                # Añadir CSV de metadatos (solo aceptadas)
                meta_csv_path = BASE / "metadata_curado.csv"
                aceptadas.to_csv(meta_csv_path, index=False, encoding="utf-8")
                z.write(meta_csv_path, arcname="metadata_curado.csv")
                try:
                    meta_csv_path.unlink(missing_ok=True)
                except Exception:
                    pass

                # Añadir imágenes
                for _, row in aceptadas.iterrows():
                    p = Path(row["ruta"])
                    if p.exists():
                        z.write(p, arcname=f"imagenes/{p.name}")

            st.success(f"ZIP generado: {out_zip.as_posix()}")

    else:
        st.info("Exportaciones disponibles solo para Docente.")


# -------------------------------------------------------
# RESULTADOS DE APRENDIZAJE (comparar pretest / postest)
# -------------------------------------------------------
elif seccion == "Resultados de aprendizaje":
    st.header("Resultados de aprendizaje (pretest / postest)")

    if not is_docente:
        st.info("Solo el Docente puede ver este análisis.")
        st.stop()

    quizzes = list_quizzes()
    if not quizzes:
        st.info("No se encontraron quizzes en data/quizzes.")
        st.stop()

    # separar por rol
    pre_quizzes  = [q for q in quizzes if q["quiz_role"] == "pre"]
    post_quizzes = [q for q in quizzes if q["quiz_role"] == "post"]

    if not pre_quizzes:
        st.warning("Aún no has creado quizzes marcados como pretest (quiz_role='pre').")
        st.stop()
    if not post_quizzes:
        st.warning("Aún no has creado quizzes marcados como postest (quiz_role='post').")
        st.stop()

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        pre_sel = st.selectbox(
            "Selecciona el quiz de **pretest**",
            options=pre_quizzes,
            format_func=lambda q: f"{q['quiz_id']} · {q['title']}"
        )
    with col_sel2:
        post_sel = st.selectbox(
            "Selecciona el quiz de **postest**",
            options=post_quizzes,
            format_func=lambda q: f"{q['quiz_id']} · {q['title']}"
        )

    # cargar datos de cada quiz
    df_pre  = load_roster_for_quiz(pre_sel)
    df_post = load_roster_for_quiz(post_sel)

    if df_pre.empty:
        st.warning("El pretest seleccionado no tiene registros de puntuación (roster.csv vacío).")
        st.stop()
    if df_post.empty:
        st.warning("El postest seleccionado no tiene registros de puntuación (roster.csv vacío).")
        st.stop()

    # unir por usuario
    merged = df_pre.merge(
        df_post,
        on="user",
        how="inner",
        suffixes=("_pre", "_post")
    )

    if merged.empty:
        st.warning("No hay estudiantes que aparezcan en ambos quizzes con el mismo nombre de usuario.")
        st.stop()

    # calcular delta
    merged["delta"] = merged["score_post"] - merged["score_pre"]

    # opcional: porcentaje sobre el máximo
    q_pre_json  = jload(pre_sel["path"] / "quiz.json", default={})
    q_post_json = jload(post_sel["path"] / "quiz.json", default={})
    max_pre  = len(q_pre_json.get("questions", [])) or 1
    max_post = len(q_post_json.get("questions", [])) or 1

    merged["rel_pre"]  = merged["score_pre"]  / max_pre
    merged["rel_post"] = merged["score_post"] / max_post

    st.subheader("Tabla comparativa por estudiante")
    st.dataframe(merged[["user","score_pre","score_post","delta","rel_pre","rel_post"]], use_container_width=True)

    # métricas globales
    mean_pre   = merged["score_pre"].mean()
    mean_post  = merged["score_post"].mean()
    mean_delta = merged["delta"].mean()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Media pretest", f"{mean_pre:.2f}")
    with col_m2:
        st.metric("Media postest", f"{mean_post:.2f}")
    with col_m3:
        st.metric("Ganancia promedio (Δ)", f"{mean_delta:.2f}")

    # pequeño gráfico de barras
    st.subheader("Distribución de puntuaciones")
    try:
        df_plot = merged.melt(id_vars="user", value_vars=["score_pre","score_post"], var_name="tipo", value_name="score")
        st.bar_chart(df_plot, x="user", y="score", color="tipo")
    except Exception:
        st.info("No se pudo generar el gráfico, pero la tabla ya resume los resultados.")

    # exportar CSV
    if st.button("Exportar CSV de resultados pre/post"):
        out_path = QUIZ_DIR / f"learning_{pre_sel['quiz_id']}_{post_sel['quiz_id']}.csv"
        merged.to_csv(out_path, index=False)
        st.success(f"Archivo exportado en: {out_path.as_posix()}")

# -------------------------------------------------------
# QUIZ MANUAL (docente define preguntas y opciones)
# -------------------------------------------------------
elif seccion == "Quiz manual":
    st.header("Quiz manual")

    # Helpers locales (no pisan los tuyos)
    def qm_jload(p: Path, default=None):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def qm_jsave(p: Path, obj):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def qm_append_csv(path: Path, row: dict, cols: list):
        df = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=cols)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)


    # Tabs según rol
    if is_docente:
        qm_tab_doc, qm_tab_res = st.tabs(["Docente: cargar y dirigir", "Resultados"])
    else:
        qm_tab_est, qm_tab_res = st.tabs(["Estudiante: unirse y responder", "Resultados"])

    # ---------------- DOCENTE ----------------
    if is_docente:
        with qm_tab_doc:
            # Crear quiz nuevo (SOLO aquí: ya no aparece en Resultados)
            st.subheader("Crear un nuevo quiz manual")
            with st.form("qm_create"):
                qm_title = st.text_input("Título del quiz", "Evaluación de Estructuras de Datos")
                qm_code  = st.text_input("Código de acceso (opcional, sino se genera)", "")

                # 🔴 NUEVO: rol del quiz según el diseño del perfil
                qm_role = st.radio(
                    "Rol del quiz en el diseño (según el perfil)",
                    options=["pre", "post", "other"],
                    index=0,
                    format_func=lambda v: {"pre": "pretest", "post": "postest", "other": "otro"}[v]
                )

                create_q = st.form_submit_button("Crear quiz")

            if create_q:
                quiz_id = f"QZ-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid4())[:6]}"
                qdir = QUIZ_DIR / quiz_id
                qdir.mkdir(parents=True, exist_ok=True)
                quiz = {
                    "quiz_id": quiz_id,
                    "title": qm_title.strip() or quiz_id,
                    "access_code": (qm_code.strip() or quiz_id),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "quiz_role": qm_role,          # 🔴 aquí se guarda pre/post/other
                    "questions": []
                }

                qm_jsave(qdir / "quiz.json", quiz)
                qm_jsave(qdir / "state.json", {"status": "waiting", "current_idx": -1, "started_at": None, "finished_at": None})
                (qdir / "answers.csv").write_text("quiz_id,user,q_idx,choice_idx,correct,timestamp\n", encoding="utf-8")
                (qdir / "roster.csv").write_text("user,joined_at,score\n", encoding="utf-8")
                st.success(f"Quiz creado: {quiz_id}. Código: {quiz['access_code']}")

            st.divider()
            st.subheader("Cargar preguntas y dirigir el quiz")

            # (si quieres eliminar por completo el campo de ID/Código, borra las 15 líneas siguientes)
            quiz_id_in = st.text_input("ID/Código del quiz", key="qm_in")
            qdir, q = None, None
            if quiz_id_in:
                qdir = QUIZ_DIR / quiz_id_in
                q = qm_jload(qdir / "quiz.json")
            if not q:
                encontrados = []
                for sub in QUIZ_DIR.glob("QZ-*"):
                    qq = qm_jload(sub / "quiz.json")
                    if qq and qq.get("access_code") == quiz_id_in:
                        encontrados.append(sub)
                if encontrados:
                    qdir = encontrados[0]
                    q = qm_jload(qdir / "quiz.json")

            if not q:
                st.warning("Crea un nuevo quiz o ingresa un ID/código válido.")
            else:
                st.info(f"Título: {q['title']}  ·  Código: {q.get('access_code', q['quiz_id'])}")              # Editar tipo de quiz (pre / post / otro)
                role_actual = q.get("quiz_role", "otro")
                opciones_role = ["pretest","postest","otro"]
                idx_role = opciones_role.index(role_actual) if role_actual in opciones_role else 2
                nuevo_role = st.selectbox(
                    "Tipo de quiz en el diseño (para análisis pre/post)",
                    opciones_role,
                    index=idx_role,
                    key=f"qm_role_edit_{q['quiz_id']}"
                )
                if st.button("Guardar tipo de quiz", key=f"btn_save_role_{q['quiz_id']}"):
                    q["quiz_role"] = nuevo_role
                    qm_jsave(qdir / "quiz.json", q)
                    st.success("Tipo de quiz actualizado.")

                with st.expander("Generación automática desde Catálogo (beta)"):
                # Filtros y opciones
                    temas = ["(todos)"] + sorted(df_meta["tema"].dropna().astype(str).unique().tolist()) if not df_meta.empty else ["(todos)"]
                    colg1, colg2 = st.columns(2)
                    with colg1:
                        g_topic = st.selectbox("Filtrar por tema", temas, index=0, key="gen_topic")
                        g_state = st.selectbox("Filtrar por estado", ["aceptar","pendiente","ajustar","descartar","(todos)"], index=0, key="gen_state")
                    with colg2:
                        g_mode  = st.radio("Tipo de pregunta", ["concepto","prompt"], index=0, horizontal=True, key="gen_mode")
                        g_strat = st.selectbox("Estrategia de distractores", ["mix","tfidf","mutacion","llm"], index=0, key="gen_strat")
                    colg3, colg4, colg5 = st.columns(3)
                    with colg3:
                        g_n = st.number_input("Nº de preguntas a generar", min_value=1, max_value=30, value=5, step=1, key="gen_n")
                    with colg4:
                        g_k = st.number_input("Distractores por pregunta", min_value=3, max_value=4, value=3, step=1, key="gen_k")
                    with colg5:
                        g_seed = st.number_input("Semilla aleatoria (opcional)", min_value=0, max_value=999999, value=0, step=1, key="gen_seed")

                    if LLM_ENABLE:
                        st.caption("LLM activado por entorno (LLM_ENABLE=1). Se usará si eliges estrategia 'llm' o 'mix'.")
                    else:
                        st.caption("LLM desactivado. Solo TF-IDF y mutación de texto.")

                    if st.button("Generar y añadir al quiz", key="gen_btn_add"):
                        gen = auto_generate_questions_from_catalog(
                            df_meta=df_meta,
                            mode=g_mode,
                            topic_filter=None if g_topic in ("", "(todos)") else g_topic,
                            state_filter=g_state,
                            n_questions=int(g_n),
                            k_distractors=int(g_k),
                            strategy=g_strat,
                            seed=int(g_seed)
                        )
                        if not gen:
                            st.warning("No se generaron preguntas. Revisa filtros/estado o cambia a 'concepto'.")
                        else:
                            # anexar a tu quiz actual
                            q["questions"].extend(gen)
                            qm_jsave(qdir / "quiz.json", q)
                            st.success(f"Se añadieron {len(gen)} preguntas al quiz. Total ahora: {len(q['questions'])}")
                            st.toast("Preguntas generadas y guardadas.", icon="✅")

                st.markdown("Agregar una pregunta")
                with st.form("qm_add"):
                    stem = st.text_area("Enunciado de la pregunta", height=80)
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        use_catalog = st.checkbox("Usar imagen del catálogo", value=False)
                        img_id_sel = st.selectbox("image_id del catálogo", df_meta["image_id"].tolist() if (use_catalog and not df_meta.empty) else [])
                    with col_img2:
                        img_upload = st.file_uploader("o subir imagen (.png/.jpg)", type=["png","jpg","jpeg"], key="qm_up")

                    opt1 = st.text_input("Opción 1")
                    opt2 = st.text_input("Opción 2")
                    opt3 = st.text_input("Opción 3")
                    opt4 = st.text_input("Opción 4")
                    correct_idx = st.radio("¿Cuál es la correcta?", [0,1,2,3], format_func=lambda i: f"Opción {i+1}")
                    add_q = st.form_submit_button("Añadir pregunta")

                if add_q:
                    image_path = ""
                    image_id   = ""
                    if use_catalog and img_id_sel:
                        row = df_meta[df_meta["image_id"] == img_id_sel].iloc[0]
                        image_path = _safe_path(row.get("ruta", ""))
                        image_id   = _safe_path(row.get("image_id", ""))
                    elif img_upload is not None:
                        raw = img_upload.read()
                        fn = f"IMG-QZ-{date.today().isoformat()}-{str(uuid4())[:8]}.png"
                        dest = IMG_DIR / fn
                        with open(dest, "wb") as f:
                            f.write(raw)
                        image_path = dest.as_posix()

                    options = [opt1.strip(), opt2.strip(), opt3.strip(), opt4.strip()]
                    if not stem.strip() or any(o == "" for o in options):
                        st.error("Completa enunciado y las 4 opciones.")
                    else:
                        q["questions"].append({
                            "stem": stem.strip(),
                            "options": options,
                            "correct_idx": int(correct_idx),
                            "image_path": _safe_path(image_path),
                            "image_id": _safe_path(image_id),
                        })
                        qm_jsave(qdir / "quiz.json", q)
                        st.success(f"Pregunta añadida. Total: {len(q['questions'])}")

                st.markdown("Preguntas cargadas")
                if q["questions"]:
                    for i, item in enumerate(q["questions"], 1):
                        st.write(f"{i}) {item['stem']}")
                        if item.get("image_path"):
                            st.caption(f"Imagen: {item['image_path']}")
                        st.caption("Opciones: " + " | ".join([f"{k+1}) {t}" for k, t in enumerate(item['options'])]))
                else:
                    st.info("Aún no hay preguntas.")

                st.divider()
                st.subheader("Dirección de la sesión")
                state = qm_jload(qdir / "state.json")
                ustate_path = qdir / "user_state.json"
                if ustate_path.exists():
                    st.write("Estado individual (autodirigido):")
                    st.json(qm_jload(ustate_path))

                qtime = st.number_input("Tiempo por pregunta (segundos)", min_value=5, max_value=300, value=30)

                # NUEVO: modo autodirigido por estudiante
                self_mode = st.toggle("Modo autodirigido por estudiante (el estudiante navega sus preguntas)", value=True)

                colA, colB, colC, colD = st.columns(4)
                with colA:
                    if st.button("Iniciar", disabled=(state["status"] != "waiting" or len(q["questions"]) == 0)):
                        state.update({
                            "status": "running",
                            "current_idx": 0,  # se usa solo si mode == 'teacher'
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                            "question_start": time.time(),
                            "qtime": int(qtime),
                            "mode": ("self" if self_mode else "teacher")
                        })
                        qm_jsave(qdir / "state.json", state); st.rerun()
                        log_event("quiz.start", ref_id=q["quiz_id"])

                with colB:
                    if st.button("Anterior", disabled=(state["status"] != "running" or state.get("mode","teacher") != "teacher" or state["current_idx"] <= 0)):
                        state["current_idx"] -= 1; state["question_start"] = time.time()
                        qm_jsave(qdir / "state.json", state); st.rerun()
                        log_event("quiz.prev", ref_id=q["quiz_id"], extra={"idx": state["current_idx"]})

                with colC:
                    if st.button("Siguiente", disabled=(state["status"] != "running" or state.get("mode","teacher") != "teacher" or state["current_idx"] >= len(q["questions"]) - 1)):
                        state["current_idx"] += 1; state["question_start"] = time.time()
                        qm_jsave(qdir / "state.json", state); st.rerun()
                        log_event("quiz.next", ref_id=q["quiz_id"], extra={"idx": state["current_idx"]})

                with colD:
                    if st.button("Finalizar", disabled=(state["status"] == "finished")):
                        state.update({"status": "finished", "finished_at": datetime.now().isoformat(timespec="seconds")})
                        qm_jsave(qdir / "state.json", state); st.rerun()
                        log_event("quiz.finish", ref_id=q["quiz_id"])

                st.write(f"Estado: {state['status']}  ·  Modo: {state.get('mode','teacher')}")
                if state["status"] in ("running", "finished") and state.get("mode","teacher") == "teacher" and state["current_idx"] >= 0 and q["questions"]:
                    qi = state["current_idx"]
                    item = q["questions"][qi]
                    img_path = _safe_path(item.get("image_path", ""))
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Q{qi+1}: {item['stem']}", use_column_width=True)
                    else:
                        st.write(f"Q{qi+1}: {item['stem']}")
                    if isinstance(item.get("options"), (list, tuple)):
                        for i, opt in enumerate(item["options"]):
                            st.write(f"{i+1}) {opt}")
                else:
                    if state.get("mode","teacher") == "self":
                        st.info("Modo autodirigido: el docente no navega preguntas; los estudiantes avanzan en sus propios dispositivos.")

    # ---------------- ESTUDIANTE ----------------
    if not is_docente:
        with qm_tab_est:
            st.subheader("Unirse y responder")

            # 🔁 Si hay que resetear, limpiar campos ANTES de crear widgets
            if st.session_state.get("qm_reset", False):
                st.session_state["qm_reset"] = False
                st.session_state["qm_join"] = ""
                st.session_state["qm_user"] = ""

            # Mostrar mensaje de finalización si existe
            if "quiz_done_msg" in st.session_state:
                st.success(st.session_state["quiz_done_msg"])
                del st.session_state["quiz_done_msg"]

            code_join = st.text_input("ID o código del quiz", key="qm_join")
            user = st.text_input("Tu nombre", key="qm_user")
            if code_join and user:
                qdir = QUIZ_DIR / code_join
                q = qm_jload(qdir / "quiz.json")
                if not q:
                    for sub in QUIZ_DIR.glob("QZ-*"):
                        qq = qm_jload(sub / "quiz.json")
                        if qq and qq.get("access_code") == code_join:
                            qdir, q = sub, qq
                            break
                if not q:
                    st.error("ID/código no válido.")
                else:
                    st.caption(f"Quiz: {q['title']}")
                    roster_path = qdir / "roster.csv"
                    roster_df = pd.read_csv(roster_path) if roster_path.exists() and roster_path.stat().st_size > 0 else pd.DataFrame(columns=["user","joined_at","score"])
                    if user not in roster_df["user"].tolist():
                        qm_append_csv(roster_path, {"user": user, "joined_at": datetime.now().isoformat(timespec="seconds"), "score": 0}, ["user","joined_at","score"])
                        log_event("quiz.join", ref_id=q["quiz_id"], extra={"user": user})
                        st.success("Ingreso registrado.")
                    state = qm_jload(qdir / "state.json")
                    if state["status"] == "waiting":
                        st.info("Esperando a que el docente inicie…")
                    elif state["status"] == "running":
                        mode = state.get("mode", "teacher")

                        if mode == "self":
                            # Estado por usuario
                            ustate = _get_or_init_user_state(qdir, user)
                            qi = int(ustate.get("idx", 0))
                            qi = max(0, min(qi, len(q["questions"]) - 1))
                            st.caption(f"Progreso: {qi+1} / {len(q['questions'])}")

                            item = q["questions"][qi]
                            img_path = _safe_path(item.get("image_path", ""))
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=f"Pregunta {qi+1}: {item['stem']}", use_column_width=True)
                            else:
                                st.write(f"Pregunta {qi+1}: {item['stem']}")

                            # Preseleccionar respuesta previa si existe
                            ans_path = qdir / "answers.csv"
                            prev_choice = None
                            if ans_path.exists() and ans_path.stat().st_size > 0:
                                tmp = pd.read_csv(ans_path)
                                mask = (tmp["quiz_id"] == q["quiz_id"]) & (tmp["user"] == user) & (tmp["q_idx"] == qi)
                                if mask.any():
                                    prev_choice = int(tmp.loc[mask, "choice_idx"].iloc[0])

                            choice = st.radio(
                                "Selecciona tu respuesta",
                                list(enumerate(item["options"])),
                                index=(prev_choice if prev_choice is not None else 0),
                                format_func=lambda x: x[1],
                                key=f"qm_choice_{q['quiz_id']}_{user}_{qi}"
                            )

                            colx1, colx2, colx3, colx4 = st.columns([1,1,1,1])
                            with colx1:
                                if st.button("Guardar", key=f"save_{q['quiz_id']}_{user}_{qi}"):
                                    try:
                                        choice_idx = int(choice[0]) if isinstance(choice, (tuple, list)) else int(choice)
                                    except Exception:
                                        st.error("Selecciona una opción válida."); st.stop()
                                    correct = 1 if choice_idx == int(item["correct_idx"]) else 0
                                    _upsert_answer(ans_path, q["quiz_id"], user, qi, choice_idx, correct)
                                    _recompute_user_score(qdir, q["quiz_id"], user)
                                    st.success("Respuesta guardada.")

                            with colx2:
                                if st.button("Anterior", disabled=(qi <= 0), key=f"prev_{q['quiz_id']}_{user}_{qi}"):
                                    _update_user_state(qdir, user, idx=qi-1)
                                    st.rerun()

                            with colx3:
                                if st.button("Siguiente", disabled=(qi >= len(q["questions"])-1), key=f"next_{q['quiz_id']}_{user}_{qi}"):
                                    _update_user_state(qdir, user, idx=qi+1)
                                    st.rerun()

                            with colx4:
                                if st.button("Finalizar mi evaluación", key=f"finish_{q['quiz_id']}_{user}"):
                                    _update_user_state(
                                        qdir,
                                        user,
                                        finished=True,
                                        finished_at=datetime.now().isoformat(timespec="seconds")
                                    )
                                    score = _recompute_user_score(qdir, q["quiz_id"], user)

                                    # Guardamos mensaje para mostrarlo luego en la pantalla inicial
                                    st.session_state["quiz_done_msg"] = (
                                        f"Evaluación finalizada. Aciertos: {score}/{len(q['questions'])}"
                                    )

                                    # 🔁 marcar que en el próximo ciclo se reseteen los inputs
                                    st.session_state["qm_reset"] = True

                                    # Forzar nuevo ciclo de ejecución
                                    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


                            if state["status"] == "finished":
                                score = _recompute_user_score(qdir, q["quiz_id"], user)
                                st.warning(f"El docente finalizó el quiz. Tu puntuación registrada: {score}/{len(q['questions'])}")

                        else:
                            # Modo clásico (docente navega)
                            qi = state["current_idx"]
                            item = q["questions"][qi]
                            img_path = _safe_path(item.get("image_path", ""))
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=f"Pregunta {qi+1}: {item['stem']}", use_column_width=True)
                            else:
                                st.write(f"Pregunta {qi+1}: {item['stem']}")
                            if isinstance(item.get("options"), (list, tuple)):
                                for i, opt in enumerate(item["options"]):
                                    st.write(f"{i+1}) {opt}")
                            choice = st.radio("Selecciona tu respuesta", list(enumerate(item["options"])), format_func=lambda x: x[1], key=f"qm_choice_{code_join}_{qi}")
                            if st.button("Responder", key=f"qm_btn_{code_join}_{qi}"):
                                try:
                                    choice_idx = int(choice[0]) if isinstance(choice, (tuple, list)) else int(choice)
                                except Exception:
                                    st.error("Selecciona una opción válida."); st.stop()
                                correct = 1 if choice_idx == int(item["correct_idx"]) else 0
                                ans_path = qdir / "answers.csv"
                                _upsert_answer(ans_path, q["quiz_id"], user, qi, choice_idx, correct)
                                _recompute_user_score(qdir, q["quiz_id"], user)
                                st.success("Respuesta guardada.")


    # ---------------- RESULTADOS (para ambos roles) ----------------
    with qm_tab_res:
        st.subheader("Resultados")
        code_out = st.text_input("ID o código del quiz", key="qm_out")
        if code_out:
            qdir = QUIZ_DIR / code_out
            q = qm_jload(qdir / "quiz.json")
            if not q:
                for sub in QUIZ_DIR.glob("QZ-*"):
                    qq = qm_jload(sub / "quiz.json")
                    if qq and qq.get("access_code") == code_out:
                        qdir, q = sub, qq
                        break
            if not q:
                st.error("No encontrado.")
            else:
                roster_path = qdir / "roster.csv"
                if roster_path.exists() and roster_path.stat().st_size > 0:
                    roster_df = pd.read_csv(roster_path)
                    st.write("Ranking (por aciertos):")
                    st.dataframe(roster_df.sort_values("score", ascending=False), use_container_width=True)
                ans_path = qdir / "answers.csv"
                if ans_path.exists() and ans_path.stat().st_size > 0:
                    df_ans = pd.read_csv(ans_path)
                    if st.button("Exportar respuestas"):
                        out = qdir / "respuestas_export.csv"
                        df_ans.to_csv(out, index=False)
                        st.success(f"Exportado {out.as_posix()}")
                else:
                    st.info("Sin respuestas aún.")

            st.divider()
        st.subheader("Resultados de aprendizaje (comparación PRE vs POST)")

        col_pre, col_post = st.columns(2)
        with col_pre:
            pre_code = st.text_input("ID o código del quiz PRE", key="qm_pre_code")
        with col_post:
            post_code = st.text_input("ID o código del quiz POST", key="qm_post_code")

        if pre_code and post_code:
            # Resolver quiz PRE
            pre_dir = QUIZ_DIR / pre_code
            pre_q = qm_jload(pre_dir / "quiz.json")
            if not pre_q:
                for sub in QUIZ_DIR.glob("QZ-*"):
                    qq = qm_jload(sub / "quiz.json")
                    if qq and qq.get("access_code") == pre_code:
                        pre_dir, pre_q = sub, qq
                        break

            # Resolver quiz POST
            post_dir = QUIZ_DIR / post_code
            post_q = qm_jload(post_dir / "quiz.json")
            if not post_q:
                for sub in QUIZ_DIR.glob("QZ-*"):
                    qq = qm_jload(sub / "quiz.json")
                    if qq and qq.get("access_code") == post_code:
                        post_dir, post_q = sub, qq
                        break

            if not pre_q or not post_q:
                st.error("No se pudo cargar alguno de los quizzes (revisa IDs/códigos).")
            else:
                st.caption(
                    f"PRE: {pre_q['title']} (rol={pre_q.get('quiz_role','otro')})  ·  "
                    f"POST: {post_q['title']} (rol={post_q.get('quiz_role','otro')})"
                )

                pre_roster_path = pre_dir / "roster.csv"
                post_roster_path = post_dir / "roster.csv"

                if not pre_roster_path.exists() or not post_roster_path.exists():
                    st.warning("Alguno de los quizzes no tiene aún respuestas registradas.")
                else:
                    pre_df = pd.read_csv(pre_roster_path)
                    post_df = pd.read_csv(post_roster_path)

                    pre_df = pre_df[["user","score"]].rename(columns={"score": "score_pre"})
                    post_df = post_df[["user","score"]].rename(columns={"score": "score_post"})

                    merged = pre_df.merge(post_df, on="user", how="inner")
                    if merged.empty:
                        st.info("No hay estudiantes en común entre PRE y POST.")
                    else:
                        merged["delta"] = merged["score_post"] - merged["score_pre"]
                        st.markdown("**Tabla de ganancia de aprendizaje (solo quienes hicieron ambos quizzes)**")
                        st.dataframe(merged, use_container_width=True)

                        if st.button("Exportar CSV de ganancia", key="btn_export_gain"):
                            out_path = QUIZ_DIR / f"ganancia_{pre_q['quiz_id']}_{post_q['quiz_id']}.csv"
                            merged.to_csv(out_path, index=False)
                            st.success(f"Archivo generado: {out_path.as_posix()}")


# -------------------------------------------------------
# ENCUESTA DE PERCEPCIÓN
# -------------------------------------------------------
elif seccion == "Encuesta de percepción":
    st.header("Encuesta de percepción sobre GEN-EDViz")

    # --- Vista ESTUDIANTE: responder encuesta ---
    if not is_docente:
        st.markdown("Responde honestamente cómo percibes el uso de GEN-EDViz en la asignatura.")

        nombre = st.text_input("Tu nombre (o seudónimo)", value=user_name or "")
        tema   = st.text_input("Tema o unidad que estás evaluando", "")

        st.markdown("### Indica tu grado de acuerdo (1 = muy en desacuerdo, 5 = muy de acuerdo)")

        likert_labels = ["1", "2", "3", "4", "5"]

        claridad = st.slider(
            "Los materiales generados con GEN-EDViz son claros y comprensibles.",
            min_value=1, max_value=5, value=4, format="%d"
        )
        utilidad = st.slider(
            "GEN-EDViz me ayuda a entender mejor las estructuras de datos.",
            min_value=1, max_value=5, value=4, format="%d"
        )
        aprendizaje = st.slider(
            "Siento que he mejorado mi aprendizaje gracias a las imágenes generadas.",
            min_value=1, max_value=5, value=4, format="%d"
        )
        motivacion = st.slider(
            "El uso de GEN-EDViz hace la clase más motivante/interesante.",
            min_value=1, max_value=5, value=4, format="%d"
        )
        sesgos = st.slider(
            "Las imágenes son neutrales y respetuosas (sin sesgos evidentes).",
            min_value=1, max_value=5, value=4, format="%d"
        )

        enviado = st.button("Enviar respuestas")

        if enviado:
            if not nombre.strip():
                st.warning("Escribe tu nombre antes de enviar.")
            else:
                now = datetime.now().isoformat(timespec="seconds")
                row = {
                    "user": nombre.strip(),
                    "role": st.session_state.get("auth_role", ""),
                    "tema": tema.strip(),
                    "claridad": claridad,
                    "utilidad": utilidad,
                    "aprendizaje": aprendizaje,
                    "motivacion": motivacion,
                    "sesgos": sesgos,
                    "timestamp": now
                }

                # actualizar df_percep local y guardar
                df_loc = df_percep.copy()
                df_loc = pd.concat([df_loc, pd.DataFrame([row])], ignore_index=True)
                save_csv(df_loc, PERCEP_PATH)

                st.success("Gracias, tu respuesta ha sido registrada. ✅")
                log_event("percepcion.submit", ref_id=tema.strip(), extra=row)

    # --- Vista DOCENTE: resumen de percepción ---
    else:
        st.markdown("Vista del docente: resumen de la encuesta de percepción.")

        if df_percep.empty:
            st.info("Aún no hay respuestas registradas en la encuesta.")
        else:
            # Filtro por tema (opcional)
            temas = ["(todos)"] + sorted(df_percep["tema"].fillna("").unique().tolist())
            tema_sel = st.selectbox("Filtrar por tema", temas, index=0)

            view = df_percep.copy()
            if tema_sel != "(todos)":
                view = view[view["tema"] == tema_sel]

            if view.empty:
                st.info("No hay respuestas para ese tema.")
            else:
                st.subheader("Medias por ítem")
                medias = view[["claridad","utilidad","aprendizaje","motivacion","sesgos"]].mean()
                st.write(medias.to_frame("media (1–5)"))

                # ---- Alfa de Cronbach de la escala Likert ----
                items_cols = ["claridad","utilidad","aprendizaje","motivacion","sesgos"]
                sub_items = view[items_cols].apply(pd.to_numeric, errors="coerce")
                alpha = cronbach_alpha(sub_items)

                if not np.isnan(alpha):
                    st.metric("Alfa de Cronbach (consistencia interna)", f"{alpha:.3f}")
                    st.caption("Valores ≥ 0.70 se consideran aceptables como regla general.")
                else:
                    st.info("No se puede calcular alfa de Cronbach (muy pocas respuestas o varianza cero).")

                # número de respuestas
                st.metric("Nº de respuestas", len(view))

                # Exportar CSV
                if st.button("Exportar respuestas de percepción"):
                    out = DATA_DIR / f"percepcion_export_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
                    view.to_csv(out, index=False)
                    st.success(f"Exportado {out.as_posix()}")

# -------------------------------------------------------
# RETROALIMENTACIÓN COMPARTIDA (pizarra de ideas + votos)
# -------------------------------------------------------
elif seccion == "Retroalimentación compartida":
    st.header("Retroalimentación compartida")

    # --- AUTOREFRESH SOLO PARA ESTA SECCIÓN ---
    try:
        from streamlit_autorefresh import st_autorefresh
        auto = st.toggle("Auto-actualizar cada 5 s", value=False, key="fb_auto_toggle")
        if auto:
            st_autorefresh(interval=10000, key="fb_auto_refresh")
    except Exception:
        pass

    # ---------- helpers ----------
    def fb_append_csv(path: Path, row_dict: dict, cols: list):
        df = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=cols)
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        df.to_csv(path, index=False)

    def fb_load_csv(path: Path, cols: list):
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df
        return pd.DataFrame(columns=cols)

    def fb_jload(p: Path, default=None):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def fb_jsave(p: Path, obj):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ---------- TABS POR ROL ----------
    if is_docente:
        tab_d, tab_p = st.tabs(["Docente: crear pizarra", "Panel y exportación"])
    else:
        tab_e, tab_p = st.tabs(["Estudiante: aportar/votar", "Panel y exportación"])

    # ------------- DOCENTE -------------
    if is_docente:
        with tab_d:
            st.subheader("Crear nueva pizarra de ideas")
            with st.form("fb_new_board"):
                title = st.text_input("Título de la pizarra", "Ideas clave de Grafos")
                tema  = st.text_input("Tema (coincidir con catálogo si quieres)", "grafos")
                pregunta = st.text_area("Pregunta guía", "¿Qué hace comprensible una imagen de BFS/DFS?")
                crear = st.form_submit_button("Crear pizarra")

            if crear:
                board_id = f"FB-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid4())[:6]}"
                bdir = FEED_DIR / board_id
                bdir.mkdir(parents=True, exist_ok=True)

                board = {
                    "board_id": board_id,
                    "title": title,
                    "tema": tema,
                    "guiding_question": pregunta,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "created_by": user_name or "docente",
                }
                fb_jsave(bdir / "board.json", board)

                # CSVs base
                (bdir / "ideas.csv").write_text("idea_id,user,text,tags,upvotes,timestamp\n", encoding="utf-8")
                (bdir / "votes.csv").write_text("user,idea_id,timestamp\n", encoding="utf-8")

                log_event("board.create", ref_id=board_id, extra={"title": title, "tema": tema})
                st.success(f"Pizarra creada: {board_id}. Comparte ese ID con los estudiantes.")

            st.divider()
            st.subheader("Abrir pizarra existente")
            fb_id_in = st.text_input("ID de pizarra", key="fb_id_in")
            if fb_id_in:
                bdir = FEED_DIR / fb_id_in
                board = fb_jload(bdir / "board.json")
                if not board:
                    st.error("No se encontró la pizarra.")
                else:
                    st.write(board)
                    st.info("Comparte el ID con los estudiantes para que aporten ideas y voten.")

    # ------------- ESTUDIANTE -------------
    if not is_docente:
        with tab_e:
            st.subheader("Unirse a pizarra y aportar")
            fb_id_join = st.text_input("ID de pizarra", key="fb_id_join")
            user = st.text_input("Tu nombre", key="fb_user")
            if fb_id_join and user:
                bdir = FEED_DIR / fb_id_join
                board = fb_jload(bdir / "board.json")
                if not board:
                    st.error("ID no válido.")
                else:
                    st.caption(f"Pizarra: {board['title']}  ·  Tema: {board['tema']}")
                    st.write(f"Pregunta guía: {board['guiding_question']}")

                    # Aportar idea
                    st.markdown("Aporta una idea")
                    idea_txt = st.text_area("Idea / argumento", "")
                    tags_txt = st.text_input("Etiquetas (separadas por coma)", "claridad, accesibilidad")
                    if st.button("Publicar idea", key="fb_pub"):
                        if idea_txt.strip():
                            ideas_path = bdir / "ideas.csv"
                            idea_id = f"ID-{str(uuid4())[:8]}"
                            fb_append_csv(
                                ideas_path,
                                {
                                    "idea_id": idea_id,
                                    "user": user,
                                    "text": idea_txt.strip(),
                                    "tags": tags_txt.strip(),
                                    "upvotes": 0,
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                },
                                ["idea_id","user","text","tags","upvotes","timestamp"],
                            )
                            st.success("Idea publicada.")
                        else:
                            st.warning("Escribe una idea antes de publicar.")

                    st.divider()
                    st.markdown("Lista de ideas")
                    ideas_df = fb_load_csv(bdir / "ideas.csv", ["idea_id","user","text","tags","upvotes","timestamp"])
                    if ideas_df.empty:
                        st.info("Aún no hay ideas.")
                    else:
                        st.dataframe(ideas_df.sort_values(["upvotes","timestamp"], ascending=[False, True]), use_container_width=True)

                        # Votar
                        st.markdown("Vota una idea")
                        sel = st.selectbox("Selecciona idea", ideas_df["idea_id"].tolist())
                        if st.button("Votar", key="fb_vote"):
                            votes_path = bdir / "votes.csv"
                            vdf = fb_load_csv(votes_path, ["user","idea_id","timestamp"])
                            ya = ((vdf["user"] == user) & (vdf["idea_id"] == sel)).any()
                            if ya:
                                st.warning("Ya votaste esa idea.")
                            else:
                                fb_append_csv(
                                    votes_path,
                                    {"user": user, "idea_id": sel, "timestamp": datetime.now().isoformat(timespec="seconds")},
                                    ["user","idea_id","timestamp"],
                                )
                                # sumar voto en ideas.csv
                                ideas_df.loc[ideas_df["idea_id"] == sel, "upvotes"] = pd.to_numeric(
                                    ideas_df.loc[ideas_df["idea_id"] == sel, "upvotes"], errors="coerce"
                                ).fillna(0).astype(int) + 1
                                ideas_df.to_csv(bdir / "ideas.csv", index=False)
                                st.success("Voto registrado.")

    # ------------- PANEL Y EXPORTACIÓN (AMBOS ROLES) -------------
    with tab_p:
        st.subheader("Panel de consolidación y exportación")
        fb_id_out = st.text_input("ID de pizarra", key="fb_id_out")
        if fb_id_out:
            bdir = FEED_DIR / fb_id_out
            board = fb_jload(bdir / "board.json")
            if not board:
                st.error("No se encontró la pizarra.")
            else:
                st.write(board)
                ideas_df = fb_load_csv(bdir / "ideas.csv", ["idea_id","user","text","tags","upvotes","timestamp"])
                if ideas_df.empty:
                    st.info("Sin ideas todavía.")
                else:
                    st.markdown("Top de ideas por votos")
                    st.dataframe(ideas_df.sort_values(["upvotes","timestamp"], ascending=[False, True]), use_container_width=True)

                    # Conteo de etiquetas
                    tag_counts = {}
                    for t in ideas_df["tags"].fillna(""):
                        for tag in [x.strip().lower() for x in t.split(",") if x.strip()]:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    if tag_counts:
                        tags_df = pd.DataFrame(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True), columns=["tag","count"])
                        st.markdown("Frecuencia de etiquetas")
                        st.dataframe(tags_df, use_container_width=True)
                        st.bar_chart(tags_df.set_index("tag"))

                    # Generar conclusiones sencillas
                    if st.button("Generar conclusiones", key="fb_concl"):
                        temas_fuertes = [t for t, c in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3]] if tag_counts else []
                        top3 = ideas_df.sort_values(["upvotes","timestamp"], ascending=[False, True]).head(3)["text"].tolist()
                        conclusiones = []
                        if temas_fuertes:
                            conclusiones.append(f"Los temas más recurrentes fueron: {', '.join(temas_fuertes)}.")
                        if top3:
                            conclusiones.append("Las ideas mejor valoradas resaltan:")
                            for i, t in enumerate(top3, 1):
                                conclusiones.append(f"{i}) {t}")
                        if not conclusiones:
                            conclusiones = ["Aún no hay suficiente participación para conclusiones automáticas."]

                        resumen_path = bdir / "resumen.txt"
                        with open(resumen_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(conclusiones))
                        st.success("Conclusiones generadas y guardadas en resumen.txt")
                        st.text("\n".join(conclusiones))

                    # Exportar CSV consolidado
                    if st.button("Exportar CSV de ideas", key="fb_export"):
                        out = bdir / "ideas_export.csv"
                        ideas_df.to_csv(out, index=False)
                        st.success(f"Exportado {out.as_posix()}")


# -------------------------------------------------------
# Ajustes
# -------------------------------------------------------
else:
    st.header("Ajustes y notas")
    st.write("""
- Este monitor implementa la rúbrica 4×4 colaborativa y métricas objetivas simples.
- Como apoyo a la rúbrica, el prototipo calcula métricas de contraste aproximado y densidad de bordes
  que se muestran al docente, **sin sustituir el criterio humano**.
- Incluye también un cuestionario de percepción tipo Likert y el cálculo de alfa de Cronbach para
  evaluar la consistencia interna de la escala, en línea con el perfil metodológico.
- Puedes ampliar con modelos de similitud texto–imagen (CLIP) y un pequeño predictor de calidad
  entrenado con tus propias evaluaciones (regresión a la media de la rúbrica).
- Los archivos se guardan en la carpeta ./data
""")

