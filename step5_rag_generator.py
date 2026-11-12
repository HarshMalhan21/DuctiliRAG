import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

from step1_shap_bridge import predict_and_create_rag_query
from step4_hybrid_retriever import (
    search_failure_knowledge_base,
    get_chunks_by_source,
    get_figure_metadata
)

load_dotenv()


def _get_secret(key_name):
    val = os.getenv(key_name)
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key_name)
        except Exception:
            pass
    if val:
        val = str(val).strip().strip('"').strip("'")
        os.environ[key_name] = val
    return val


GEMINI_KEYS = [k for k in [_get_secret(
    "GEMINI_API_KEY"), _get_secret("GEMINI_API_KEY_BACKUP")] if k]
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
HF_TOKEN = _get_secret("HF_TOKEN")
DISCOVERED_GROQ_MODELS = []
LAST_LLM_ERROR = ""


def clean_and_render_math(markdown_text, source_list=None):
    r"""Converts \(...\) and \[...\] into Streamlit KaTeX $...$ / $$...$$, removes <br>, and prevents nested $ errors."""
    text = markdown_text

    fixed_lines = []
    for line in text.split("\n"):
        if "|" in line:
            line = re.sub(r"\s*<br\s*/?>\s*", " • ", line, flags=re.IGNORECASE)
        else:
            line = re.sub(r"\s*<br\s*/?>\s*", "\n", line, flags=re.IGNORECASE)
        fixed_lines.append(line)
    text = "\n".join(fixed_lines)

    text = re.sub(r"<sub>\s*([^<]+?)\s*</sub>",
                  r"_{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>\s*([^<]+?)\s*</sup>",
                  r"^{\1}", text, flags=re.IGNORECASE)

    text = re.sub(r"\\\[\s*(.*?)\s*\\\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text, flags=re.DOTALL)
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)",
                  lambda m: f"${m.group(1).strip()}$", text)

    text = re.sub(
        r"^\s*\[\s*([A-Za-z0-9_\\]+?\s*=[^\[\]\n]+?)\s*\]\s*$",
        lambda m: f"$$\n{m.group(1).strip()}\n$$",
        text,
        flags=re.MULTILINE
    )

    math_token_pattern = re.compile(
        r"(\$\$.*?\$\$|\$[^$\n]+\$)", flags=re.DOTALL)
    segments = math_token_pattern.split(text)

    def _fix_raw_latex_parens(match):
        inner = match.group(1).strip()
        if re.search(r"\\(?:frac|dfrac|sigma|pi|sqrt|sum|times|approx|ge|le|neq|Delta|alpha|beta)", inner):
            return f"(${inner}$)"
        return match.group(0)

    for i in range(len(segments)):
        if not (segments[i].startswith("$") and segments[i].endswith("$")):
            segments[i] = re.sub(
                r"\(([^()\n$]{2,140})\)", _fix_raw_latex_parens, segments[i])

    text = "".join(segments)
    text = text.replace(r"\dfrac", r"\frac")

    text = re.sub(
        r"\s*[\[\(]([^\[\]\(\)$\\]*?(?:Page|Pages|Section|Sections|Source|Excerpt|Case Study|Reference|slides_|NASA_|ASTM_|Historical_|Rolling)[^\[\]\(\)$\\]*?)[\]\)]",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s+([.,;:])", r"\1", text)

    if source_list:
        unique_srcs = list(dict.fromkeys(source_list))
        return text.strip() + "\n\n---\n**📚 Referenced Literature:** " + " • ".join(unique_srcs[:5])
    return text.strip()


def get_active_groq_models(groq_client):
    global DISCOVERED_GROQ_MODELS
    if DISCOVERED_GROQ_MODELS:
        return DISCOVERED_GROQ_MODELS
    try:
        all_ids = [m.id for m in groq_client.models.list(
        ).data if getattr(m, "active", True)]
        skip = ("whisper", "tts", "orpheus", "guard", "safeguard",
                "embed", "playai", "scout", "maverick")
        chat = [mid for mid in all_ids if not any(
            w in mid.lower() for w in skip)]
        # Prioritize flagship 120B -> Qwen 3.8 27B -> GPT-OSS 20B
        order = ["gpt-oss-120b", "qwen3.8",
                 "gpt-oss-20b", "llama-3.3", "minimax", "muse"]
        chat.sort(key=lambda m: next(
            (i for i, p in enumerate(order) if p in m.lower()), 99))
        DISCOVERED_GROQ_MODELS = chat[:4]
    except Exception:
        DISCOVERED_GROQ_MODELS = ["openai/gpt-oss-120b",
                                  "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    return DISCOVERED_GROQ_MODELS


def call_llm_with_fallback(system_instruction, user_prompt):
    global LAST_LLM_ERROR
    errors = []

    groq_key = _get_secret("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            g_client = Groq(api_key=groq_key, timeout=22.0, max_retries=0)
            for g_model in get_active_groq_models(g_client):
                is_qwen = "qwen" in g_model.lower()
                sys_msg = (
                    "/no_think\n" + system_instruction) if is_qwen else system_instruction
                try:
                    resp = g_client.chat.completions.create(
                        model=g_model,
                        messages=[{"role": "system", "content": sys_msg}, {
                            "role": "user", "content": user_prompt}],
                        temperature=0.0,
                        max_tokens=1500
                    )
                    text = resp.choices[0].message.content
                    if text:
                        text = re.sub(r"<think>.*?</think>", "",
                                      text, flags=re.DOTALL).strip()
                        if len(text) > 80:
                            return text, f"groq/{g_model}"
                except Exception as e:
                    errors.append(f"Groq({g_model}): {str(e)[:90]}")
        except Exception as e:
            errors.append(f"GroqInit: {str(e)[:90]}")
    else:
        errors.append("GROQ_API_KEY missing")

    # Frontier Gemini 3 Models (3.8 Flash -> 3.6 Flash -> 3.5 Flash-Lite)
    gemini_keys = [k for k in [_get_secret(
        "GEMINI_API_KEY"), _get_secret("GEMINI_API_KEY_BACKUP")] if k]
    if gemini_keys:
        for api_key in gemini_keys:
            try:
                client = genai.Client(
                    api_key=api_key, http_options=types.HttpOptions(timeout=20000))
                for g_model in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]:
                    try:
                        resp = client.models.generate_content(
                            model=g_model,
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction, temperature=0.0)
                        )
                        if resp.text:
                            return resp.text.strip(), g_model
                    except Exception as e:
                        errors.append(f"Gemini({g_model}): {str(e)[:90]}")
            except Exception as e:
                errors.append(f"GeminiInit: {str(e)[:90]}")
    else:
        errors.append("GEMINI_API_KEY missing")

    # Hugging Face Frontier Open-Weight Models
    hf_token = _get_secret("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            hf_client = InferenceClient(api_key=hf_token, timeout=20.0)
            for hf_model in ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]:
                try:
                    resp = hf_client.chat.completions.create(
                        model=hf_model,
                        messages=[{"role": "system", "content": system_instruction}, {
                            "role": "user", "content": user_prompt}],
                        temperature=0.0,
                        max_tokens=1600
                    )
                    if resp.choices[0].message.content:
                        return resp.choices[0].message.content.strip(), f"hf/{hf_model}"
                except Exception as e:
                    errors.append(f"HF({hf_model}): {str(e)[:90]}")
        except Exception as e:
            errors.append(f"HFInit: {str(e)[:90]}")

    LAST_LLM_ERROR = " | ".join(errors[-3:])
    print(f" [ALL LLM PROVIDERS FAILED] {LAST_LLM_ERROR}")
    return "RATE_LIMIT_ERROR", "none"


def format_retrieved_context(retrieved_items):
    blocks, images, labels, seen = [], [], [], set()
    for idx, item in enumerate(retrieved_items, start=1):
        chunk = item["chunk"] if "chunk" in item else item
        meta = chunk["metadata"]
        src_clean = meta["source"].replace("_", " ")
        labels.append(f"{src_clean} ({meta['section']})")
        blocks.append(
            f"--- LITERATURE EXCERPT #{idx} ({src_clean}) ---\n{chunk['text']}\n")

        if meta.get("image_path"):
            for raw_p in str(meta["image_path"]).split("|"):
                p = raw_p.strip()
                if p and os.path.exists(p) and p not in seen:
                    seen.add(p)
                    fm = get_figure_metadata(p)
                    images.append({
                        "image_path": p,
                        "title": fm.get("title", f"Figure ({meta['section']})"),
                        "explanation": fm.get("explanation", ""),
                        "source_page": f"{src_clean} • {meta['section']}"
                    })
    return "\n".join(blocks), images, labels


MATH_STYLE_RULES = (
    "STRICT FORMATTING & MATH RULES:\n"
    "1. Format all mathematical formulas and symbols using single dollar signs `$ ... $` for inline math "
    "(e.g., `$S_e = 0.5 S_{ut}$`, `$\\frac{\\sigma_a}{S_e} + \\frac{\\sigma_m}{S_{ut}} = 1$`, `$10^6$` cycles) "
    "and double dollar signs `$$ ... $$` for standalone equations.\n"
    "2. NEVER use `\\( ... \\)` or `\\[ ... \\]` delimiters, NEVER use `<br>` tags inside tables, "
    "and NEVER append `(Source X)` or `[Page X]` tags at the end of lines."
)


def generate_ml_failure_report(raw_steel_inputs, ml_diag=None, stream=False):
    ml_diag = ml_diag or predict_and_create_rag_query(raw_steel_inputs)
    retrieved = search_failure_knowledge_base(ml_diag["rag_query"], top_k=4)
    context_text, linked_images, source_labels = format_retrieved_context(
        retrieved)

    shap_txt = "\n".join(
        [f"- {d['feature']}={d['value']} ({d['shap_score']} MPa): {d['meaning']}" for d in ml_diag["top_4_shap_drivers"]])
    fixes_txt = "\n".join(
        [f"- **{act.split(':')[0]}:** {act.split(':', 1)[1].strip()}" if ":" in act else f"- {act}" for act in ml_diag["verified_actions"]])

    sys_p = (
        "You are a Senior Metallurgical Failure Analysis Engineer writing a thorough, authoritative laboratory report.\n"
        + MATH_STYLE_RULES
    )
    usr_p = (
        f"Specimen Profile: {ml_diag['specimen_summary']}\n"
        f"Predicted Fatigue Limit: {ml_diag['predicted_mpa']} MPa ({ml_diag['grade']})\n"
        f"Top Negative SHAP Drivers:\n{shap_txt}\n"
        f"Pre-Verified Engineering Remedies:\n{fixes_txt}\n\n"
        f"Retrieved Literature Evidence:\n{context_text}\n\n"
        "Generate a comprehensive Metallurgical Failure Analysis Report with these 4 sections in Markdown:\n"
        "### 1. Root Cause & Microstructural Diagnosis\n"
        "### 2. Historical Case Study Parallels\n"
        "### 3. Applicable ASTM / JIS Testing & Inspection Standards\n"
        "### 4. Recommended Metallurgical & Process Fixes"
    )

    report_md, used_model = call_llm_with_fallback(sys_p, usr_p)
    is_err = (report_md == "RATE_LIMIT_ERROR")
    if not is_err:
        if "### 4" not in report_md and "4. Recommended" not in report_md:
            report_md = report_md.strip(
            ) + f"\n\n### 4. Recommended Metallurgical & Process Fixes\n{fixes_txt}"
        report_md = clean_and_render_math(report_md, source_labels)

    return {
        "ml_diagnostics": ml_diag, "retrieved_items": retrieved,
        "linked_images": linked_images if not is_err else [],
        "report_markdown": report_md, "error": is_err, "model_used": used_model
    }


def ask_metallurgy_question(user_question, filter_material="All", filter_source="All", chat_history=None, stream=False):
    retrieved = search_failure_knowledge_base(
        user_question, top_k=5, filter_material=filter_material, filter_source=filter_source)
    best_score = retrieved[0]["rerank_score"] if retrieved else -10.0

    if best_score < -3.5:
        return {
            "answer": "⚠️ **Question Outside Document Scope:** No matching metallurgical failure evidence was found in the selected documents.",
            "retrieved_items": retrieved, "linked_images": [], "confidence_passed": False, "error": False, "model_used": "Gate"
        }

    context_text, linked_images, source_labels = format_retrieved_context(
        retrieved)
    sys_p = (
        "You are a Senior Metallurgical & Mechanical Failure Analysis Engineer. "
        "Answer the user's question thoroughly and comprehensively using ONLY the provided excerpts and Visual Figure descriptions.\n"
        + MATH_STYLE_RULES
    )
    usr_p = f"Engineering Question: {user_question}\n\nRetrieved Literature & Visual Evidence:\n{context_text}\n\nProvide a thorough, well-structured engineering answer."

    ans_md, used_model = call_llm_with_fallback(sys_p, usr_p)
    is_err = (ans_md == "RATE_LIMIT_ERROR")
    if not is_err:
        ans_md = clean_and_render_math(ans_md, source_labels)

    return {
        "answer": ans_md, "retrieved_items": retrieved,
        "linked_images": linked_images if not is_err else [],
        "confidence_passed": True, "error": is_err, "model_used": used_model
    }


def summarize_document(source_name, stream=False):
    doc_chunks = get_chunks_by_source(source_name, max_chunks=9)
    if not doc_chunks:
        return {"summary": "No chunks found.", "linked_images": [], "chunk_count": 0, "error": True, "model_used": "none"}

    context_text, linked_images, source_labels = format_retrieved_context(
        doc_chunks)
    sys_p = (
        "You are a Senior Engineering Technical Reviewer. Write a comprehensive, detailed 4-section Executive Engineering Summary "
        "covering all key concepts, quantitative data, equations/factors, and Visual Figure descriptions in the document.\n"
        + MATH_STYLE_RULES
    )
    usr_p = (
        f"Summarize `{source_name}` thoroughly in 4 Markdown sections:\n{context_text}\n\n"
        "### 1. Document Scope & Objectives\n"
        "### 2. Key Technical Findings, Visual Figures & Quantitative Data\n"
        "### 3. Failure Mechanisms / Standards Discussed\n"
        "### 4. Practical Engineering Takeaways"
    )

    sum_md, used_model = call_llm_with_fallback(sys_p, usr_p)
    is_err = (sum_md == "RATE_LIMIT_ERROR")
    if not is_err:
        sum_md = clean_and_render_math(sum_md, source_labels)

    return {"summary": sum_md, "linked_images": linked_images if not is_err else [], "chunk_count": len(doc_chunks), "error": is_err, "model_used": used_model}
