from dotenv import load_dotenv
import numpy as np
import joblib
import os
import sys
import re
import json
import base64
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")


load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "metallurgy_failure_kb"
EMBED_CACHE_FILE = "embeddings_cache.pkl"
FIG_CACHE_FILE = "figure_captions_cache.json"

if not hasattr(sys, "_DUCTILIRAG_CACHE"):
    sys._DUCTILIRAG_CACHE = {
        "lc_embed_model": None, "reranker_model": None,
        "vectorstore": None, "bm25_retriever": None, "lc_docs": None
    }
DB_CACHE = sys._DUCTILIRAG_CACHE


def is_generic_caption(meta_dict):
    if not meta_dict or not isinstance(meta_dict, dict):
        return True
    t = meta_dict.get("title", "")
    e = meta_dict.get("explanation", "")
    if "Technical Figure" in t or t.endswith("Figure") or "Figure 1 (" in t or "Figure 2 (" in t:
        return True
    if len(e) < 45 or "Extracted technical" in e or "Visible diagram labels:" in e:
        return True
    return False


def load_figure_caption_cache():
    if os.path.exists(FIG_CACHE_FILE):
        try:
            with open(FIG_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {k: v for k, v in raw.items() if not is_generic_caption(v)}
        except Exception:
            return {}
    return {}


def get_figure_metadata(img_path):
    cache = load_figure_caption_cache()
    norm_p = os.path.normpath(img_path)
    if norm_p in cache:
        return cache[norm_p]
    base = os.path.basename(norm_p)
    for k, v in cache.items():
        if os.path.basename(k) == base:
            return v
    return {"title": base.replace(".png", "").replace("_", " "), "explanation": "Technical figure from referenced section."}


def inspect_figure_with_vision_ai(task):
    img_path, doc_title, page_num, fig_num, near_text = task
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return img_path, None

    prompt = (
        f"You are a Senior Metallurgical & Mechanical Engineering Professor inspecting Figure #{fig_num} on Page {page_num} of '{doc_title}'.\n"
        f"Page text context: {near_text[:500]}\n\n"
        "Look closely at this specific cropped image and write:\n"
        "1. TITLE: A unique, specific 5-to-10 word engineering title describing THIS exact visual (never write 'Technical Figure' or generic slide headers; "
        "distinguish between a testing machine schematic, a dimensional specimen drawing, a lug lubrication-hole diagram, a threaded bolt fracture, "
        "a forged connecting-rod fracture, or a fracture surface with Zones A, B, C).\n"
        "2. EXPLANATION: 2 clear sentences explaining what this specific image shows, including any visible labels (A, B, C, F, 0.30 in, axes, or curves) "
        "and the failure mechanism or test principle illustrated.\n\n"
        "Reply in EXACTLY two lines:\nTITLE: <title>\nEXPLANATION: <explanation>"
    )

    raw_reply = None
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            g_client = Groq(api_key=groq_key, timeout=14.0, max_retries=0)
            for v_model in ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]:
                try:
                    resp = g_client.chat.completions.create(
                        model=v_model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"}}
                        ]}],
                        temperature=0.0, max_tokens=220
                    )
                    if resp.choices[0].message.content:
                        raw_reply = resp.choices[0].message.content
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if not raw_reply:
        gemini_keys = [k for k in [
            os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_BACKUP")] if k]
        for g_key in gemini_keys:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(
                    api_key=g_key, http_options=types.HttpOptions(timeout=14000))
                for gm in ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]:
                    try:
                        resp = client.models.generate_content(
                            model=gm,
                            contents=[types.Part.from_bytes(
                                data=img_bytes, mime_type="image/png"), prompt],
                            config=types.GenerateContentConfig(temperature=0.0)
                        )
                        if resp.text:
                            raw_reply = resp.text
                            break
                    except Exception:
                        continue
                if raw_reply:
                    break
            except Exception:
                continue

    if raw_reply:
        t_m = re.search(r"TITLE:\s*(.+)", raw_reply)
        e_m = re.search(r"EXPLANATION:\s*(.+)", raw_reply, flags=re.DOTALL)
        if t_m and e_m:
            title = t_m.group(1).strip().strip("*\"'")
            exp = " ".join(e_m.group(1).strip().replace("**", "").split())
            return img_path, {"title": f"{title} (Page {page_num})", "explanation": exp}

    return img_path, None


class CloudSafeQueryEmbedder:
    """Computes 768-d BGE query embeddings via HF Inference API if local PyTorch weights are unavailable."""

    def __init__(self):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN"), timeout=12.0)

    def embed_query(self, text):
        try:
            vec = self.client.feature_extraction(
                text, model="BAAI/bge-base-en-v1.5")
            arr = np.array(vec, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            norm = np.linalg.norm(arr)
            return (arr / norm).tolist() if norm > 0 else arr.tolist()
        except Exception:
            return [0.0] * 768

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class CloudSafeReranker:
    """Lightweight pairwise relevance scorer used when CrossEncoder hub weights are unreachable."""

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        scores = []
        for q, doc in pairs:
            q_toks = set(tokenize_for_bm25(q))
            d_toks = set(tokenize_for_bm25(doc))
            if not q_toks or not d_toks:
                scores.append(-2.0)
                continue
            overlap = len(q_toks.intersection(d_toks)) / float(len(q_toks))
            # Map overlap ratio [0..1] to calibrated logit range [-2.5 .. +4.5]
            scores.append(-2.5 + (7.0 * overlap))
        return np.array(scores, dtype=np.float32)


def get_models():
    if DB_CACHE["lc_embed_model"] is None or DB_CACHE["reranker_model"] is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"

            DB_CACHE["lc_embed_model"] = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
            )
            DB_CACHE["reranker_model"] = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device=device
            )
        except Exception as e:
            print(
                f" [Cloud Fallback] Switching to zero-RAM Serverless Embedder & Scorer ({e})")
            DB_CACHE["lc_embed_model"] = CloudSafeQueryEmbedder()
            DB_CACHE["reranker_model"] = CloudSafeReranker()
    return DB_CACHE["lc_embed_model"], DB_CACHE["reranker_model"]


def get_corpus_stats_fast():
    if not os.path.exists("chunks_data.pkl"):
        return 0, []
    raw_chunks = joblib.load("chunks_data.pkl")
    if os.path.exists("parsed_docs"):
        valid = {f.replace(".md", "") for f in os.listdir(
            "parsed_docs") if f.endswith(".md")}
        synced = [c for c in raw_chunks if c["metadata"].get(
            "source") in valid]
        if len(synced) != len(raw_chunks):
            for idx, c in enumerate(synced):
                c["chunk_id"] = idx
            joblib.dump(synced, "chunks_data.pkl")
            DB_CACHE["vectorstore"] = None
            raw_chunks = synced
    sources = sorted(list({c["metadata"].get("source", "")
                     for c in raw_chunks if c["metadata"].get("source")}))
    return len(raw_chunks), sources


def tokenize_for_bm25(text):
    return "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()


def build_or_load_database(force_reload=False):
    if not force_reload and DB_CACHE["vectorstore"] is not None:
        return DB_CACHE["vectorstore"], DB_CACHE["bm25_retriever"], DB_CACHE["lc_docs"]

    import chromadb
    from chromadb.api.client import SharedSystemClient
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever

    lc_embed_model, _ = get_models()
    raw_chunks = joblib.load("chunks_data.pkl")
    embed_map = joblib.load(EMBED_CACHE_FILE) if os.path.exists(
        EMBED_CACHE_FILE) else {}

    missing = [c["text"] for c in raw_chunks if c["text"] not in embed_map]
    if missing:
        for k, v in zip(missing, lc_embed_model.embed_documents(missing)):
            embed_map[k] = v
        joblib.dump(embed_map, EMBED_CACHE_FILE)

    lc_docs, doc_ids, vecs = [], [], []
    for c in raw_chunks:
        cid, m = c["chunk_id"], c["metadata"]
        doc_ids.append(f"chunk_{cid}")
        vecs.append(embed_map[c["text"]])
        lc_docs.append(Document(page_content=c["text"], metadata={
            "chunk_id": int(cid), "source": str(m.get("source", "")), "section": str(m.get("section", "")),
            "materials_str": ",".join(m.get("materials", [])), "failure_modes_str": ",".join(m.get("failure_modes", [])),
            "image_path": str(m.get("image_path") or "")
        }))

    SharedSystemClient.clear_system_cache()
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        vs = Chroma(client=client, collection_name=COLLECTION_NAME,
                    embedding_function=lc_embed_model, collection_metadata={"hnsw:space": "cosine"})
        if vs._collection.count() != len(lc_docs):
            vs._collection.upsert(ids=doc_ids, embeddings=vecs, documents=[
                                  d.page_content for d in lc_docs], metadatas=[d.metadata for d in lc_docs])
    except Exception:
        SharedSystemClient.clear_system_cache()
        vs = Chroma(client=chromadb.EphemeralClient(), collection_name=COLLECTION_NAME,
                    embedding_function=lc_embed_model, collection_metadata={"hnsw:space": "cosine"})
        vs._collection.upsert(ids=doc_ids, embeddings=vecs, documents=[
                              d.page_content for d in lc_docs], metadatas=[d.metadata for d in lc_docs])

    bm25 = BM25Retriever.from_documents(
        lc_docs, preprocess_func=tokenize_for_bm25)
    bm25.k = min(30, len(lc_docs))
    DB_CACHE.update(
        {"vectorstore": vs, "bm25_retriever": bm25, "lc_docs": lc_docs})
    return vs, bm25, lc_docs


def search_failure_knowledge_base(query_text, top_k=5, filter_material=None, filter_source=None):
    vs, bm25, lc_docs = build_or_load_database()
    _, reranker = get_models()
    total = len(lc_docs)

    chroma_res = vs.similarity_search_with_relevance_scores(
        query_text, k=min(25, total))
    d_rank = {doc.metadata["chunk_id"]: i +
              1 for i, (doc, _) in enumerate(chroma_res)}
    d_score = {doc.metadata["chunk_id"]: float(s) for doc, s in chroma_res}
    s_rank = {doc.metadata["chunk_id"]: i + 1 for i,
              doc in enumerate(bm25.invoke(query_text))}

    candidates = []
    for doc in lc_docs:
        cid = doc.metadata["chunk_id"]
        if cid not in d_rank and cid not in s_rank:
            continue
        m = doc.metadata
        mats = [x for x in m.get("materials_str", "").split(",") if x]
        if filter_material and filter_material != "All" and filter_material.lower() not in mats:
            continue
        if filter_source and filter_source != "All" and filter_source.lower() not in m.get("source", "").lower():
            continue

        rrf = (1.0 / (60 + d_rank.get(cid, total))) + \
            (1.0 / (60 + s_rank.get(cid, total)))
        candidates.append({
            "chunk": {"chunk_id": cid, "text": doc.page_content, "metadata": {
                "source": m.get("source", ""), "section": m.get("section", ""),
                "materials": mats, "image_path": m.get("image_path") or None
            }},
            "vector_score": round(d_score.get(cid, 0.0), 4), "rrf_score": round(rrf, 5)
        })

    candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    top_c = candidates[:max(6, top_k + 1)]
    if not top_c:
        return []

    logits = np.array(reranker.predict(
        [[query_text, c["chunk"]["text"]] for c in top_c], batch_size=16, show_progress_bar=False))
    max_rrf = max(c["rrf_score"] for c in top_c) or 1.0
    for i, c in enumerate(top_c):
        prob = float(1.0 / (1.0 + np.exp(-float(logits[i]))))
        c.update({"rerank_score": round(float(logits[i]), 4), "confidence_prob": round(prob, 4),
                  "final_score": round(0.5 * (c["rrf_score"] / max_rrf) + 0.5 * prob, 4)})

    top_c.sort(key=lambda x: x["final_score"], reverse=True)
    return top_c[:top_k]


def get_indexed_sources():
    return get_corpus_stats_fast()[1]


def get_chunks_by_source(source_name, max_chunks=9):
    if not os.path.exists("chunks_data.pkl"):
        return []
    matched = [c for c in joblib.load(
        "chunks_data.pkl") if c["metadata"].get("source") == source_name]
    if len(matched) <= max_chunks:
        return matched
    return [matched[i] for i in np.linspace(0, len(matched) - 1, max_chunks, dtype=int)]


def extract_cropped_figures_from_page(page, doc_title, page_num):
    import fitz
    from PIL import Image, ImageFilter

    pw, ph = page.rect.width, page.rect.height
    p_area = pw * ph
    text_blocks = page.get_text("blocks")

    tiles = [fitz.Rect(i["bbox"]) &
             page.rect for i in page.get_image_info(xrefs=True)]
    tiles = [r for r in tiles if not r.is_empty and r.width >=
             30 and r.height >= 5 and (r.width * r.height) / p_area <= 0.65]

    changed = True
    while changed:
        changed = False
        merged, used = [], [False] * len(tiles)
        for i in range(len(tiles)):
            if used[i]:
                continue
            curr = tiles[i]
            for j in range(i + 1, len(tiles)):
                if not used[j]:
                    touch = fitz.Rect(curr.x0 - 8, curr.y0 -
                                      12, curr.x1 + 8, curr.y1 + 12)
                    u = curr | tiles[j]
                    if not (touch & tiles[j]).is_empty and (u.width * u.height) / p_area <= 0.42:
                        curr, used[j], changed = u, True, True
            merged.append(curr)
        tiles = merged

    candidates = [r for r in tiles if r.width >= 58 and r.height >=
                  48 and 0.006 <= (r.width * r.height) / p_area <= 0.60]

    try:
        v_rects = [fitz.Rect(d["rect"]) &
                   page.rect for d in page.get_drawings()]
        v_rects = [r for r in v_rects if not r.is_empty and (r.width * r.height) / p_area <= 0.10
                   and not (r.width > pw * 0.4 and r.height < 16) and not (r.height > ph * 0.4 and r.width < 16)]
        clusters = []
        for dr in v_rects:
            pad = fitz.Rect(dr.x0 - 10, dr.y0 - 10, dr.x1 + 10, dr.y1 + 10)
            hit = False
            for c in clusters:
                if not (pad & c["r"]).is_empty:
                    c["r"] |= dr
                    c["n"] += 1
                    hit = True
                    break
            if not hit:
                clusters.append({"r": dr, "n": 1})
        for c in clusters:
            vr = c["r"]
            if c["n"] >= 8 and vr.width >= 80 and vr.height >= 55 and 0.010 <= (vr.width * vr.height) / p_area <= 0.40:
                box_txt = page.get_text("text", clip=vr).strip()
                if not any(b in box_txt for b in ["•", "→"]) and len(box_txt.split()) <= 32:
                    if not any(not (vr & ex).is_empty for ex in candidates):
                        candidates.append(vr)
    except Exception:
        pass

    candidates.sort(key=lambda r: (round(r.y0 / 45.0), r.x0))
    saved_paths, seen_sigs = [], []

    for r in candidates:
        cx = (r.x0 + r.x1) / 2.0
        qx0 = (pw / 2.0 + 3) if (pw > ph and r.width <
                                 pw * 0.48 and cx >= pw / 2.0) else 0
        qx1 = (pw / 2.0 - 3) if (pw > ph and r.width <
                                 pw * 0.48 and cx < pw / 2.0) else pw
        crop_r = fitz.Rect(max(qx0, r.x0 - 6), max(0, r.y0 - 6),
                           min(qx1, r.x1 + 6), min(ph, r.y1 + 6))

        for tb in text_blocks:
            tr, ts = fitz.Rect(tb[:4]), tb[4].strip()
            if (tr.height >= 14 and len(ts) >= 6) and (r & tr).is_empty and not (crop_r & tr).is_empty:
                if tr.y1 <= r.y0 + 10:
                    crop_r.y0 = max(crop_r.y0, tr.y1 + 2)

        thumb = page.get_pixmap(clip=crop_r, dpi=15, colorspace=fitz.csGRAY)
        arr = np.frombuffer(thumb.samples, dtype=np.uint8)
        if len(arr) < 64:
            continue
        sig = arr[np.linspace(0, len(arr) - 1, 64, dtype=int)
                  ].astype(np.float32)
        asp = crop_r.width / max(crop_r.height, 1.0)
        if any(abs(asp - pa) < 0.18 and np.mean(np.abs(sig - ps)) < 14.0 for ps, pa in seen_sigs):
            continue

        seen_sigs.append((sig, asp))
        fig_num = len(saved_paths) + 1
        img_path = os.path.normpath(os.path.join(
            "extracted_figures", f"{doc_title}_Page_{page_num}_Fig_{fig_num}.png"))

        zoom = max(3.0, min(6.0, 1350.0 / max(crop_r.width, 1.0)))
        pix = page.get_pixmap(matrix=fitz.Matrix(
            zoom, zoom), clip=crop_r, alpha=False)
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).filter(
            ImageFilter.UnsharpMask(radius=1.1, percent=105, threshold=3)
        ).save(img_path, format="PNG")

        saved_paths.append(img_path)
        if len(saved_paths) == 8:
            break

    return saved_paths


def ingest_uploaded_pdf_to_db(pdf_bytes, filename, progress_callback=None):
    import fitz
    from step3_chunk_and_tag import chunk_single_markdown_file

    for d in ["pdfs", "parsed_docs", "extracted_figures"]:
        os.makedirs(d, exist_ok=True)

    pdf_path = os.path.join("pdfs", filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    doc_title = filename.replace(".pdf", "")
    md_filename = f"{doc_title}.md"
    for old_f in os.listdir("extracted_figures"):
        if old_f.startswith(doc_title):
            try:
                os.remove(os.path.join("extracted_figures", old_f))
            except Exception:
                pass

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_records, vlm_tasks = [], []
    fig_cache = load_figure_caption_cache()

    for idx in range(total_pages):
        p_num = idx + 1
        if progress_callback:
            progress_callback(
                p_num, total_pages, f"Extracting text & HD figures on page {p_num}/{total_pages}...")
        page = doc[idx]
        text = page.get_text("text").strip()
        figs = extract_cropped_figures_from_page(page, doc_title, p_num)
        for f_idx, f_path in enumerate(figs, start=1):
            if f_path not in fig_cache:
                vlm_tasks.append((f_path, doc_title, p_num, f_idx, text))
        page_records.append((p_num, text, figs))
    doc.close()

    if vlm_tasks:
        if progress_callback:
            progress_callback(total_pages, total_pages,
                              f"Inspecting {len(vlm_tasks)} figures with Vision AI...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            for f_path, res_meta in pool.map(inspect_figure_with_vision_ai, vlm_tasks):
                if res_meta:
                    fig_cache[f_path] = res_meta
        with open(FIG_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(fig_cache, f, indent=2)

    md_pages = [f"# {doc_title}\n"]
    for p_num, text, figs in page_records:
        snap = f"\n**[PAGE_SNAPSHOT: {' | '.join(figs)}]**\n" if figs else ""
        descs = [
            f"[Visual Figure on Page {p_num} — {get_figure_metadata(p)['title']}: {get_figure_metadata(p)['explanation']}]" for p in figs]
        v_block = ("\n" + "\n".join(descs) + "\n") if descs else ""
        if len(text) >= 20 or snap:
            md_pages.append(f"\n## Page {p_num}\n{snap}{v_block}\n{text}\n")

    with open(os.path.join("parsed_docs", md_filename), "w", encoding="utf-8") as f:
        f.write("\n".join(md_pages))

    existing = [c for c in joblib.load(
        "chunks_data.pkl") if c["metadata"]["source"] != doc_title]
    next_id = max([c["chunk_id"] for c in existing], default=-1) + 1
    new_chunks = chunk_single_markdown_file(
        md_filename, start_chunk_id=next_id)
    joblib.dump(existing + new_chunks, "chunks_data.pkl")
    build_or_load_database(force_reload=True)
    return doc_title


if __name__ == "__main__":
    for pdf_f in os.listdir("pdfs"):
        if pdf_f.endswith(".pdf") and "NASA_" not in pdf_f:
            with open(os.path.join("pdfs", pdf_f), "rb") as f:
                ingest_uploaded_pdf_to_db(f.read(), pdf_f)
    build_or_load_database(force_reload=True)
    print("Indexed Documents in ChromaDB:", get_indexed_sources())
