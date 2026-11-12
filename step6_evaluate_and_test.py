import json
from step1_shap_bridge import predict_and_create_rag_query
from step4_hybrid_retriever import (
    build_or_load_database,
    doc_to_chunk_dict,
    search_failure_knowledge_base
)
from step5_rag_generator import ask_metallurgy_question, call_llm_with_fallback

GOLDEN_EVAL_QUESTIONS = [
    {
        "question": "What caused the crack in the AISI 4140 steel International Space Station (ISS) payload canister rail at Kennedy Space Center?",
        "expected_source": "NASA_Kennedy_Space_Center_Failure_Analysis",
        "expected_keyword": "weld"
    },
    {
        "question": "In the NASA study of 230 laboratory reports on failed aircraft metal components (Table 3), how many failures were caused by fatigue?",
        "expected_source": "NASA_Aircraft_Metal_Fatigue_Failures",
        "expected_keyword": "141"
    },
    {
        "question": "What does Figure 8 on Page 19 of the NASA aircraft metal fatigue report show?",
        "expected_source": "NASA_Aircraft_Metal_Fatigue_Failures",
        "expected_keyword": "wing attachment fitting"
    },
    {
        "question": "In the NASA aircraft fatigue study, how many fatigue failures were attributed to improper maintenance versus fabrication defects?",
        "expected_source": "NASA_Aircraft_Metal_Fatigue_Failures",
        "expected_keyword": "52"
    },
    {
        "question": "What did SEM/EDS analysis reveal on the fracture surface of the AISI 4140 payload canister rail at Kennedy Space Center?",
        "expected_source": "NASA_Kennedy_Space_Center_Failure_Analysis",
        "expected_keyword": "nickel plating"
    },
    {
        "question": "Why are fatigue cracks often initiated at fillets, bolt holes, or keyways in metallic aircraft components?",
        "expected_source": "NASA_Aircraft_Metal_Fatigue_Failures",
        "expected_keyword": "stress"
    },
    {
        "question": "Which ASTM standard is used for rating non-metallic inclusions (Type A sulfide, Type B alumina) in steel?",
        "expected_source": "ASTM_and_Steel_Fatigue_Reference",
        "expected_keyword": "astm e45"
    },
    {
        "question": "What causes subsurface fish-eye fatigue fracture in high-strength Cr-Mo alloy steel gear shafts?",
        "expected_source": "Historical_Steel_Failure_Case_Studies",
        "expected_keyword": "alumina"
    },
    {
        "question": "Why does an uncarburized 0.35% C plain carbon steel shaft tempered at 600°C fail at 380 MPa under rotating bending?",
        "expected_source": "Historical_Steel_Failure_Case_Studies",
        "expected_keyword": "spheroidized cementite"
    },
    {
        "question": "What is the role of Molybdenum (Mo) in preventing intergranular fracture and temper embrittlement at 550°C?",
        "expected_source": "Historical_Steel_Failure_Case_Studies",
        "expected_keyword": "phosphorus"
    },
    {
        "question": "Which ASTM standard governs force-controlled constant amplitude rotating-bending fatigue testing at 10^7 cycles?",
        "expected_source": "ASTM_and_Steel_Fatigue_Reference",
        "expected_keyword": "astm e466"
    },
    {
        "question": "How does a low forging reduction ratio (RedRatio < 500) degrade rotating bending fatigue endurance?",
        "expected_source": "ASTM_and_Steel_Fatigue_Reference",
        "expected_keyword": "microporosity"
    }
]


# Baseline 1: Pure LangChain Chroma Vector Search
def search_pure_chroma(query_text, top_k=3):
    vectorstore, _, _ = build_or_load_database()
    docs = vectorstore.similarity_search(query_text, k=top_k)
    return [doc_to_chunk_dict(d) for d in docs]


# Baseline 2: Pure LangChain BM25Retriever Keyword Search
def search_pure_bm25(query_text, top_k=3):
    _, bm25_retriever, _ = build_or_load_database()
    docs = bm25_retriever.invoke(query_text)[:top_k]
    return [doc_to_chunk_dict(d) for d in docs]


def evaluate_chunk_list(chunk_list, exp_src, exp_kw):
    for rank_idx, c in enumerate(chunk_list):
        if exp_src in c["metadata"]["source"] and exp_kw in c["text"].lower():
            return 1, 1.0 / (rank_idx + 1)
    return 0, 0.0


def run_ablation_evaluation():
    print("\n=== STAGE 6: LANGCHAIN CHROMADB + BM25 + RRF BENCHMARK ===")
    total = len(GOLDEN_EVAL_QUESTIONS)

    chroma_hits, chroma_rr = 0, 0.0
    bm25_hits, bm25_rr = 0, 0.0
    hybrid_hits, hybrid_rr = 0, 0.0

    for i, item in enumerate(GOLDEN_EVAL_QUESTIONS, start=1):
        q = item["question"]
        exp_src = item["expected_source"]
        exp_kw = item["expected_keyword"].lower()

        hit_c, rr_c = evaluate_chunk_list(
            search_pure_chroma(q, top_k=3), exp_src, exp_kw)
        chroma_hits += hit_c
        chroma_rr += rr_c

        hit_b, rr_b = evaluate_chunk_list(
            search_pure_bm25(q, top_k=3), exp_src, exp_kw)
        bm25_hits += hit_b
        bm25_rr += rr_b

        hybrid_res = search_failure_knowledge_base(q, top_k=3)
        hybrid_chunks = [r["chunk"] for r in hybrid_res]
        hit_h, rr_h = evaluate_chunk_list(hybrid_chunks, exp_src, exp_kw)
        hybrid_hits += hit_h
        hybrid_rr += rr_h

        rank_str = f"Rank #{int(round(1.0 / rr_h))}" if hit_h else "Miss"
        print(f" Q{i:02d}: [Hybrid RRF: {rank_str}] -> {q[:65]}...")

    print("\n--- Evaluating LLM Generation Faithfulness & Answer Relevancy ---")
    sample_eval_qs = [GOLDEN_EVAL_QUESTIONS[0]["question"],
                      GOLDEN_EVAL_QUESTIONS[1]["question"], GOLDEN_EVAL_QUESTIONS[7]["question"]]
    faithful_count = 0

    for sq in sample_eval_qs:
        ans_data = ask_metallurgy_question(sq)
        ans_text = ans_data["answer"]
        has_citation = "[" in ans_text and "]" in ans_text
        judge_prompt = (
            f"Question: {sq}\nGenerated Answer: {ans_text}\n"
            "Does this answer directly address the question and cite evidence sources in brackets? Reply ONLY 'YES' or 'NO'."
        )
        verdict, _ = call_llm_with_fallback(
            "You are a strict RAG evaluator.", judge_prompt)
        if "YES" in verdict.upper() and has_citation:
            faithful_count += 1

    faithfulness_score = round((faithful_count / len(sample_eval_qs)) * 100, 1)

    metrics = {
        "total_questions": total,
        "pure_chroma_recall_at_3": round((chroma_hits / total) * 100, 1),
        "pure_chroma_mrr": round((chroma_rr / total) * 100, 1),
        "pure_bm25_recall_at_3": round((bm25_hits / total) * 100, 1),
        "pure_bm25_mrr": round((bm25_rr / total) * 100, 1),
        "hybrid_rrf_recall_at_3": round((hybrid_hits / total) * 100, 1),
        "hybrid_rrf_mrr": round((hybrid_rr / total) * 100, 1),
        "generation_faithfulness": faithfulness_score
    }

    with open("eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n--- FINAL ABLATION RESULTS (LangChain ChromaDB + BM25 + RRF) ---")
    print(
        f"1. Pure LangChain BM25Retriever  -> Recall@3: {metrics['pure_bm25_recall_at_3']}% | MRR: {metrics['pure_bm25_mrr']}%")
    print(
        f"2. Pure LangChain Chroma Vector  -> Recall@3: {metrics['pure_chroma_recall_at_3']}% | MRR: {metrics['pure_chroma_mrr']}%")
    print(
        f"3. Hybrid RRF + Cross-Encoder    -> Recall@3: {metrics['hybrid_rrf_recall_at_3']}% | MRR: {metrics['hybrid_rrf_mrr']}%")
    print(
        f"4. LLM Answer Faithfulness Score -> {metrics['generation_faithfulness']}% (Grounded with Citations)")
    return metrics


# =====================================================================
# AUTOMATED PYTEST SUITE (Run via `pytest step6_evaluate_and_test.py -v`)
# =====================================================================

def test_ml_shap_bridge_output():
    sample = {
        "NT": 885, "THT": 855, "THt": 30, "THQCr": 8,
        "CT": 30, "Ct": 0, "Dt": 0, "QmT": 30, "TT": 600, "Tt": 60,
        "C": 0.35, "Si": 0.25, "Mn": 0.75, "Ni": 0.05, "Cr": 0.15, "Cu": 0.05, "Mo": 0.0,
        "RedRatio": 500.0, "dA": 0.08, "dB": 0.05, "dC": 0.02
    }
    res = predict_and_create_rag_query(sample)
    assert 200.0 < res["predicted_mpa"] < 1200.0
    assert len(res["top_4_shap_drivers"]) > 0


def test_langchain_chromadb_loaded():
    vectorstore, _, lc_docs = build_or_load_database()
    assert vectorstore._collection.count() == len(lc_docs)
    assert len(lc_docs) > 50


def test_rrf_hybrid_retriever_generalization():
    results = search_failure_knowledge_base(
        "AISI 4140 steel payload canister rail crack Kennedy Space Center", top_k=3)
    assert len(results) == 3
    assert results[0]["final_score"] > 0.0
    assert 0.0 <= results[0]["confidence_prob"] <= 1.0


def test_confidence_gate_blocks_off_topic():
    results = search_failure_knowledge_base(
        "How to bake a chocolate birthday cake with frosting", top_k=3)
    assert results[0]["rerank_score"] < -3.5


if __name__ == "__main__":
    run_ablation_evaluation()
