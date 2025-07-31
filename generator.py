#!/usr/bin/env python
import torch
from functools import lru_cache
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    __version__ as _TFM_VERSION
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print(f"Transformers version: {_TFM_VERSION}")

# cap CPU threads to reduce overhead
torch.set_num_threads(2)

# Models
GPT2_MODEL            = "gpt2"
GPTJ_MODEL            = "EleutherAI/gpt-j-6B"
EMBEDDING_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
LOW_RISK_THRESHOLD    = 5.0

# pick device and dtype
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# Caches
@lru_cache(maxsize=1)
def load_outline_gen():
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    model     = AutoModelForCausalLM.from_pretrained(
        GPT2_MODEL,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=60,
        num_return_sequences=1,
        return_full_text=False
    )

@lru_cache(maxsize=1)
def load_refine_gen():
    tokenizer = AutoTokenizer.from_pretrained(GPTJ_MODEL)
    model     = AutoModelForCausalLM.from_pretrained(
        GPTJ_MODEL,
        torch_dtype=DTYPE,
        load_in_8bit=torch.cuda.is_available(),
        device_map="auto" if torch.cuda.is_available() else None
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        do_sample=False,
        max_new_tokens=150,
        return_full_text=False
    )

@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def build_vector_index(issues: list):
    """
    issues: list of dicts, each with 'Issue key', 'Summary', 'Description'
    Returns: faiss.IndexFlatL2, numpy.ndarray of embeddings, and the issues list
    """
    embedder = load_embedding_model()
    texts = [f"{iss['Summary']} {iss['Description']}" for iss in issues]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, issues

def semantic_search(issue: dict, index, issues_list: list, top_k: int = 3):
    """
    Finds top_k similar issues via cosine / L2 search.
    Returns: list of issue dicts.
    """
    embedder   = load_embedding_model()
    query_text = f"{issue['Summary']} {issue['Description']}"
    q_emb       = embedder.encode([query_text], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, top_k)
    return [issues_list[i] for i in idxs[0] if i < len(issues_list)]

def make_outline_prompt(issue: dict, similar_issues: list) -> str:
    """
    Builds a prompt that includes similar issues context
    """
    prompt = "Here are past similar issues and their contexts:\n"
    for sim in similar_issues:
        prompt += (
            f"- Issue Key: {sim['Issue key']}\n"
            f"  Summary: {sim['Summary']}\n"
            f"  Description: {sim['Description']}\n\n"
        )
    prompt += (
        "Based on these examples, create a concise test‐case outline\n"
        "for the following issue:\n\n"
        f"Issue Key: {issue['Issue key']}\n"
        f"Summary: {issue['Summary']}\n"
        f"Description: {issue['Description']}\n\n"
        "Outline format:\n"
        "Title: <short title>\n"
        "Steps:\n"
        "1. <first step>\n"
        "2. <second step>\n"
        "...\n"
    )
    return prompt

def make_refinement_prompt(issue: dict, outline: str, risk: float) -> str:
    return (
        "You are a QA expert. Refine the outline below by adding, for each step:\n"
        "  ✔ Expected: <expected outcome>\n"
        "Also add a final “Risk Evaluation” note referencing the risk score.\n\n"
        "ISSUE KEY:\n"
        f"{issue['Issue key']}\n\n"
        "OUTLINE:\n"
        f"{outline}\n\n"
        f"Risk Score: {risk}\n\n"
        "Respond exactly in this format:\n"
        "Test Case Title: <...>\n\n"
        "Steps and Expected Outcomes:\n"
        "1. <step>    ✔ Expected: <outcome>\n"
        "2. ...\n\n"
        "Risk Evaluation: <text>\n"
    )

def generate_test_case(
    issue: dict,
    risk_score: float,
    index,
    issues_list: list,
    outline_gen=None,
    refine_gen=None
) -> str:
    """
    Generates an outline + (optional) detailed test case.
    Uses semantic search to retrieve similar issues before prompting.
    """
    if outline_gen is None:
        outline_gen = load_outline_gen()
    if refine_gen is None:
        refine_gen = load_refine_gen()

    # 1) Retrieve similar issues for richer context
    similar = semantic_search(issue, index, issues_list, top_k=3)

    # 2) Generate the outline with context
    prompt = make_outline_prompt(issue, similar)
    outline = outline_gen(prompt)[0]["generated_text"].strip()

    # 3) If risk is low, skip refinement
    if risk_score < LOW_RISK_THRESHOLD:
        return outline

    # 4) Refine into full test case
    ref_prompt = make_refinement_prompt(issue, outline, risk_score)
    refined = refine_gen(ref_prompt)[0]["generated_text"].strip()
    return refined
