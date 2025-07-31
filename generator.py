import re
from functools import lru_cache
from typing import List, Dict, Any

import torch
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# model names & thresholds
OUTLINE_MODEL   = "gpt2"
REFINE_MODEL    = "distilgpt2"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
LOW_RISK_THRESH = 5.0

# device setup for gpu/cpu
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32


@lru_cache(maxsize=1)
def load_outline_gen():
    tok   = AutoTokenizer.from_pretrained(OUTLINE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        OUTLINE_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    return pipeline("text-generation", model=model, tokenizer=tok, device=DEVICE)


@lru_cache(maxsize=1)
def load_refine_gen():
    tok   = AutoTokenizer.from_pretrained(REFINE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REFINE_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    return pipeline("text-generation", model=model, tokenizer=tok, device=DEVICE)


@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def build_vector_index(
    issues: List[Dict[str, Any]]
) -> faiss.IndexFlatL2:
    texts = [f"{i['Summary']} {i['Description']}" for i in issues]
    emb   = load_embedder().encode(texts, convert_to_numpy=True)
    dim   = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index


def semantic_search(
    issue: Dict[str, Any],
    index: faiss.IndexFlatL2,
    issues_list: List[Dict[str, Any]],
    top_k: int = 2
) -> List[Dict[str, Any]]:
    query = f"{issue['Summary']} {issue['Description']}"
    q_emb = load_embedder().encode([query], convert_to_numpy=True)
    _, idxs = index.search(q_emb, top_k)
    return [issues_list[i] for i in idxs[0] if i < len(issues_list)]


def make_outline_prompt(
    issue: Dict[str, Any],
    similar: List[Dict[str, Any]]
) -> str:
    prompt = (
        "You are a QA test-case generator. Write each step as a complete, imperative "
        "sentence describing exactly what the tester does.\n\n"
        "Here are past similar issues:\n"
    )
    for s in similar:
        prompt += f"- {s['Issue key']}: {s['Summary']}\n  {s['Description']}\n\n"
    prompt += (
        "Based on these examples, create a concise test-case outline.\n"
        "Each step must be a full, actionable sentence (e.g. "
        "\"1. Launch the application by navigating to the home page.\").\n\n"
        f"Issue Key: {issue['Issue key']}\n"
        f"Summary:   {issue['Summary']}\n"
        f"Description:{issue['Description']}\n\n"
        "Outline format example:\n"
        "Title: Verify registration of commercial appliances\n"
        "Steps:\n"
        "1. Launch the application by entering the URL in a browser.\n"
        "2. Navigate to the “Commercial Appliances” section.\n"
        "3. Select an appliance to register.\n"
        "…\n\n"
        "Now generate:\n"
        "Title:\n"
        "Steps:\n"
        "1.\n"
        "2.\n"
        "3.\n"
    )
    return prompt


def make_refine_prompt(
    issue: Dict[str, Any],
    outline: str,
    risk: float
) -> str:
    return (
        "You are a QA expert. Refine the outline below so that for every step:\n"
        "  • the step is written in a full sentence, and\n"
        "  • it is immediately followed by “✔ Expected: <the exact outcome>”.\n"
        "Finally, include a “Risk Evaluation” paragraph referencing the given risk score.\n\n"
        f"Issue Key:\n{issue['Issue key']}\n\n"
        "Outline:\n"
        f"{outline}\n\n"
        f"Risk Score: {risk}\n\n"
        "Please produce exactly this structure:\n\n"
        "Test Case Title: <a short, descriptive title>\n\n"
        "Steps and Expected Outcomes:\n"
        "1. <full step sentence>    ✔ Expected: <the validation>\n"
        "2. <full step sentence>    ✔ Expected: <the validation>\n"
        "...\n\n"
        "Risk Evaluation: <your evaluation here>\n"
    )


def parse_test_case_text(
    text: str,
    issue_key: str
) -> List[Dict[str, Any]]:
    """
    Parses lines like '1. action ✔ Expected: outcome' into dict rows.
    """
    rows, in_steps = [], False
    for line in text.splitlines():
        txt = line.strip()
        if not in_steps and txt.lower().startswith("steps"):
            in_steps = True
            continue
        if in_steps and txt.lower().startswith("risk evaluation"):
            break
        m = re.match(r"(\d+)\.\s*(.+?)\s*✔\s*Expected:\s*(.+)", txt)
        if m:
            no, act, exp = m.groups()
            rows.append({
                "Issue key":        issue_key,
                "Step No":          int(no),
                "Action":           act.strip(),
                "Expected Outcome": exp.strip()
            })
    return rows


def generate_test_case(
    issue: Dict[str, Any],
    risk_score: float,
    index: faiss.IndexFlatL2,
    issues_list: List[Dict[str, Any]],
    outline_gen=None,
    refine_gen=None
) -> str:
    """
    Returns a markdown string: Test Case Title + numbered steps with ✔ Expected.
    """
    if outline_gen is None:
        outline_gen = load_outline_gen()
    if refine_gen is None:
        refine_gen  = load_refine_gen()

    # 1) find similar examples
    similar = semantic_search(issue, index, issues_list)

    # 2) generate outline
    prompt_o = make_outline_prompt(issue, similar)
    with torch.no_grad():
        out = outline_gen(
            prompt_o,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            return_full_text=False
        )
    outline = out[0]["generated_text"].strip()

    #refine
    if risk_score >= LOW_RISK_THRESH:
        prompt_r = make_refine_prompt(issue, outline, risk_score)
        with torch.no_grad():
            ref = refine_gen(
                prompt_r,
                max_new_tokens=150,
                do_sample=False,
                return_full_text=False
            )
        return ref[0]["generated_text"].strip()

    return outline
