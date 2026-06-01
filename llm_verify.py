import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# old (kg only, no rag)
# INPUT_FILE = "unified_claims_for_nli_verbalized.csv"
# OUTPUT_FILE = "llm_verifier_results.csv"

# new
INPUT_FILE = "unified_claims_wiki_fallback.csv"
OUTPUT_FILE = "llm_verifier_results_wiki.csv"


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading open-source LLM verifier...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype=torch.float16,
#     device_map={"": "cuda:0"},
#     trust_remote_code=True
# )

model.eval()

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Load the 500 manually annotated claims
gold_df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/annotated_500_llm_labels.csv")

# Keep only claims that appear in the 500-label file
df = df[df["claim"].isin(gold_df["claim"])].copy()

# Remove duplicate claim rows
df = df.drop_duplicates(subset="claim", keep="first").copy()

print(f"Total claims after filtering to annotated sample: {len(df)}")


# Choose verbalized premise if available, otherwise fallback to normal premise
if "nli_premise_verbalized" in df.columns:
    premise_col = "nli_premise_verbalized"
else:
    premise_col = "nli_premise"

print(f"Using premise column: {premise_col}")

# ============================================================
# PROMPT
# ============================================================

def build_prompt(claim, evidence):
    return f"""
You are a strict fact verification model.

You must classify the CLAIM using ONLY the provided KG EVIDENCE.

Labels:
- SUPPORTS: the evidence clearly supports the claim.
- REFUTES: the evidence clearly contradicts the claim.
- NOT_ENOUGH_INFO: the evidence is missing, unrelated, ambiguous, or insufficient.

Important rules:
- Do not use outside knowledge.
- Do not guess.
- If the evidence does not directly verify the claim, choose NOT_ENOUGH_INFO.
- Output only one label: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.

KG EVIDENCE:
{evidence}

CLAIM:
{claim}

Final label:
""".strip()

# def build_prompt(claim, evidence):
#     return f"""
# You are a fact verification model.

# Classify the CLAIM using the KG EVIDENCE.

# Labels:
# - SUPPORTS: the evidence supports the claim directly OR through clear semantic equivalence.
# - REFUTES: the evidence contradicts the claim directly OR through clear semantic mismatch.
# - NOT_ENOUGH_INFO: the evidence is missing, unrelated, or cannot help verify the claim.

# Important rules:
# - Use only the KG evidence.
# - You may use normal language understanding for synonyms and paraphrases.
# - Treat equivalent meanings as SUPPORTS.
#   Examples:
#   novelist ≈ author
#   footballer ≈ soccer player
#   actor ≈ performer
#   headquartered in ≈ based in
#   country of citizenship ≈ nationality
# - Do NOT choose NOT_ENOUGH_INFO just because the wording is different.
# - Choose NOT_ENOUGH_INFO only when the evidence is truly missing or unrelated.

# KG EVIDENCE:
# {evidence}

# CLAIM:
# {claim}

# Output only one label: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.
# """.strip()

def extract_label(text):
    text = text.upper()

    # direct exact label match
    for label in LABELS:
        if label in text:
            return label

    # common fallback normalization
    if "NOT ENOUGH" in text or "INSUFFICIENT" in text or "UNKNOWN" in text:
        return "NOT_ENOUGH_INFO"

    if "SUPPORT" in text or "ENTAILED" in text or "ENTAILMENT" in text:
        return "SUPPORTS"

    if "REFUTE" in text or "CONTRADICT" in text or "CONTRADICTION" in text:
        return "REFUTES"

    return "NOT_ENOUGH_INFO"


# ============================================================
# LLM INFERENCE
# ============================================================

def run_llm_verifier(claim, evidence):
    prompt = build_prompt(claim, evidence)

    messages = [
        {
            "role": "system",
            "content": "You are a strict evidence-based fact verification assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return extract_label(response), response


# ============================================================
# RUN VERIFICATION
# ============================================================

llm_labels = []
raw_outputs = []

for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"Processing {i}/{len(df)}...")

    claim = str(row["claim"])

    has_evidence = row.get("has_evidence", True)

    evidence = row.get(premise_col, "")

    if pd.isna(evidence) or str(evidence).strip() in ["", "None", "nan"]:
        llm_labels.append("NOT_ENOUGH_INFO")
        raw_outputs.append("NO_EVIDENCE")
        continue

    if has_evidence is False:
        llm_labels.append("NOT_ENOUGH_INFO")
        raw_outputs.append("NO_EVIDENCE")
        continue

    label, raw = run_llm_verifier(claim, evidence)

    llm_labels.append(label)
    raw_outputs.append(raw)

df["llm_verifier_label"] = llm_labels
df["llm_verifier_raw_output"] = raw_outputs

print("\n=== LLM VERIFIER LABEL DISTRIBUTION ===")
print(df["llm_verifier_label"].value_counts())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved → {OUTPUT_FILE}")