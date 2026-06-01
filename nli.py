import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── load model ─────────────────────────────────────────────────
print("Loading DeBERTa NLI model...")
model_name = "cross-encoder/nli-deberta-v3-small"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# label order for this model: contradiction, entailment, neutral
# we'll confirm this from model.config.id2label
print(f"Label mapping: {model.config.id2label}")

# ── load unified CSV ───────────────────────────────────────────
# df = pd.read_csv("unified_claims_for_nli.csv")
# df = pd.read_csv("unified_claims_for_nli_verbalized.csv")

print(f"\nTotal claims: {len(df)}")
print(f"NLI-ready:    {df['has_evidence'].sum()}")
print(f"No evidence:  {(~df['has_evidence']).sum()}")

# ── NLI for one pair ───────────────────────────────────────────
def run_nli(premise, hypothesis):
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, 3)
    probs = F.softmax(logits, dim=-1)[0]         # (3,)
    return probs                                  # tensor of 3 scores

def map_verdict(probs, id2label, threshold=0.5):
    max_idx   = probs.argmax().item()
    max_score = probs[max_idx].item()

    if max_score < threshold:
        return "NOT_ENOUGH_INFO", max_score

    label = id2label[max_idx].lower()
    if label == "entailment":
        return "SUPPORTS", max_score
    elif label == "contradiction":
        return "REFUTES", max_score
    else:
        return "NOT_ENOUGH_INFO", max_score

# ── run NLI on all claims ──────────────────────────────────────
print("\nRunning NLI...")
id2label = model.config.id2label

entailment_scores    = []
contradiction_scores = []
neutral_scores       = []
nli_verdicts         = []
nli_confidences      = []

total = len(df)

for i, row in df.iterrows():
    if i % 200 == 0:
        print(f"  Processing {i}/{total}...")

    # no evidence → skip NLI
    if not row["has_evidence"] or str(row["nli_premise_verbalized"]).strip() in ("", "None", "nan"):
        entailment_scores.append(None)
        contradiction_scores.append(None)
        neutral_scores.append(None)
        nli_verdicts.append("NOT_ENOUGH_INFO")
        nli_confidences.append(None)
        continue

    # probs = run_nli(str(row["nli_premise"]), str(row["claim"]))
    probs = run_nli(str(row["nli_premise_verbalized"]), str(row["claim"]))

    # extract scores by label name (safe — doesn't assume order)
    scores_dict = {id2label[j].lower(): probs[j].item() for j in range(len(probs))}

    entailment_scores.append(round(scores_dict.get("entailment", 0), 4))
    contradiction_scores.append(round(scores_dict.get("contradiction", 0), 4))
    neutral_scores.append(round(scores_dict.get("neutral", 0), 4))

    verdict, confidence = map_verdict(probs, id2label, threshold=0.5)
    nli_verdicts.append(verdict)
    nli_confidences.append(round(confidence, 4))

# ── attach results ─────────────────────────────────────────────
df["nli_entailment"]    = entailment_scores
df["nli_contradiction"] = contradiction_scores
df["nli_neutral"]       = neutral_scores
df["nli_verdict"]       = nli_verdicts
df["nli_confidence"]    = nli_confidences

# ── report ─────────────────────────────────────────────────────
print(f"\n=== NLI VERDICT DISTRIBUTION ===")
print(df["nli_verdict"].value_counts())

print(f"\n=== SAMPLE: SUPPORTS ===")
sup = df[df["nli_verdict"] == "SUPPORTS"][["claim","nli_premise","nli_verdict","nli_confidence"]].head(5)
print(sup.to_string(index=False))

print(f"\n=== SAMPLE: REFUTES ===")
ref = df[df["nli_verdict"] == "REFUTES"][["claim","nli_premise","nli_verdict","nli_confidence"]].head(5)
print(ref.to_string(index=False))

# ── save ───────────────────────────────────────────────────────
df.to_csv("nli_results_verbalized.csv", index=False)
print(f"\nSaved → nli_results_verbalized.csv")