import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# LOAD FILES
# ============================================================

# llm_verifier_df = pd.read_csv("llm_verifier_results.csv")
# baseline_df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/annotated_500_llm_labels.csv")

llm_verifier_df = pd.read_csv("llm_verifier_results_wiki.csv")
baseline_df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/annotated_500_llm_labels.csv")

# ============================================================
# NORMALIZE LABELS
# ============================================================

baseline_df["label"] = (
    baseline_df["label"]
    .str.upper()
    .str.strip()
    .str.replace(" ", "_")
)

baseline_df["llm_label"] = (
    baseline_df["llm_label"]
    .str.upper()
    .str.strip()
    .str.replace(" ", "_")
)

llm_verifier_df["llm_verifier_label"] = (
    llm_verifier_df["llm_verifier_label"]
    .str.upper()
    .str.strip()
    .str.replace(" ", "_")
)

# ============================================================
# MERGE
# ============================================================

merged = baseline_df.merge(
    llm_verifier_df[["claim", "llm_verifier_label", "has_evidence"]],
    on="claim",
    how="inner"
).drop_duplicates(subset="claim", keep="first")

print(f"Claims evaluated: {len(merged)}")

gold = merged["label"]
baseline_pred = merged["llm_label"]
kg_llm_pred = merged["llm_verifier_label"]

# ============================================================
# ACCURACY COMPARISON
# ============================================================

baseline_acc = accuracy_score(gold, baseline_pred)
kg_llm_acc = accuracy_score(gold, kg_llm_pred)

print("\n=== ACCURACY COMPARISON ===")
print(f"Baseline LLM accuracy:     {baseline_acc:.4f}")
print(f"KG + Open LLM accuracy:    {kg_llm_acc:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n=== BASELINE LLM REPORT ===")
print(classification_report(
    gold,
    baseline_pred,
    labels=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
    zero_division=0
))

print("\n=== KG + OPEN LLM VERIFIER REPORT ===")
print(classification_report(
    gold,
    kg_llm_pred,
    labels=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
    zero_division=0
))

# ============================================================
# HALLUCINATION RESCUE RATE
# ============================================================

baseline_wrong = merged[merged["llm_label"] != merged["label"]].copy()

rescued = baseline_wrong[
    baseline_wrong["llm_verifier_label"] == baseline_wrong["label"]
]

if len(baseline_wrong) > 0:
    rescue_rate = len(rescued) / len(baseline_wrong)
else:
    rescue_rate = 0

print("\n=== HALLUCINATION RESCUE ===")
print(f"Baseline wrong cases:      {len(baseline_wrong)}")
print(f"Rescued by KG + Open LLM:  {len(rescued)}")
print(f"Rescue rate:               {rescue_rate:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

cm = confusion_matrix(gold, kg_llm_pred, labels=labels)

print("\n=== KG + OPEN LLM CONFUSION MATRIX ===")
print(pd.DataFrame(cm, index=[f"gold_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]))

# ============================================================
# SAVE COMPARISON
# ============================================================

# merged.to_csv("kg_open_llm_vs_baseline_comparison_2.csv", index=False)
# print("\nSaved → kg_open_llm_vs_baseline_comparison_2.csv")

merged.to_csv("wiki_vs_baseline_comparison_2.csv", index=False)
print("\nSaved → wiki_vs_baseline_comparison_2.csv")
