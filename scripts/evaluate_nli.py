import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# ── load ───────────────────────────────────────────────────────
nli_df = pd.read_csv("nli_results_verbalized.csv")
llm_df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/annotated_500_llm_labels.csv")

# ── normalize ──────────────────────────────────────────────────
llm_df["label"]     = llm_df["label"].str.upper().str.strip().str.replace(" ", "_")
llm_df["llm_label"] = llm_df["llm_label"].str.upper().str.strip().str.replace(" ", "_")

# ── merge ──────────────────────────────────────────────────────
merged = llm_df.merge(
    nli_df[["claim", "nli_entailment", "nli_contradiction", "nli_neutral", "nli_confidence", "has_evidence"]],
    on="claim", how="inner"
).drop_duplicates(subset="claim", keep="first")

print(f"Claims: {len(merged)}")

# ── re-apply verdict with different thresholds ─────────────────
def apply_threshold(row, threshold):
    # no evidence → always NEI
    if not row["has_evidence"]:
        return "NOT_ENOUGH_INFO"

    ent  = row["nli_entailment"]
    con  = row["nli_contradiction"]
    neu  = row["nli_neutral"]

    # handle NaN (no evidence rows)
    if pd.isna(ent):
        return "NOT_ENOUGH_INFO"

    max_score = max(ent, con, neu)
    if max_score < threshold:
        return "NOT_ENOUGH_INFO"

    if ent == max_score:
        return "SUPPORTS"
    elif con == max_score:
        return "REFUTES"
    else:
        return "NOT_ENOUGH_INFO"

# ── sweep thresholds ───────────────────────────────────────────
thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
gold = merged["label"]

print(f"\n{'Threshold':<12} {'Accuracy':<12} {'SUPPORTS F1':<14} {'REFUTES F1':<14} {'NEI F1':<10} {'NEI predicted'}")
print("-" * 80)

for t in thresholds:
    merged[f"pred_{t}"] = merged.apply(lambda r: apply_threshold(r, t), axis=1)
    pred = merged[f"pred_{t}"]
    acc  = accuracy_score(gold, pred)
    report = classification_report(
        gold, pred,
        labels=["SUPPORTS","REFUTES","NOT_ENOUGH_INFO"],
        output_dict=True, zero_division=0
    )
    sup_f1 = report["SUPPORTS"]["f1-score"]
    ref_f1 = report["REFUTES"]["f1-score"]
    nei_f1 = report["NOT_ENOUGH_INFO"]["f1-score"]
    nei_count = (pred == "NOT_ENOUGH_INFO").sum()
    print(f"{t:<12} {acc:<12.4f} {sup_f1:<14.4f} {ref_f1:<14.4f} {nei_f1:<10.4f} {nei_count}")
