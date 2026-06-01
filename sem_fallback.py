import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── connect ────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "Kaythevu@4")
)

# ── load model ─────────────────────────────────────────────────
print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── load CSV ───────────────────────────────────────────────────
df = pd.read_csv("extracted_claims_v4_new.csv")
fallback_df = df[df["route"] == "semantic_search"].copy()

# ── Neo4j query ────────────────────────────────────────────────
def get_entity_triples(tx, subject):
    result = tx.run("""
        MATCH (e:Entity {name: $subject})-[r]->(v:Value)
        RETURN e.name AS entity, type(r) AS relation, v.name AS value
    """, subject=subject)
    return [(rec["entity"], rec["relation"], rec["value"]) for rec in result]

def fetch_neighborhood(subject):
    with driver.session() as session:
        return session.execute_read(get_entity_triples, subject.lower().strip())

# ── triple → sentence ──────────────────────────────────────────
def triple_to_sentence(entity, relation, value):
    rel_phrase = relation.replace("_", " ").lower()
    return f"{entity} {rel_phrase} {value}"

# ── semantic search for one claim ─────────────────────────────
def semantic_fallback(claim, triple_sentences, triples, top_k=3, threshold=0.4):
    # embed claim + triple sentences with batching
    claim_vec   = model.encode([claim], batch_size=32, show_progress_bar=False)
    triple_vecs = model.encode(triple_sentences, batch_size=32, show_progress_bar=False)

    # cosine similarity
    scores = cosine_similarity(claim_vec, triple_vecs)[0]

    # rank by score
    ranked = sorted(zip(scores, triple_sentences, triples), reverse=True)

    # filter by threshold
    top = [(score, sent, triple) for score, sent, triple in ranked[:top_k] if score >= threshold]

    return top if top else None

# ── run on all fallback claims with valid subjects ─────────────
print("Running semantic fallback...")
output = []
total = len(fallback_df)

for i, (_, row) in enumerate(fallback_df.iterrows()):
    if i % 100 == 0:
        print(f"  Processing {i}/{total}...")

    subject = row["subject"]

    # skip NaN subjects
    if not isinstance(subject, str) or subject.strip() == "":
        output.append({
            **row.to_dict(),           # take all orig cols from extracted_claims.csv and keep them
            "sem_top_triple": None,    # add new cols
            "sem_top_score":  None,
            "sem_evidence":   None,
            # "sem_verdict":    "NOT_ENOUGH_INFO"
        })
        continue

    # fetch KG neighborhood
    triples = fetch_neighborhood(subject)

    if not triples:
        output.append({
            **row.to_dict(),
            "sem_top_triple": None,
            "sem_top_score":  None,
            "sem_evidence":   None,
            # "sem_verdict":    "NOT_ENOUGH_INFO"
        })
        continue

    triple_sentences = [triple_to_sentence(*t) for t in triples]
    results = semantic_fallback(row["claim"], triple_sentences, triples, top_k=3, threshold=0.4)

    if results is None:
        top_triple = None
        top_score  = None
        evidence   = None
    else:
        top_score, top_sent, top_triple = results[0]
        evidence = " | ".join([s for _, s, _ in results])
        # ── NO VERDICT HERE — that's NLI's job ──

    output.append({
        **row.to_dict(),
        "sem_top_triple": f"{top_triple[0]} --[{top_triple[1]}]--> {top_triple[2]}" if top_triple else None,
        "sem_top_score":  round(float(top_score), 4) if top_score is not None else None,
        "sem_evidence":   evidence,
        # ── no sem_verdict column at all ──
    })

# ── save results ───────────────────────────────────────────────
output_df = pd.DataFrame(output)
output_df.to_csv("sem_fallback_tryyyy.csv", index=False)

print(f"\nDone!")
print(f"Total rows: {len(output_df)}")
print(f"Rows with evidence (sem_top_triple found): {output_df['sem_top_triple'].notna().sum()}")
print(f"Rows without evidence (no KG match):       {output_df['sem_top_triple'].isna().sum()}")

print(f"\nSample results:")
sample = output_df[output_df["sem_top_triple"].notna()][["claim","sem_top_triple","sem_top_score"]].head(5)
print(sample.to_string(index=False))

driver.close()