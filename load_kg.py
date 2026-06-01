import pandas as pd
from neo4j import GraphDatabase
from db_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/kg_facts_new.csv")
print(len(df))

def clean_df(val):
    #check for datatype, convert evrything to str
    if not isinstance(val,str):
        return str(val)
    if "T00:00:00Z" in val:
        return val.split('T')[0]
    return val.strip()

def normalize_prop(prop):
    return prop.strip().upper().replace(" ","_")

df["value"]=df["value"].apply(clean_df)
df["property"]=df["property"].apply(normalize_prop)
df["entity"]=df["entity"].str.lower().str.strip()
df["value"]=df["value"].str.lower().str.strip()

#print(df.head())

# ── batch loader ──────────────────────────────────────────────
def load_batch(tx, batch, prop_name):
    tx.run(f"""
        UNWIND $batch AS row
        MERGE (e:Entity {{name: row.e}})
        MERGE (v:Value  {{name: row.v}})
        MERGE (e)-[:{prop_name}]->(v)
    """, batch=batch)

# ── load all facts ────────────────────────────────────────────
print("\nLoading into Neo4j...")
BATCH_SIZE = 200
total = 0

# group by property so each batch has one relationship type
with driver.session() as session:
    for prop_name, group in df.groupby("property"):
        batch = []
        for _, row in group.iterrows():
            batch.append({
                "e": row["entity"],
                "v": row["value"]
            })
            if len(batch) >= BATCH_SIZE:
                session.execute_write(load_batch, batch, prop_name)
                total += len(batch)
                batch = []
        # load remaining rows for this property
        if batch:
            session.execute_write(load_batch, batch, prop_name)
            total += len(batch)

        print(f"  loaded {prop_name} — {len(group)} facts")

print(f"\nDone. Total facts loaded: {total}")
driver.close()


