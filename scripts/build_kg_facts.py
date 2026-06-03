import requests
import pandas as pd
import json
import re
import time
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# ── STEP 1: load fever ────────────────────────────────────────
print("Loading FEVER claims...")
fever_claims = []
with open("C:/Users/vdsha/Downloads/shared_task_dev.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        fever_claims.append({
            "claim": item["claim"],
            "label": item["label"]
        })
print(f"Loaded {len(fever_claims)} claims")

# ── STEP 2: extract entities from fever claims ────────────────
print("\nExtracting entities from FEVER claims...")
entity_counts = Counter()

for i, item in enumerate(fever_claims):
    doc = nlp(item["claim"])
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            entity_counts[ent.text.lower().strip()] += 1
    if (i + 1) % 2000 == 0:
        print(f"  processed {i+1} claims...")

# keep only entities mentioned 5+ times
frequent_entities = {e: count for e, count in entity_counts.items()
                     if count >= 5}
print(f"Entities mentioned 5+ times: {len(frequent_entities)}")

# ── STEP 3: look up each entity on wikidata ───────────────────
print("\nLooking up entities on Wikidata...")

def search_wikidata(entity_name):
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action":   "wbsearchentities",
                "search":   entity_name,
                "language": "en",
                "format":   "json",
                "limit":    1
            },
            headers={"User-Agent": "FeverExplorer/1.0"},
            timeout=10
        )
        if r.status_code != 200 or not r.text.strip():
            return None
        results = r.json().get("search", [])
        if results:
            return {
                "id":          results[0]["id"],
                "label":       results[0].get("label", ""),
                "description": results[0].get("description", "")
            }
    except Exception as e:
        pass
    return None

wikidata_matches = {}
entity_list = list(frequent_entities.keys())

for i, entity in enumerate(entity_list):
    result = search_wikidata(entity)
    if result:
        wikidata_matches[entity] = result
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(entity_list)} looked up, "
              f"{len(wikidata_matches)} matched so far...")
    time.sleep(1)

print(f"\nEntities found on Wikidata: {len(wikidata_matches)}")

# ── STEP 4: fetch facts using simple property queries ─────────
print("\nFetching facts for matched entities...")

# skip obviously irrelevant descriptions
SKIP_KEYWORDS = [
    "fictional", "tv series", "film", "album", "song",
    "video game", "episode", "character"
]

def should_skip(description):
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in SKIP_KEYWORDS)

# use simple direct property lookups instead of open-ended query
PROPERTIES = {
    "P569":  "date of birth",
    "P570":  "date of death",
    "P19":   "place of birth",
    "P27":   "country of citizenship",
    "P106":  "occupation",
    "P571":  "inception date",
    "P112":  "founded by",
    "P17":   "country",
    "P159":  "headquarters",
    "P452":  "industry",
    "P178":  "developer",
    "P495":  "country of origin",
    "P166":  "award received",
    "P39":   "position held",
    "P264":  "record label",
    "P136":  "genre",
    "P161":  "cast member",      # covers film/TV appearance claims
    "P57":   "director",         # covers director claims
    "P58":   "screenwriter",     # covers writer claims
}

def fetch_facts_simple(wikidata_id, entity_name):
    facts = []
    for prop_id, prop_name in PROPERTIES.items():
        query = f"""
        SELECT ?valueLabel WHERE {{
          wd:{wikidata_id} wdt:{prop_id} ?value.
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
          }}
        }}
        LIMIT 5
        """
        try:
            r = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": query, "format": "json"},
                headers={"User-Agent": "FeverExplorer/1.0"},
                timeout=10
            )
            if r.status_code != 200 or not r.text.strip():
                continue
            data = r.json()
            for item in data["results"]["bindings"]:
                value = item.get("valueLabel", {}).get("value", "")
                if value and not value.startswith("Q"):
                    facts.append({
                        "entity":   entity_name,
                        "property": prop_name,
                        "value":    value
                    })
        except Exception:
            pass
        time.sleep(0.2)
    return facts

all_facts = []
match_list = list(wikidata_matches.items())

for i, (entity, info) in enumerate(match_list):
    # skip fictional characters, films, albums etc
    if should_skip(info.get("description", "")):
        continue

    facts = fetch_facts_simple(info["id"], entity)
    if facts:
        all_facts.extend(facts)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(match_list)} entities processed, "
              f"{len(all_facts)} facts so far...")
    time.sleep(0.5)

print(f"\nTotal facts fetched: {len(all_facts)}")

if not all_facts:
    print("ERROR: no facts fetched — check your internet connection")
else:
    df_facts = pd.DataFrame(all_facts)
    print(f"\nSample facts:")
    print(df_facts.head(30).to_string())

    # ── STEP 5: check coverage ────────────────────────────────
    print("\nChecking FEVER coverage...")
    covered_entities = set(df_facts["entity"].unique())

    def mentions_covered_entity(claim_text, entities):
        claim_lower = claim_text.lower()
        matched = []
        for e in entities:
            pattern = r'\b' + re.escape(e) + r'\b'
            if re.search(pattern, claim_lower):
                matched.append(e)
        return matched

    matched_claims = []
    for item in fever_claims:
        matches = mentions_covered_entity(item["claim"], covered_entities)
        if matches:
            matched_claims.append({
                "claim":   item["claim"],
                "label":   item["label"],
                "matched": matches
            })

    df_matched = pd.DataFrame(matched_claims)
    print(f"\nFEVER claims your KG can verify: {len(df_matched)}")
    print(f"That's {len(df_matched)/len(fever_claims)*100:.1f}% of FEVER dev set")
    print(f"\nLabel distribution:")
    print(df_matched["label"].value_counts())
    print(f"\nSample covered claims:")
    print(df_matched[["claim","label","matched"]].head(20).to_string())

    # ── STEP 6: save ──────────────────────────────────────────
    df_facts.to_csv("C:/Users/vdsha/Downloads/kg_facts_new.csv", index=False)
    df_matched.to_csv("C:/Users/vdsha/Downloads/covered_claims_new.csv", index=False)
    print("\nSaved kg_facts_new.csv and covered_claims_new.csv to Downloads")
