import pandas as pd

original = pd.read_csv("extracted_claims_v4_new.csv")
sem      = pd.read_csv("sem_fallback_results3.csv")

# ── fix 1: deduplicate sem on claim ───────────────────────────
print(f"Sem before dedup: {len(sem)}")
sem = sem.drop_duplicates(subset="claim", keep="first")
print(f"Sem after dedup:  {len(sem)}")

# ── property ID → human readable label ────────────────────────
prop_map = {
    "P6":   "head of government",
    "P17":  "country",
    "P18":  "image",
    "P19":  "place of birth",
    "P20":  "place of death",
    "P21":  "gender",
    "P22":  "father",
    "P25":  "mother",
    "P26":  "spouse",
    "P27":  "country of citizenship",
    "P31":  "instance of",
    "P36":  "capital",
    "P37":  "official language",
    "P39":  "position held",
    "P40":  "child",
    "P50":  "author",
    "P57":  "director",
    "P58":  "screenwriter",
    "P84":  "architect",
    "P86":  "composer",
    "P102": "member of political party",
    "P106": "occupation",
    "P108": "employer",
    "P112": "founded by",
    "P123": "publisher",
    "P127": "owned by",
    "P131": "located in",
    "P136": "genre",
    "P137": "operator",
    "P140": "religion",
    "P150": "contains administrative division",
    "P155": "follows",
    "P156": "followed by",
    "P161": "cast member",
    "P162": "producer",
    "P175": "performer",
    "P176": "manufacturer",
    "P179": "part of series",
    "P180": "depicts",
    "P190": "twinned with",
    "P195": "collection",
    "P199": "business division",
    "P206": "located next to body of water",
    "P210": "party chief representative",
    "P217": "inventory number",
    "P218": "ISO 639-1 code",
    "P219": "ISO 639-2 code",
    "P220": "ISO 639-3 code",
    "P229": "IATA airline designator",
    "P238": "IATA airport code",
    "P239": "ICAO airport code",
    "P240": "FAA airport identifier",
    "P246": "element symbol",
    "P263": "official residence",
    "P264": "record label",
    "P272": "production company",
    "P276": "location",
    "P279": "subclass of",
    "P306": "operating system",
    "P344": "director of photography",
    "P349": "NDL identifier",
    "P355": "subsidiary",
    "P360": "is a list of",
    "P361": "part of",
    "P364": "original language",
    "P400": "platform",
    "P403": "mouth of waterway",
    "P407": "language of work",
    "P412": "voice type",
    "P413": "position played",
    "P417": "patron saint",
    "P421": "time zone",
    "P422": "taxonomic type",
    "P423": "shooting handedness",
    "P425": "field of this occupation",
    "P427": "taxonomic type",
    "P428": "botanist author abbreviation",
    "P429": "field goal percentage",
    "P437": "distribution format",
    "P449": "original network",
    "P450": "astronaut mission",
    "P451": "unmarried partner",
    "P452": "industry",
    "P453": "character role",
    "P457": "foundational text",
    "P460": "said to be the same as",
    "P461": "opposite of",
    "P462": "color",
    "P463": "member of",
    "P466": "occupant",
    "P467": "legislated by",
    "P469": "lakes on river",
    "P470": "electorate",
    "P474": "country calling code",
    "P476": "CELEX number",
    "P477": "Canadian heritage river number",
    "P478": "volume",
    "P479": "input device",
    "P480": "FilmAffinity ID",
    "P481": "Pimberly ID",
    "P483": "recorded at",
    "P485": "archives at",
    "P486": "MeSH ID",
    "P488": "chairperson",
    "P489": "currency symbol",
    "P490": "provisional designation",
    "P495": "country of origin",
    "P500": "exception to",
    "P501": "guarantee",
    "P502": "HURDAT identifier",
    "P503": "IHO hydrographic region",
    "P505": "general manager",
    "P506": "ISO 15924 alpha-4 code",
    "P509": "cause of death",
    "P511": "honorific prefix",
    "P512": "academic degree",
    "P515": "phase of matter",
    "P516": "powered by",
    "P517": "interaction",
    "P518": "applies to part",
    "P520": "armament",
    "P521": "scheduled service destination",
    "P522": "type of orbit",
    "P523": "temporal range start",
    "P524": "temporal range end",
    "P525": "Redistricting Commission",
    "P527": "has part",
    "P528": "catalog code",
    "P529": "runway",
    "P530": "diplomatic relation",
    "P531": "diplomatic mission sent",
    "P533": "diplomatic mission received",
    "P535": "Find A Grave memorial ID",
    "P536": "ATP tournament ID",
    "P537": "ITFT tournament ID",
    "P539": "Museofile",
    "P541": "office contested",
    "P542": "officially opened by",
    "P543": "officially closed by",
    "P545": "territory overlaps",
    "P546": "docking port",
    "P547": "commemorates",
    "P548": "version type",
    "P549": "Mathematics Genealogy Project ID",
    "P551": "residence",
    "P552": "handedness",
    "P553": "website account on",
    "P554": "website username",
    "P555": "doubles record",
    "P556": "crystal system",
    "P557": "DiseasesDB ID",
    "P558": "singles record",
    "P559": "terminus",
    "P560": "direction",
    "P561": "no value rank",
    "P562": "central bank",
    "P563": "ICD-O",
    "P564": "singles record",
    "P565": "crystal habit",
    "P566": "basionym",
    "P567": "underlies",
    "P568": "overlies",
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception date",
    "P576": "dissolved date",
    "P577": "publication date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
    "P586": "IPNI author ID",
    "P587": "MMSI",
    "P588": "codon",
    "P589": "point group",
    "P590": "GNIS ID",
    "P591": "EC enzyme number",
    "P592": "ChEMBL ID",
    "P593": "HomoloGene ID",
    "P594": "Ensembl gene ID",
    "P595": "Guide to Pharmacology ID",
    "P597": "ITF player ID",
    "P598": "commander",
    "P599": "ITF tournament ID",
    "P600": "Rome statute article",
    "Precip": "average precipitation",
}

def pid_to_label(pid):
    if pd.isna(pid) or str(pid).strip() == "":
        return ""
    return prop_map.get(str(pid).strip(), str(pid).strip())

# ── split original ─────────────────────────────────────────────
exact_df    = original[original["route"] == "kg_exact_match"].copy()
retrieve_df = original[original["route"] == "kg_retrieve"].copy()
semantic_df = original[original["route"] == "semantic_search"].copy()

# ── exact match: build natural language premise ────────────────
def build_exact_premise(row):
    subj = str(row["subject"]).strip()
    prop = pid_to_label(row["property_id"])
    val  = str(row["object_text"]).strip() if pd.notna(row["object_text"]) else ""
    if prop and val:
        return f"{subj} {prop} {val}"
    elif val:
        return f"{subj} {val}"
    return ""

exact_df["nli_premise"]      = exact_df.apply(build_exact_premise, axis=1)
exact_df["evidence_source"]  = "exact_match"
exact_df["sem_top_triple"]   = None
exact_df["sem_top_score"]    = None
exact_df["sem_evidence"]     = None
# exact_df["sem_verdict"]      = None

# ── kg_retrieve: same as exact match ──────────────────────────
retrieve_df["nli_premise"]     = retrieve_df.apply(build_exact_premise, axis=1)
retrieve_df["evidence_source"] = "kg_retrieve"
retrieve_df["sem_top_triple"]  = None
retrieve_df["sem_top_score"]   = None
retrieve_df["sem_evidence"]    = None
# retrieve_df["sem_verdict"]     = None

# ── semantic fallback: merge + use sem_top_triple as premise ───
sem_cols = ["claim", "sem_top_triple", "sem_top_score", "sem_evidence"]
semantic_merged = semantic_df.merge(sem[sem_cols], on="claim", how="left")

def triple_to_premise(triple_str):
    if pd.isna(triple_str) or str(triple_str).strip() in ("", "None"):
        return ""
    # "colin kaepernick --[OCCUPATION]--> american football player"
    # → "colin kaepernick occupation american football player"
    import re
    match = re.match(r"(.+?)\s*--\[(.+?)\]-->\s*(.+)", str(triple_str))
    if match:
        subj = match.group(1).strip()
        rel  = match.group(2).strip().replace("_", " ").lower()
        val  = match.group(3).strip()
        return f"{subj} {rel} {val}"
    return str(triple_str).strip()

semantic_merged["nli_premise"] = semantic_merged["sem_top_triple"].apply(triple_to_premise)

semantic_merged["evidence_source"] = "semantic_fallback"

# ── stack all together ─────────────────────────────────────────
unified = pd.concat([exact_df, retrieve_df, semantic_merged], ignore_index=True)
unified["has_evidence"] = unified["nli_premise"].apply(
    lambda x: str(x).strip() not in ("", "None", "nan")
)

# ── report ─────────────────────────────────────────────────────
print(f"Total rows:               {len(unified)}")
print(f"NLI-ready (has evidence): {unified['has_evidence'].sum()}")
print(f"No evidence:              {(~unified['has_evidence']).sum()}")
print(f"\nEvidence source distribution:")
print(unified["evidence_source"].value_counts())

print(f"\nSample NLI premises (exact match):")
print(unified[unified["evidence_source"]=="exact_match"][["claim","nli_premise"]].head(5).to_string(index=False))

print(f"\nSample NLI premises (semantic fallback):")
print(unified[unified["evidence_source"]=="semantic_fallback"][["claim","nli_premise"]].head(5).to_string(index=False))

unified.to_csv("unified_claims_for_nli.csv", index=False)
print(f"\nSaved → unified_claims_for_nli.csv")