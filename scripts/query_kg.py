import re
import calendar
import pandas as pd
from neo4j import GraphDatabase
from db_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ── load extracted triples ────────────────────────────────────
df = pd.read_csv("D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/factgraph/extracted_claims_v4_new.csv")
print(f"Total rows: {len(df)}")
print(f"Route distribution:\n{df['route'].value_counts()}\n")

# ── property mapping ──────────────────────────────────────────
PROPERTY_ID_TO_REL = {
    "P569": "DATE_OF_BIRTH",
    "P570": "DATE_OF_DEATH",
    "P19":  "PLACE_OF_BIRTH",
    "P27":  "COUNTRY_OF_CITIZENSHIP",
    "P106": "OCCUPATION",
    "P112": "FOUNDED_BY",
    "P17":  "COUNTRY",
    "P159": "HEADQUARTERS",
    "P178": "DEVELOPER",
    "P495": "COUNTRY_OF_ORIGIN",
    "P39":  "POSITION_HELD",
    "P264": "RECORD_LABEL",
    "P136": "GENRE",
    "P161": "CAST_MEMBER",
    "P57":  "DIRECTOR",
    "P58":  "SCREENWRITER",
    "P452": "INDUSTRY",
    # not in KG - will be marked property_not_in_kg:
    # P20, P26, P127, P131, P361, P463, P166
}

# ── vocabulary for P106 fix ───────────────────────────────────
OCCUPATIONS = {
    "actor", "actress", "voice actor", "comedian", "presenter", "host",
    "broadcaster", "journalist", "reporter", "anchor", "correspondent",
    "columnist", "editor", "publisher", "singer", "musician", "composer",
    "rapper", "songwriter", "drummer", "guitarist", "bassist", "pianist",
    "conductor", "dj", "director", "film director", "producer", "screenwriter",
    "playwright", "animator", "stuntman", "stuntwoman", "writer", "author",
    "novelist", "poet", "painter", "sculptor", "photographer", "illustrator",
    "cartoonist", "model", "dancer", "choreographer", "athlete", "footballer",
    "basketball player", "baseball player", "golfer", "boxer", "wrestler",
    "swimmer", "runner", "coach", "manager", "referee", "trainer",
    "entrepreneur", "businessman", "businesswoman", "executive", "ceo",
    "chairman", "investor", "banker", "accountant", "programmer", "engineer",
    "developer", "inventor", "architect", "scientist", "professor", "teacher",
    "lecturer", "historian", "economist", "philosopher", "theologian",
    "politician", "lawyer", "attorney", "judge", "senator", "congressman",
    "congresswoman", "representative", "ambassador", "diplomat", "secretary",
    "governor", "chancellor", "president", "soldier", "general", "admiral",
    "officer", "pilot", "astronaut", "priest", "minister", "bishop", "pastor",
    "rabbi", "imam", "king", "queen", "prince", "princess", "emperor",
    "empress", "doctor", "surgeon", "physician", "nurse", "chef", "cook",
    "restaurateur", "realtor", "activist", "philanthropist", "reformer",
    "television presenter", "professional footballer", "singer-songwriter",
    "record producer", "band", "organization", "company",
    "prime minister", "film actress", "stage actor", "fashion model",
    "film writer", "film composer", "film screenwriter", "television director",
    "opera singer", "hip hop record producer",
}

# ── FIX 1: date normalization ─────────────────────────────────
MONTH_MAP = {m.lower(): str(i).zfill(2)
             for i, m in enumerate(calendar.month_name) if m}

def normalize_date(text):
    """
    Convert extracted date strings to YYYY-MM-DD format to match KG.
    KG stores dates as '1987-11-03' (Wikidata ISO with time stripped).
    """
    if not isinstance(text, str):
        return text
    text = text.strip()
    text_lower = text.lower()

    # already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', text):
        return text
    # already YYYY
    if re.match(r'^\d{4}$', text):
        return text

    # extract year (1000-2029)
    year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text_lower)
    year = year_match.group(1) if year_match else None

    # extract month name
    month = None
    for m_name, m_num in MONTH_MAP.items():
        if m_name in text_lower:
            month = m_num
            break

    # extract day (strip ordinal suffixes st/nd/rd/th), exclude years
    day = None
    day_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', text_lower)
    if day_match:
        d = int(day_match.group(1))
        if 1 <= d <= 31:
            day = str(d).zfill(2)

    if year and month and day:
        return f"{year}-{month}-{day}"
    if year and month:
        return f"{year}-{month}"
    if year:
        return year

    # unparseable (e.g. "nineties", "20th century", "April") - return as-is
    return text


# ── FIX 2: P106 occupation cleaning ──────────────────────────
def clean_occupation(text):
    """
    Strip leading modifiers and reject long spans that don't match
    known occupations. Returns cleaned occupation or None if invalid.
    """
    if not isinstance(text, str):
        return None
    text = text.lower().strip()

    # strip leading adverbs/negations
    text = re.sub(
        r"^\b(only|solely|not|just|also|even|still|merely|never)\b\s*",
        "", text
    )

    # strip nationality/modifier adjectives
    MODIFIERS = {
        "former", "professional", "retired", "american", "british", "indian",
        "canadian", "australian", "french", "german", "italian", "irish",
        "scottish", "welsh", "english", "japanese", "chinese", "russian",
        "spanish", "mexican", "korean", "swedish", "norwegian", "danish",
        "finnish", "polish", "dutch", "portuguese", "greek", "turkish",
        "swiss", "austrian", "belgian", "czech", "hungarian", "romanian",
        "bulgarian", "ukrainian", "serbian", "croatian", "puerto rican",
        "south african", "nigerian",
    }
    adj_pattern = r"\b(" + "|".join(re.escape(a) for a in MODIFIERS) + r")\b"
    text = re.sub(adj_pattern, "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")

    if not text:
        return None

    # exact match in known occupations -> accept as-is
    if text in OCCUPATIONS:
        return text

    words = text.split()

    # 1-2 words: accept if first word is a known occupation
    if len(words) <= 2:
        if text in OCCUPATIONS or words[0] in OCCUPATIONS:
            return text

    # 3+ words: only accept exact match, else try first word
    if len(words) >= 3:
        if text in OCCUPATIONS:
            return text
        if words[0] in OCCUPATIONS:
            return words[0]
        return None  # too long and unknown -> reject

    return text


# ── FIX 3: normalize object based on property type ───────────
def normalize_object(obj_text, property_id):
    """
    Apply property-specific normalization before querying.
    Returns None if object is too dirty to query with.
    """
    if not isinstance(obj_text, str):
        return None

    # date properties: normalize to YYYY-MM-DD
    if property_id in {"P569", "P570"}:
        normalized = normalize_date(obj_text)
        return normalized.lower().strip() if normalized else None

    # occupation: clean and validate
    if property_id == "P106":
        cleaned = clean_occupation(obj_text)
        return cleaned.lower().strip() if cleaned else None

    # P178 (developer/producer): reject dirty NPs, only query clean titles
    if property_id == "P178":
        if not is_clean_title(obj_text):
            return None
        return obj_text.lower().strip()

    # P58 (screenwriter): KG stores (film)->SCREENWRITER->(person)
    # extractor puts person as subject, film/phrase as object
    # object cleaning: extract film title if it's in "screenplay for X" pattern
    if property_id == "P58":
        match = re.search(r'screenplay for (.+)', obj_text, re.I)
        if match:
            title = match.group(1).strip()
            return title.lower().strip() if is_clean_title(title) else None
        if not is_clean_title(obj_text):
            return None
        return obj_text.lower().strip()

    # default: just lowercase + strip
    return obj_text.lower().strip()


# ── normalize helpers ─────────────────────────────────────────
def normalize_name(text):
    if not isinstance(text, str):
        return None
    return text.lower().strip()

def normalize_property(prop):
    if not isinstance(prop, str):
        return None
    return prop.strip().upper()


# ── P178 object cleaner ───────────────────────────────────────
GENERIC_OBJ_WORDS = {
    'album', 'record', 'song', 'film', 'movie', 'book', 'show', 'series',
    'ep', 'single', 'track', 'work', 'media', 'content', 'stuff', 'something',
    'debut', 'tv', 'web', 'studio', 'reality', 'drama', 'doves', 'leeches',
    'prisoners', 'seals', 'cats', 'material', 'stuff', 'things',
}

JUNK_START_WORDS = {
    'at', 'his', 'her', 'their', 'my', 'an', 'a', 'the', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
    'twelve', 'seventeen', 'multiple', 'least', 'only', 'some', 'live',
    'animated', 'online', 'comedy', 'fantasy', 'horror', 'action', 'arabic',
    'prussian', 'indian', 'collection', 'extended', 'song', 'film', 'less',
    'over', 'about', 'various', 'zero', 'no',
}

def is_clean_title(text):
    """
    Returns True if object looks like a real title/name worth querying.
    Rejects vague NPs, quantified phrases, and generic nouns.
    """
    if not text:
        return False
    text_lower = text.lower().strip()
    words = text_lower.split()
    if len(words) > 4:
        return False
    if words[0] in JUNK_START_WORDS:
        return False
    if len(words) == 1 and text_lower in GENERIC_OBJ_WORDS:
        return False
    return True


# ── query functions ───────────────────────────────────────────
def check_triple_exists(tx, subject, rel_type, obj):
    query = f"""
        MATCH (e:Entity {{name: $subject}})-[:{rel_type}]->(v:Value {{name: $obj}})
        RETURN count(*) > 0 AS found
    """
    result = tx.run(query, subject=subject, obj=obj)
    record = result.single()
    return record["found"] if record else False


def check_triple_exists_both_directions(tx, subject, rel_type, obj):
    """
    Try both (subject)-[r]->(obj) and (obj)-[r]->(subject).
    Used for P161 where extractor sometimes swaps actor/film.
    Returns (found, direction) where direction is 'forward', 'swapped', or None.
    """
    # forward
    query = f"""
        MATCH (e:Entity {{name: $subject}})-[:{rel_type}]->(v:Value {{name: $obj}})
        RETURN count(*) > 0 AS found
    """
    result = tx.run(query, subject=subject, obj=obj)
    record = result.single()
    if record and record["found"]:
        return True, "forward"

    # swapped
    query = f"""
        MATCH (e:Entity {{name: $obj}})-[:{rel_type}]->(v:Value {{name: $subject}})
        RETURN count(*) > 0 AS found
    """
    result = tx.run(query, subject=subject, obj=obj)
    record = result.single()
    if record and record["found"]:
        return True, "swapped"

    return False, None


def get_entity_exists(tx, subject):
    result = tx.run(
        "MATCH (e:Entity {name: $subject}) RETURN count(*) > 0 AS found",
        subject=subject
    )
    record = result.single()
    return record["found"] if record else False


def get_property_values(tx, subject, rel_type):
    query = f"""
        MATCH (e:Entity {{name: $subject}})-[:{rel_type}]->(v:Value)
        RETURN v.name AS value
    """
    result = tx.run(query, subject=subject)
    return [r["value"] for r in result]


# ── main querying loop ────────────────────────────────────────
results = []

with driver.session() as session:

    for idx, row in df.iterrows():
        route        = row["route"]
        claim        = row["claim"]
        subject_raw  = row["subject"]
        property_raw = row["property_id"]
        object_raw   = row["object_text"] 

        subject  = normalize_name(subject_raw)
        prop     = normalize_property(property_raw)

        # FIX 3: use property-aware object normalization
        obj = normalize_object(object_raw, prop) if prop else normalize_name(object_raw)

        result_row = {
            "claim":            claim,
            "subject":          subject_raw,
            "predicate":        row["predicate"],
            "object":           object_raw,
            "object_queried":   obj,        # what we actually queried with
            "property_id":      property_raw,
            "confidence":       row["confidence"],
            "route":            route,
            "mapping_source":   row["mapping_source"],
            "kg_result":        None,
            "kg_retrieved":     None,
            "entity_exists":    None,
            "match_direction":  None,       # for P161 bidirectional
        }

        # ── kg_exact_match ────────────────────────────────────
        if route == "kg_exact_match":
            if not subject or not prop or not obj:
                result_row["kg_result"] = "skip_missing_fields"
            else:
                rel_type = PROPERTY_ID_TO_REL.get(prop)
                if rel_type is None:
                    result_row["kg_result"] = "property_not_in_kg"
                else:
                    try:
                        # P161: try both directions (actor/film sometimes swapped)
                        if prop == "P161":
                            found, direction = session.execute_read(
                                check_triple_exists_both_directions,
                                subject, rel_type, obj
                            )
                            result_row["kg_result"] = "match" if found else "no_match"
                            if found:
                                result_row["match_direction"] = direction

                        # P58: KG stores (film)->SCREENWRITER->(person)
                        # swap subject/object before querying
                        elif prop == "P58":
                            found = session.execute_read(
                                check_triple_exists, obj, rel_type, subject
                            )
                            result_row["kg_result"] = "match" if found else "no_match"

                        # all other properties: standard forward query
                        else:
                            found = session.execute_read(
                                check_triple_exists, subject, rel_type, obj
                            )
                            result_row["kg_result"] = "match" if found else "no_match"

                        if result_row["kg_result"] == "no_match":
                            result_row["entity_exists"] = session.execute_read(
                                get_entity_exists, subject
                            )
                    except Exception as ex:
                        result_row["kg_result"] = f"error: {ex}"

        # ── kg_retrieve ───────────────────────────────────────
        elif route == "kg_retrieve":
            if not subject or not prop:
                result_row["kg_result"] = "skip_missing_fields"
            else:
                rel_type = PROPERTY_ID_TO_REL.get(prop)
                if rel_type is None:
                    result_row["kg_result"] = "property_not_in_kg"
                else:
                    try:
                        values = session.execute_read(
                            get_property_values, subject, rel_type
                        )
                        if values:
                            result_row["kg_result"]    = "retrieved"
                            result_row["kg_retrieved"] = " | ".join(values)
                        else:
                            result_row["kg_result"] = "not_found"
                            result_row["entity_exists"] = session.execute_read(
                                get_entity_exists, subject
                            )
                    except Exception as ex:
                        result_row["kg_result"] = f"error: {ex}"

        # ── semantic_search: skip for now ─────────────────────
        elif route == "semantic_search":
            result_row["kg_result"] = "pending_semantic"

        results.append(result_row)

        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1} / {len(df)} rows...")

driver.close()

# ── save ──────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv("kg_query_results_v3.csv", index=False)

# ── summary ───────────────────────────────────────────────────
print("\n=== KG QUERY RESULTS V3 ===")
print(f"Total rows: {len(out_df)}")
print(f"\nkg_result distribution:\n{out_df['kg_result'].value_counts()}")

exact = out_df[out_df["route"] == "kg_exact_match"]
print(f"\nkg_exact_match rows: {len(exact)}")
if len(exact) > 0:
    match_rate = (exact["kg_result"] == "match").sum() / len(exact)
    print(f"  match:              {(exact['kg_result'] == 'match').sum()}  ({match_rate:.1%})")
    print(f"  no_match:           {(exact['kg_result'] == 'no_match').sum()}")
    print(f"  property_not_in_kg: {(exact['kg_result'] == 'property_not_in_kg').sum()}")
    print(f"  skip_missing:       {(exact['kg_result'] == 'skip_missing_fields').sum()}")

    no_match = exact[exact["kg_result"] == "no_match"]
    if len(no_match) > 0:
        entity_exists = no_match["entity_exists"].sum()
        print(f"\n  Of {len(no_match)} no_match rows:")
        print(f"    entity exists in KG: {int(entity_exists)}")
        print(f"    entity not in KG:    {len(no_match) - int(entity_exists)}")

    # per-property match breakdown
    print("\n  Match rate by property:")
    for pid, rel in sorted(PROPERTY_ID_TO_REL.items()):
        prop_rows = exact[exact["property_id"] == pid]
        if len(prop_rows) == 0:
            continue
        matched = (prop_rows["kg_result"] == "match").sum()
        print(f"    {pid} ({rel:<25}) {matched:>4} / {len(prop_rows):>4}  ({matched/len(prop_rows):.0%})")

retrieve = out_df[out_df["route"] == "kg_retrieve"]
print(f"\nkg_retrieve rows: {len(retrieve)}")
if len(retrieve) > 0:
    print(f"  retrieved:  {(retrieve['kg_result'] == 'retrieved').sum()}")
    print(f"  not_found:  {(retrieve['kg_result'] == 'not_found').sum()}")

print(f"\nSaved to: bleh.csv")
