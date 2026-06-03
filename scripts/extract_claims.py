import re
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ============================================================
#  SETUP
# ============================================================

nlp = spacy.load("en_core_web_sm")
if "merge_entities" not in nlp.pipe_names:
    nlp.add_pipe("merge_entities")

SEM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Raised a bit to reduce over-eager semantic assignments
SEMANTIC_THRESHOLD = 0.64

# ============================================================
#  PROPERTY LABELS / CONFIDENCE
# ============================================================

PROPERTY_LABELS = {
    "P569": "date of birth",
    "P570": "date of death",
    "P19":  "place of birth",
    "P20":  "place of death",
    "P26":  "spouse",
    "P27":  "country of citizenship",
    "P106": "occupation",
    "P112": "founded by",
    "P127": "owned by",
    "P17":  "country",
    "P131": "located in administrative entity",
    "P159": "headquarters location",
    "P178": "developer / producer",
    "P361": "part of",
    "P452": "industry",
    "P463": "member of",
    "P495": "country of origin",
    "P166": "award received",
    "P39":  "position held",
    "P264": "record label",
    "P136": "genre",
    "P161": "cast member / narrator / host",
    "P57":  "director",
    "P58":  "screenwriter",
}

HIGH_CONF_PROPS = {
    "P569", "P570", "P19", "P20", "P26", "P27", "P106",
    "P166", "P264", "P136", "P452", "P495", "P127",
    "P159", "P361", "P463"
}

MEDIUM_CONF_PROPS = {
    "P57", "P58", "P161", "P112", "P178", "P131", "P39"
}

# Only these are safe enough for exact string KG lookup.
# Everything else with a property should usually go to kg_retrieve.
EXACT_MATCH_READY_PROPS = {
    "P569", "P570", "P19", "P20", "P26", "P27", "P106",
    "P127", "P131", "P159", "P361", "P463", "P452",
    "P495", "P166", "P136", "P264"
}

# ============================================================
#  VOCAB / NORMALIZATION
# ============================================================

OCCUPATIONS = {
    "actor", "actress", "voice actor", "comedian", "presenter", "host",
    "broadcaster", "journalist", "reporter", "anchor", "correspondent",
    "columnist", "editor", "publisher", "singer", "musician", "composer",
    "rapper", "songwriter", "drummer", "guitarist", "bassist", "pianist",
    "conductor", "dj", "director", "film director", "producer", "screenwriter",
    "playwright", "animator", "writer", "author", "novelist", "poet",
    "photographer", "illustrator", "model", "dancer", "choreographer",
    "athlete", "footballer", "basketball player", "baseball player",
    "golfer", "boxer", "wrestler", "swimmer", "runner", "coach",
    "manager", "entrepreneur", "executive", "ceo", "chairman", "investor",
    "banker", "accountant", "programmer", "engineer", "developer",
    "inventor", "architect", "scientist", "professor", "teacher",
    "lecturer", "historian", "economist", "philosopher", "politician",
    "lawyer", "attorney", "judge", "senator", "representative",
    "ambassador", "diplomat", "governor", "president", "soldier",
    "general", "pilot", "astronaut", "priest", "minister", "bishop",
    "doctor", "surgeon", "physician", "nurse", "chef", "activist",
    "philanthropist", "television presenter", "professional footballer",
    "singer-songwriter", "record producer", "band", "organization", "company"
}

NATIONALITY_WORDS = {
    "american", "british", "indian", "chinese", "canadian", "french", "german",
    "italian", "japanese", "russian", "australian", "irish", "mexican",
    "spanish", "armenian", "swedish", "norwegian", "danish", "finnish",
    "polish", "dutch", "portuguese", "greek", "turkish", "swiss", "austrian",
    "belgian", "czech", "hungarian", "romanian", "bulgarian", "ukrainian",
    "serbian", "croatian", "slovenian", "slovak", "latvian", "lithuanian",
    "estonian", "scottish", "welsh", "english", "icelandic", "brazilian",
    "argentinian", "colombian", "chilean", "peruvian", "venezuelan",
    "ecuadorian", "bolivian", "uruguayan", "paraguayan", "cuban", "jamaican",
    "haitian", "korean", "taiwanese", "vietnamese", "thai", "indonesian",
    "malaysian", "filipino", "singaporean", "bangladeshi", "pakistani",
    "nepali", "sri lankan", "burmese", "cambodian", "mongolian", "kazakh",
    "uzbek", "afghan", "new zealander", "israeli", "palestinian", "lebanese",
    "jordanian", "syrian", "iraqi", "iranian", "persian", "saudi", "emirati",
    "kuwaiti", "qatari", "yemeni", "egyptian", "libyan", "tunisian",
    "algerian", "moroccan", "south african", "nigerian", "kenyan",
    "ethiopian", "ghanaian", "ugandan", "tanzanian", "zimbabwean",
    "senegalese", "congolese", "rwandan"
}

NATIONALITY_TO_COUNTRY = {
    "american": "United States", "british": "United Kingdom", "english": "United Kingdom",
    "scottish": "United Kingdom", "welsh": "United Kingdom", "irish": "Ireland",
    "canadian": "Canada", "australian": "Australia", "new zealander": "New Zealand",
    "french": "France", "german": "Germany", "italian": "Italy", "spanish": "Spain",
    "portuguese": "Portugal", "dutch": "Netherlands", "belgian": "Belgium",
    "swiss": "Switzerland", "austrian": "Austria", "swedish": "Sweden",
    "norwegian": "Norway", "danish": "Denmark", "finnish": "Finland",
    "icelandic": "Iceland", "greek": "Greece", "turkish": "Turkey", "polish": "Poland",
    "czech": "Czech Republic", "hungarian": "Hungary", "romanian": "Romania",
    "bulgarian": "Bulgaria", "ukrainian": "Ukraine", "russian": "Russia",
    "serbian": "Serbia", "croatian": "Croatia", "armenian": "Armenia",
    "latvian": "Latvia", "lithuanian": "Lithuania", "estonian": "Estonia",
    "indian": "India", "pakistani": "Pakistan", "bangladeshi": "Bangladesh",
    "nepali": "Nepal", "chinese": "China", "japanese": "Japan", "korean": "South Korea",
    "taiwanese": "Taiwan", "vietnamese": "Vietnam", "thai": "Thailand",
    "indonesian": "Indonesia", "malaysian": "Malaysia", "filipino": "Philippines",
    "singaporean": "Singapore", "burmese": "Myanmar", "cambodian": "Cambodia",
    "mongolian": "Mongolia", "kazakh": "Kazakhstan", "uzbek": "Uzbekistan",
    "afghan": "Afghanistan", "iranian": "Iran", "persian": "Iran", "iraqi": "Iraq",
    "syrian": "Syria", "lebanese": "Lebanon", "jordanian": "Jordan",
    "israeli": "Israel", "palestinian": "Palestine", "saudi": "Saudi Arabia",
    "emirati": "United Arab Emirates", "kuwaiti": "Kuwait", "qatari": "Qatar",
    "yemeni": "Yemen", "egyptian": "Egypt", "libyan": "Libya",
    "tunisian": "Tunisia", "algerian": "Algeria", "moroccan": "Morocco",
    "south african": "South Africa", "nigerian": "Nigeria", "kenyan": "Kenya",
    "ethiopian": "Ethiopia", "ghanaian": "Ghana", "ugandan": "Uganda",
    "tanzanian": "Tanzania", "zimbabwean": "Zimbabwe", "senegalese": "Senegal",
    "congolese": "Democratic Republic of the Congo", "rwandan": "Rwanda",
    "mexican": "Mexico", "brazilian": "Brazil", "argentinian": "Argentina",
    "colombian": "Colombia", "chilean": "Chile", "peruvian": "Peru",
    "venezuelan": "Venezuela", "cuban": "Cuba", "jamaican": "Jamaica",
}

GENERIC_MEDIA_NOUNS = {
    "film", "movie", "book", "song", "album", "series", "show",
    "record", "single", "ep", "track", "documentary", "episode",
}

GENERIC_WORK_HEADS = {
    "film", "movie", "album", "song", "book", "series", "show", "record"
}

PREFIXES = {"co", "re", "pre", "post", "non", "anti", "ex", "self", "over", "under"}

POSITION_CANONICAL = {
    "ceo": "chief executive officer",
    "chief executive": "chief executive officer",
    "chief executive officer": "chief executive officer",
    "president": "president",
    "vice president": "vice president",
    "chairman": "chairman",
    "chairperson": "chairperson",
    "founder": "founder",
    "cofounder": "co-founder",
    "co-founder": "co-founder",
    "member": "member",
    "governor": "governor",
    "senator": "senator",
    "representative": "representative",
    "prime minister": "prime minister",
    "minister": "minister",
    "mayor": "mayor",
    "captain": "captain",
    "quarterback": "quarterback",
    "starting quarterback": "quarterback",
    "record producer": "record producer",
    "executive producer": "executive producer",
    "producer": "producer",
    "director": "director",
    "screenwriter": "screenwriter",
    "host": "host",
    "presenter": "presenter",
    "judge": "judge",
    "coach": "coach",
    "manager": "manager",
}

# ============================================================
#  SEMANTIC PROPERTY PHRASES
# ============================================================

PROPERTY_TEXT = {
    "P27": ["country of citizenship", "nationality", "citizen of"],
    "P106": ["occupation", "profession", "works as", "career"],
    "P112": ["founded by", "started by", "established by"],
    "P127": ["owned by", "belongs to", "controlled by"],
    "P57": ["director", "directed by", "film director"],
    "P58": ["screenwriter", "written by", "screenplay by"],
    "P39": ["position held", "served as", "held office as"],
    "P159": ["headquarters location", "based in", "headquartered in"],
    "P131": ["located in", "administrative location"],
    "P264": ["record label", "signed with", "released on label"],
    "P136": ["genre", "music style", "type of film or music"],
    "P463": ["member of", "belongs to organization"],
    "P361": ["part of", "component of"],
    "P166": ["award received", "won award", "received award"],
    "P495": ["country of origin", "origin country"],
}

SEM_MODEL = SentenceTransformer(SEM_MODEL_NAME)
_PROPERTY_PHRASES = []
_PROPERTY_IDS = []
for _pid, _texts in PROPERTY_TEXT.items():
    for _text in _texts:
        _PROPERTY_PHRASES.append(_text)
        _PROPERTY_IDS.append(_pid)

PROPERTY_EMBS = SEM_MODEL.encode(_PROPERTY_PHRASES, convert_to_tensor=True)

# ============================================================
#  CLEANING
# ============================================================

def clean_claim(claim):
    if not isinstance(claim, str):
        return ""
    claim = claim.strip()

    def replace_hyphen(match):
        left, right = match.group(1), match.group(2)
        if left.lower() in PREFIXES:
            return left + right
        return match.group(0)

    claim = re.sub(r"(\w+)-(\w+)", replace_hyphen, claim)
    claim = re.sub(r"\s+", " ", claim)
    return claim.strip()

def normalize_entity_text(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")
    return text if text else None

def normalize_string_for_kg(text):
    text = normalize_entity_text(text)
    return text.lower().strip() if text else None

def strip_leading_modifiers(text):
    if not text:
        return None
    text = text.lower()
    text = re.sub(r"\b(only|just|also|even|still|merely|simply|solely|former|professional|retired)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    return text or None

def canonicalize_position_text(text):
    """
    Reduce role phrases like:
      'CEO of Arista Records' -> 'chief executive officer'
      'starting quarterback during ...' -> 'quarterback'
    """
    if text is None:
        return None

    text = normalize_entity_text(text)
    if not text:
        return None

    t = text.lower()

    # remove time / org / loc tail phrases
    t = re.split(r"\b(of|for|at|in|during|with|from|on)\b", t)[0].strip()
    t = strip_leading_modifiers(t)
    if not t:
        return None

    # normalize punctuation
    t = t.replace(".", "")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None

    # longest-match canonicalization
    for key in sorted(POSITION_CANONICAL, key=len, reverse=True):
        if key in t:
            return POSITION_CANONICAL[key]

    # fallback: keep short phrase if it isn't too long
    if len(t.split()) <= 3:
        return t

    # fallback: extract last noun-like word
    tokens = t.split()
    if tokens:
        return tokens[-1]

    return None

def normalize_occupation_text(text):
    if not text:
        return None
    text = normalize_entity_text(text)
    if not text:
        return None

    t = text.lower()
    # remove nationality and common modifiers
    nat_pattern = r"\b(" + "|".join(re.escape(x) for x in sorted(NATIONALITY_WORDS, key=len, reverse=True)) + r")\b"
    t = re.sub(nat_pattern, "", t)
    t = re.sub(r"\b(former|professional|retired|lead|main|only|just|also|even)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")

    # if phrase contains "and", pick known occupations from parts
    if " and " in t:
        parts = [p.strip() for p in t.split(" and ")]
        known = [p for p in parts if p in OCCUPATIONS]
        if known:
            return known[0]

    # direct match
    if t in OCCUPATIONS:
        return t

    # longest contained occupation
    for occ in sorted(OCCUPATIONS, key=len, reverse=True):
        if occ in t:
            return occ

    # allow very short fallback if noun-y and short
    if len(t.split()) <= 2:
        return t
    return None

# ============================================================
#  SPAN HELPERS
# ============================================================

def get_span_text(tok):
    if tok is None:
        return None
    span = tok.doc[tok.left_edge.i: tok.right_edge.i + 1]
    return span.text.strip()

def get_best_entity_or_span(tok):
    if tok is None:
        return None
    for ent in tok.doc.ents:
        if ent.start <= tok.i < ent.end:
            return ent.text.strip()
    return get_span_text(tok)

# ============================================================
#  DEPENDENCY HELPERS
# ============================================================

def get_root(doc):
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None

def get_subject_token(doc, root):
    subject_tok = None

    for child in root.children:
        if child.dep_ == "nsubjpass":
            return child
        if child.dep_ == "nsubj" and subject_tok is None:
            subject_tok = child

    if subject_tok is None:
        for tok in doc:
            if tok.dep_ == "nsubjpass":
                return tok
            if tok.dep_ == "nsubj" and subject_tok is None:
                subject_tok = tok

    return subject_tok

def recover_work_title_from_subject(subject_tok):
    if subject_tok is None:
        return None

    candidates = [subject_tok]
    candidates.extend(list(subject_tok.children))
    if subject_tok.head is not None:
        candidates.extend(list(subject_tok.head.children))

    for tok in candidates:
        if tok.ent_type_ in {"WORK_OF_ART", "ORG", "PRODUCT"}:
            return get_best_entity_or_span(tok)

    return None

def resolve_subject_text(subject_tok):
    if subject_tok is None:
        return None

    if subject_tok.text.lower() == "there":
        return None

    if subject_tok.lemma_.lower() in GENERIC_WORK_HEADS:
        work_title = recover_work_title_from_subject(subject_tok)
        if work_title:
            return work_title

    if subject_tok.pos_ == "NOUN":
        poss_children = [c for c in subject_tok.children if c.dep_ == "poss"]
        if poss_children:
            possible_work = recover_work_title_from_subject(subject_tok)
            if possible_work:
                return possible_work
            subject_tok = poss_children[0]

    for ent in subject_tok.doc.ents:
        if ent.start <= subject_tok.i < ent.end:
            return ent.text.replace("'s", "").strip()

    compounds = sorted([c for c in subject_tok.children if c.dep_ == "compound"], key=lambda x: x.i)
    if compounds:
        return " ".join([c.text for c in compounds] + [subject_tok.text]).replace("'s", "").strip()

    return subject_tok.text.replace("'s", "").strip()

def get_attr_token(root):
    for child in root.children:
        if child.dep_ in {"attr", "acomp", "oprd"}:
            return child
    return None

def get_direct_object_token(root):
    for child in root.children:
        if child.dep_ in {"dobj", "obj"}:
            return child
    return None

def get_root_prep_map(root):
    prep_map = {}
    for child in root.children:
        if child.dep_ == "prep":
            prep_map[child.text.lower()] = child
    return prep_map

def get_pobj_for_prep(prep_tok):
    for child in prep_tok.children:
        if child.dep_ == "pobj":
            return child
    return None

def get_xcomp_or_ccomp_child(root):
    for child in root.children:
        if child.dep_ in {"xcomp", "ccomp"}:
            return child
    return None

def has_date_entity(doc):
    return any(ent.label_ == "DATE" for ent in doc.ents)

def get_date_entity(doc):
    for ent in doc.ents:
        if ent.label_ == "DATE":
            return ent.text
    return None

def get_last_entity_of_types(doc, labels):
    vals = [ent.text for ent in doc.ents if ent.label_ in labels]
    return vals[-1] if vals else None

def improve_object_text(obj_tok):
    if obj_tok is None:
        return None

    if obj_tok.text.lower() in GENERIC_MEDIA_NOUNS:
        for child in obj_tok.children:
            if child.dep_ in {"appos", "compound"}:
                return child.text
            if child.ent_type_ in {"WORK_OF_ART", "ORG", "PRODUCT"}:
                return child.text

    for child in obj_tok.children:
        if child.dep_ == "appos":
            return child.text

    return get_best_entity_or_span(obj_tok)

def infer_subject_type(doc, subject_text):
    if not subject_text:
        return "unknown"

    st = subject_text.lower()
    for ent in doc.ents:
        if ent.text.lower() == st:
            if ent.label_ == "PERSON":
                return "person"
            if ent.label_ == "ORG":
                return "org"
            if ent.label_ in {"WORK_OF_ART", "PRODUCT"}:
                return "work"
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                return "place"

    if any(word in st for word in ["film", "movie", "album", "song", "book", "series", "show"]):
        return "work"
    if any(word in st for word in ["company", "organization", "corporation", "inc", "ltd", "band", "records", "studios"]):
        return "org"

    return "unknown"

# ============================================================
#  PREDICATE BUILDER
# ============================================================

def build_simple_predicate(root):
    lemma = root.lemma_.lower()
    if lemma == "bear":
        return "born"
    if lemma == "die":
        return "died"
    if lemma == "work":
        prep_map = get_root_prep_map(root)
        return "worked_as" if "as" in prep_map else "worked"
    if lemma == "serve":
        prep_map = get_root_prep_map(root)
        return "served_as" if "as" in prep_map else "served"
    if lemma == "appear":
        return "appeared_in"
    if lemma == "star":
        return "stars_in"
    if lemma == "own":
        return "owned_by"
    if lemma in {"found", "cofounde", "establish"}:
        return "founded_by"
    if lemma == "win":
        return "won"
    if lemma == "receive":
        return "received"
    if lemma == "direct":
        return "directed_by"
    if lemma == "release":
        return "released"
    if lemma == "write":
        return "written_by"
    if lemma == "develop":
        return "developed_by"
    if lemma == "marry":
        return "married"
    if lemma == "play":
        return "played_in"
    if lemma == "portray":
        return "portrayed"
    if lemma == "produce":
        return "produced"
    if lemma == "form":
        return "formed"
    return lemma

# ============================================================
#  RULE MAPPERS
#  Each returns (property_id, subject_override, object_text)
# ============================================================

def map_birth_property(doc, root, subject_text):
    if root.lemma_.lower() != "bear":
        return None, None, None

    prep_map = get_root_prep_map(root)

    if "on" in prep_map:
        pobj = get_pobj_for_prep(prep_map["on"])
        if pobj is not None and (pobj.ent_type_ == "DATE" or has_date_entity(doc)):
            return "P569", None, get_date_entity(doc)

    if "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None and pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
            return "P19", None, get_best_entity_or_span(pobj)

    if has_date_entity(doc):
        return "P569", None, get_date_entity(doc)

    place = get_last_entity_of_types(doc, {"GPE", "LOC", "FAC"})
    if place:
        return "P19", None, place

    return "P19", None, None

def map_death_property(doc, root, subject_text):
    if root.lemma_.lower() != "die":
        return None, None, None

    prep_map = get_root_prep_map(root)

    if "on" in prep_map or has_date_entity(doc):
        return "P570", None, get_date_entity(doc)

    if "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None and pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
            return "P20", None, get_best_entity_or_span(pobj)

    return "P570", None, None

def map_passive_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    prep_map = get_root_prep_map(root)

    if "by" not in prep_map:
        return None, None, None

    pobj = get_pobj_for_prep(prep_map["by"])
    by_obj = get_best_entity_or_span(pobj)

    if lemma == "direct":
        return "P57", None, by_obj
    if lemma in {"write", "author"}:
        return "P58", None, by_obj
    if lemma in {"found", "establish", "start", "create", "cofound", "cofounde"}:
        return "P112", None, by_obj
    if lemma in {"develop", "produce"}:
        return "P178", None, by_obj
    if lemma == "own":
        return "P127", None, by_obj

    return None, None, None

def map_active_transitive_property(doc, root, subject_text):
    """
    Active voice mapping with KG-facing direction:
      P57  : work -> director
      P58  : work -> screenwriter
      P112 : org/work -> founder
      P178 : work/product -> developer/producer
      P161 : work -> cast member / narrator / host
    """
    lemma = root.lemma_.lower()
    prep_map = get_root_prep_map(root)
    subj_type = infer_subject_type(doc, subject_text)

    if "by" in prep_map:
        return None, None, None

    dobj = get_direct_object_token(root)
    if dobj is None:
        return None, None, None

    obj_text = improve_object_text(dobj)
    obj_type = dobj.ent_type_

    # director: person directed work  ->  work, person
    if lemma == "direct":
        if subj_type in {"person", "unknown"} and obj_type in {"WORK_OF_ART", "PRODUCT", "ORG"}:
            return "P57", obj_text, subject_text

    # screenwriter: person wrote/authored work  ->  work, person
    if lemma in {"write", "author", "cowrite", "cowrote", "screenwrite"}:
        if subj_type in {"person", "unknown"} and obj_type in {"WORK_OF_ART", "PRODUCT", "ORG"}:
            return "P58", obj_text, subject_text

    # founded by: person founded org  ->  org, person
    if lemma in {"found", "establish", "cofounde", "cofound", "form", "start", "create"}:
        if subj_type in {"person", "unknown"} and obj_type in {"ORG", "PRODUCT"}:
            return "P112", obj_text, subject_text

    # producer/developer: person/org developed/produced work  ->  work, person/org
    if lemma in {"develop", "produce"}:
        if subj_type in {"person", "org", "unknown"} and obj_type in {"WORK_OF_ART", "PRODUCT", "ORG"}:
            return "P178", obj_text, subject_text

    # OWNED BY can stay forward if subject is thing/org and object is owner
    if lemma == "own":
        return "P127", None, obj_text

    # narrator/host/cast-like: person hosted/narrated work  ->  work, person
    if lemma == "narrate":
        if subj_type in {"person", "unknown"} and obj_type in {"WORK_OF_ART", "PRODUCT", "ORG"}:
            return "P161", obj_text, subject_text

    if lemma == "host":
        if subj_type in {"person", "unknown"} and obj_type in {"WORK_OF_ART", "PRODUCT", "ORG"}:
            return "P161", obj_text, subject_text

    return None, None, None

def map_marriage_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    if lemma not in {"marry", "wed"}:
        return None, None, None

    dobj = get_direct_object_token(root)
    if dobj is not None:
        return "P26", None, get_best_entity_or_span(dobj)

    prep_map = get_root_prep_map(root)
    if "to" in prep_map:
        pobj = get_pobj_for_prep(prep_map["to"])
        if pobj is not None:
            return "P26", None, get_best_entity_or_span(pobj)

    return "P26", None, None

def map_member_of_property(doc, root, subject_text):
    if root.lemma_.lower() != "be":
        return None, None, None

    attr_tok = get_attr_token(root)
    if attr_tok is None:
        return None, None, None

    if attr_tok.text.lower() == "member":
        for child in attr_tok.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                pobj = get_pobj_for_prep(child)
                if pobj is not None:
                    return "P463", None, get_best_entity_or_span(pobj)

    if attr_tok.text.lower() == "part":
        for child in attr_tok.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                pobj = get_pobj_for_prep(child)
                if pobj is not None:
                    return "P361", None, get_best_entity_or_span(pobj)

    return None, None, None

def map_award_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    if lemma not in {"win", "receive"}:
        return None, None, None

    dobj = get_direct_object_token(root)
    if dobj is None:
        return None, None, None

    obj_text = get_best_entity_or_span(dobj)
    if obj_text is None:
        return None, None, None

    non_award_words = {
        "election", "vote", "game", "match", "race", "season",
        "title", "war", "battle", "verdict", "sentence"
    }
    if obj_text.lower().split()[0] in non_award_words:
        return None, None, None

    return "P166", None, obj_text

def map_appositive_property(doc):
    """
    Example:
      'John, an actor, ...' -> occupation
      'Marie Curie, a Polish scientist, ...' -> occupation or citizenship
    """
    for tok in doc:
        if tok.dep_ == "appos":
            head = tok.head
            subj = get_best_entity_or_span(head)
            app_text_raw = get_best_entity_or_span(tok)
            app_text = (app_text_raw or "").lower()
            if not subj or not app_text:
                continue

            occ = normalize_occupation_text(app_text_raw)
            if occ and occ in OCCUPATIONS:
                return subj, "P106", occ

            # nationality only if subject looks like person
            subj_type = infer_subject_type(tok.doc, subj)
            if subj_type == "person":
                for nat in sorted(NATIONALITY_WORDS, key=len, reverse=True):
                    if nat in app_text:
                        return subj, "P27", NATIONALITY_TO_COUNTRY.get(nat, nat)

    return None, None, None

def extract_nationality_from_text(text):
    if not text:
        return None
    t = text.lower()
    for nat in sorted(NATIONALITY_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(nat)}\b", t):
            return NATIONALITY_TO_COUNTRY.get(nat, nat)
    return None

def map_copular_property(doc, root, subject_text):
    """
    Important fix:
    - nationality -> P27 only if subject looks like a person
    - role objects are canonicalized
    - work nationality goes to P495
    """
    if root.lemma_.lower() != "be":
        return None, None, None

    # participial passive inside copular
    for child in root.children:
        if child.tag_ in {"VBN", "VBD"} and child.dep_ not in {"nsubj", "nsubjpass"}:
            child_lemma = child.lemma_.lower()
            child_prep_map = {gc.text.lower(): gc for gc in child.children if gc.dep_ == "prep"}
            if "by" in child_prep_map:
                pobj = get_pobj_for_prep(child_prep_map["by"])
                by_obj = get_best_entity_or_span(pobj)
                if child_lemma == "own":
                    return "P127", None, by_obj
                if child_lemma in {"found", "establish", "create", "start", "cofounde"}:
                    return "P112", None, by_obj
                if child_lemma == "direct":
                    return "P57", None, by_obj
                if child_lemma in {"write", "author"}:
                    return "P58", None, by_obj
                if child_lemma in {"develop", "produce"}:
                    return "P178", None, by_obj

    prep_map = get_root_prep_map(root)
    attr_tok = get_attr_token(root)
    subj_type = infer_subject_type(doc, subject_text)

    if "in" in prep_map and any(tok.text.lower() in {"headquartered", "based"} for tok in doc):
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            return "P159", None, get_best_entity_or_span(pobj)

    if "from" in prep_map:
        pobj = get_pobj_for_prep(prep_map["from"])
        if pobj is not None and pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
            if subj_type == "person":
                return "P27", None, get_best_entity_or_span(pobj)
            if subj_type == "work":
                return "P495", None, get_best_entity_or_span(pobj)

    for prep_word in ("in", "within", "inside"):
        if prep_word in prep_map:
            pobj = get_pobj_for_prep(prep_map[prep_word])
            if pobj is not None and pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
                return "P131", None, get_best_entity_or_span(pobj)

    if attr_tok is None:
        return None, None, None

    attr_text_raw = get_best_entity_or_span(attr_tok)
    attr_text = (attr_text_raw or "").lower()

    # occupation first
    occ = normalize_occupation_text(attr_text_raw)
    if occ and subj_type == "person":
        return "P106", None, occ

    # nationality only for person
    nat = extract_nationality_from_text(attr_text_raw)
    if nat and subj_type == "person":
        return "P27", None, nat

    # country of origin for works/media
    if nat and subj_type == "work":
        return "P495", None, nat

    # role/position
    pos = canonicalize_position_text(attr_text_raw)
    if pos and subj_type in {"person", "unknown"}:
        # only allow clean / canonical positions
        if pos in POSITION_CANONICAL.values() or len(pos.split()) <= 2:
            return "P39", None, pos

    if attr_tok.text.lower() == "member":
        for child in attr_tok.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                pobj = get_pobj_for_prep(child)
                if pobj is not None:
                    return "P463", None, get_best_entity_or_span(pobj)

    if attr_tok.text.lower() == "part":
        for child in attr_tok.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                pobj = get_pobj_for_prep(child)
                if pobj is not None:
                    return "P361", None, get_best_entity_or_span(pobj)

    if "industry" in attr_text:
        cleaned = attr_text.replace("industry", "").strip(" ,.-")
        if cleaned:
            return "P452", None, cleaned

    if subj_type == "work":
        for genre_word in ["jazz", "rock", "pop", "drama", "comedy", "thriller", "hip hop", "rap"]:
            if re.search(rf"\b{re.escape(genre_word)}\b", attr_text):
                return "P136", None, genre_word

    return None, None, None

def map_nominal_property(doc, root, subject_text):
    if root.lemma_.lower() != "be":
        return None, None, None

    attr_tok = get_attr_token(root)
    prep_map = get_root_prep_map(root)
    subj_type = infer_subject_type(doc, subject_text)

    if "in" in prep_map and any(tok.text.lower() in {"headquartered", "based"} for tok in doc):
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            return "P159", None, get_best_entity_or_span(pobj)

    if attr_tok is None:
        return None, None, None

    attr_text = (get_best_entity_or_span(attr_tok) or "").lower()

    if "genre" in attr_text:
        obj = attr_text.replace("genre", "").strip()
        if obj:
            return "P136", None, obj

    if "industry" in attr_text:
        obj = attr_text.replace("industry", "").strip()
        if obj:
            return "P452", None, obj

    if subj_type == "work":
        nat = extract_nationality_from_text(attr_text)
        if nat:
            return "P495", None, nat
        for genre_word in ["jazz", "rock", "pop", "drama", "comedy", "thriller", "hip hop", "rap"]:
            if re.search(rf"\b{re.escape(genre_word)}\b", attr_text):
                return "P136", None, genre_word

    return None, None, None

def map_prepositional_relation(doc, root, subject_text):
    lemma = root.lemma_.lower()
    prep_map = get_root_prep_map(root)

    if lemma == "base" and "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            return "P159", None, get_best_entity_or_span(pobj)

    if lemma in {"headquarter", "locate"} and "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            return "P159", None, get_best_entity_or_span(pobj)

    if lemma in {"own", "acquire"} and "by" in prep_map:
        pobj = get_pobj_for_prep(prep_map["by"])
        if pobj is not None:
            return "P127", None, get_best_entity_or_span(pobj)

    if lemma == "belong" and "to" in prep_map:
        pobj = get_pobj_for_prep(prep_map["to"])
        if pobj is not None:
            return "P127", None, get_best_entity_or_span(pobj)

    if lemma == "sign":
        for p in ("to", "with"):
            if p in prep_map:
                pobj = get_pobj_for_prep(prep_map[p])
                if pobj is not None:
                    return "P264", None, get_best_entity_or_span(pobj)

    if lemma == "serve" and "as" in prep_map:
        pobj = get_pobj_for_prep(prep_map["as"])
        if pobj is not None:
            role = canonicalize_position_text(get_best_entity_or_span(pobj))
            return "P39", None, role

    if lemma == "work":
        if "as" in prep_map:
            pobj = get_pobj_for_prep(prep_map["as"])
            if pobj is not None:
                occ = normalize_occupation_text(get_best_entity_or_span(pobj))
                return "P106", None, occ
        if "for" in prep_map:
            pobj = get_pobj_for_prep(prep_map["for"])
            if pobj is not None:
                return "P463", None, get_best_entity_or_span(pobj)

    return None, None, None

def map_play_role_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    if lemma not in {"play", "portray", "feature"}:
        return None, None, None

    dobj = get_direct_object_token(root)
    prep_map = get_root_prep_map(root)

    # "X played in Y" -> Y has cast member X
    if subject_text and "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            work_title = get_best_entity_or_span(pobj)
            if work_title:
                return "P161", work_title, subject_text

    # "Y features X" -> Y has cast member X
    if lemma == "feature" and subject_text and dobj is not None:
        subj_type = infer_subject_type(doc, subject_text)
        if subj_type == "work":
            return "P161", subject_text, get_best_entity_or_span(dobj)

    return None, None, None

def map_cast_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    if lemma not in {"star", "feature"}:
        return None, None, None

    prep_map = get_root_prep_map(root)

    # "X stars in Y"  ->  Y has cast member X
    if subject_text and "in" in prep_map:
        pobj = get_pobj_for_prep(prep_map["in"])
        if pobj is not None:
            work_title = get_best_entity_or_span(pobj)
            if work_title:
                return "P161", work_title, subject_text

    # fallback if subject itself is work and object is person
    dobj = get_direct_object_token(root)
    if subject_text and dobj is not None:
        obj_text = get_best_entity_or_span(dobj)
        subj_type = infer_subject_type(doc, subject_text)
        if subj_type == "work":
            return "P161", subject_text, obj_text

    return None, None, None

def map_record_label_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    if lemma not in {"release", "sign", "drop", "put"}:
        return None, None, None

    prep_map = get_root_prep_map(root)
    for prep_word in ("on", "to", "with", "through", "via"):
        if prep_word in prep_map:
            pobj = get_pobj_for_prep(prep_map[prep_word])
            if pobj is not None and pobj.ent_type_ == "ORG":
                return "P264", None, get_best_entity_or_span(pobj)

    return None, None, None

def map_location_property(doc, root, subject_text):
    lemma = root.lemma_.lower()
    prep_map = get_root_prep_map(root)

    if lemma in {"lie", "locate", "situate", "reside", "exist", "base"}:
        for prep_word in ("in", "within", "inside", "at"):
            if prep_word in prep_map:
                pobj = get_pobj_for_prep(prep_map[prep_word])
                if pobj is not None and pobj.ent_type_ in {"GPE", "LOC", "FAC", "ORG"}:
                    return ("P159" if pobj.ent_type_ == "FAC" else "P131"), None, get_best_entity_or_span(pobj)

    if lemma == "be":
        attr_tok = get_attr_token(root)
        if attr_tok is not None and attr_tok.text.lower() == "part":
            for child in attr_tok.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    pobj = get_pobj_for_prep(child)
                    if pobj is not None:
                        return "P131", None, get_best_entity_or_span(pobj)

    return None, None, None

def map_have_property(doc, root, subject_text):
    if root.lemma_.lower() != "have":
        return None, None, None

    verbal_child = get_xcomp_or_ccomp_child(root)
    if verbal_child is None:
        for child in root.children:
            if child.dep_ in {"acl", "relcl", "oprd"} and child.pos_ in {"VERB", "AUX"}:
                verbal_child = child
                break

    if verbal_child is not None:
        vlemma = verbal_child.lemma_.lower()
        prep_map2 = get_root_prep_map(verbal_child)
        dobj2 = get_direct_object_token(verbal_child)

        if vlemma == "work" and "as" in prep_map2:
            pobj = get_pobj_for_prep(prep_map2["as"])
            if pobj is not None:
                return "P106", None, normalize_occupation_text(get_best_entity_or_span(pobj))

        if vlemma == "serve" and "as" in prep_map2:
            pobj = get_pobj_for_prep(prep_map2["as"])
            if pobj is not None:
                return "P39", None, canonicalize_position_text(get_best_entity_or_span(pobj))

        if vlemma == "marry":
            if dobj2 is not None:
                return "P26", None, get_best_entity_or_span(dobj2)
            if "to" in prep_map2:
                pobj = get_pobj_for_prep(prep_map2["to"])
                if pobj is not None:
                    return "P26", None, get_best_entity_or_span(pobj)

        if vlemma == "win" and dobj2 is not None:
            return "P166", None, get_best_entity_or_span(dobj2)

        if vlemma == "direct" and dobj2 is not None:
            return "P57", improve_object_text(dobj2), subject_text

        if vlemma in {"write", "author"} and dobj2 is not None:
            return "P58", improve_object_text(dobj2), subject_text

        if vlemma in {"produce", "develop"} and dobj2 is not None:
            return "P178", improve_object_text(dobj2), subject_text

    dobj = get_direct_object_token(root)
    if dobj is not None:
        obj_text = (get_best_entity_or_span(dobj) or "").lower()
        if "headquarters" in obj_text:
            place = get_last_entity_of_types(doc, {"GPE", "LOC", "FAC"})
            if place:
                return "P159", None, place
        if "industry" in obj_text:
            cleaned = obj_text.replace("industry", "").strip()
            return "P452", None, cleaned or None
        if "genre" in obj_text:
            cleaned = obj_text.replace("genre", "").strip()
            return "P136", None, cleaned or None

    return None, None, None

# ============================================================
#  SEMANTIC ASSIST
# ============================================================

def build_relation_phrase(doc, root):
    pieces = [root.lemma_.lower()]
    attr_tok = get_attr_token(root)
    if attr_tok is not None:
        attr_text = get_best_entity_or_span(attr_tok)
        if attr_text:
            pieces.append(attr_text.lower())

    prep_map = get_root_prep_map(root)
    for p, prep_tok in prep_map.items():
        pieces.append(p)
        pobj = get_pobj_for_prep(prep_tok)
        if pobj is not None:
            if pobj.ent_type_ == "DATE":
                pieces.append("date")
            elif pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
                pieces.append("place")
            elif pobj.ent_type_ == "ORG":
                pieces.append("organization")
            elif pobj.ent_type_ == "PERSON":
                pieces.append("person")
            else:
                pobj_text = get_best_entity_or_span(pobj)
                if pobj_text:
                    pieces.append(pobj_text.lower())

    for child in root.children:
        if child.tag_ in {"VBN", "VBD", "VB", "VBG"} and child.dep_ not in {"nsubj", "nsubjpass"}:
            pieces.append(child.lemma_.lower())

    return " ".join(dict.fromkeys([p for p in pieces if p]))

def semantic_property_match(relation_phrase, threshold=SEMANTIC_THRESHOLD):
    if not relation_phrase:
        return None, 0.0

    query_emb = SEM_MODEL.encode(relation_phrase, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, PROPERTY_EMBS)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return None, best_score

    return _PROPERTY_IDS[best_idx], best_score

def property_object_is_compatible(property_id, object_text, subject_type=None):
    if not property_id or not object_text:
        return False

    obj = object_text.lower()

    if property_id == "P106":
        return normalize_occupation_text(object_text) is not None

    if property_id == "P27":
        return subject_type == "person" and (
            object_text in NATIONALITY_TO_COUNTRY.values() or obj in NATIONALITY_WORDS
        )

    if property_id == "P39":
        return canonicalize_position_text(object_text) is not None

    if property_id in {"P57", "P58", "P112", "P127", "P463", "P264"}:
        return len(object_text.split()) >= 1 and len(object_text.split()) <= 8

    if property_id in {"P131", "P159", "P19", "P20", "P495"}:
        return len(object_text.split()) >= 1

    if property_id == "P178":
        # stricter than before
        return len(object_text.split()) >= 1 and len(object_text.split()) <= 6

    return True

def extract_object_for_property(doc, root, property_id, subject_text):
    subject_type = infer_subject_type(doc, subject_text)

    if property_id in {"P19", "P20", "P131", "P159", "P495"}:
        val = get_last_entity_of_types(doc, {"GPE", "LOC", "FAC"})
        if val:
            return val

    if property_id == "P27" and subject_type == "person":
        nat = extract_nationality_from_text(doc.text)
        if nat:
            return nat
        val = get_last_entity_of_types(doc, {"GPE", "LOC", "FAC"})
        if val:
            return val

    if property_id in {"P57", "P58", "P112", "P127", "P463", "P264"}:
        ents = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "WORK_OF_ART", "PRODUCT"}]
        # avoid taking subject again if possible
        for ent in ents:
            if subject_text and ent.lower() != subject_text.lower():
                return ent
        if ents:
            return ents[0]

    if property_id == "P106":
        attr_tok = get_attr_token(root)
        if attr_tok is not None:
            occ = normalize_occupation_text(get_best_entity_or_span(attr_tok))
            if occ:
                return occ

        prep_map = get_root_prep_map(root)
        if "as" in prep_map:
            pobj = get_pobj_for_prep(prep_map["as"])
            if pobj is not None:
                return normalize_occupation_text(get_best_entity_or_span(pobj))

    if property_id == "P39":
        prep_map = get_root_prep_map(root)
        if "as" in prep_map:
            pobj = get_pobj_for_prep(prep_map["as"])
            if pobj is not None:
                return canonicalize_position_text(get_best_entity_or_span(pobj))

        attr_tok = get_attr_token(root)
        if attr_tok is not None:
            return canonicalize_position_text(get_best_entity_or_span(attr_tok))

    if property_id == "P136":
        attr_tok = get_attr_token(root)
        if attr_tok is not None:
            return get_best_entity_or_span(attr_tok)

    if property_id == "P166":
        dobj = get_direct_object_token(root)
        if dobj is not None:
            return get_best_entity_or_span(dobj)

    return None

def map_semantic_relation(doc, root, subject_text):
    relation_phrase = build_relation_phrase(doc, root)
    property_id, semantic_score = semantic_property_match(relation_phrase)

    if property_id is None:
        return None, None, None, relation_phrase

    object_text = extract_object_for_property(doc, root, property_id, subject_text)
    if property_id == "P106":
        object_text = normalize_occupation_text(object_text)
    elif property_id == "P39":
        object_text = canonicalize_position_text(object_text)
    else:
        object_text = normalize_entity_text(object_text)

    subject_type = infer_subject_type(doc, subject_text)
    if not property_object_is_compatible(property_id, object_text, subject_type):
        return None, None, None, relation_phrase

    return property_id, None, object_text, relation_phrase if property_id else None

# ============================================================
#  CONFIDENCE + ROUTING
# ============================================================

def get_confidence(property_id, object_text):
    if not property_id or not object_text:
        return "low"
    if property_id in HIGH_CONF_PROPS:
        return "high"
    if property_id in MEDIUM_CONF_PROPS:
        return "medium"
    return "low"

def route_claim(subject, predicate, obj, property_id):
    """
    Important fix:
    - Only canonicalizable properties go to kg_exact_match
    - Other property-mapped claims go to kg_retrieve so you can compare
      extracted object vs KG values later instead of forcing bad exact matches
    """
    if subject is None:
        return "semantic_search"
    if property_id and obj and property_id in EXACT_MATCH_READY_PROPS:
        return "kg_exact_match"
    if property_id:
        return "kg_retrieve"
    return "semantic_search"

# ============================================================
#  MAIN EXTRACTION
# ============================================================

# final cleanup fnc:for objects
def finalize_object_for_kg(property_id, object_text, subject_text=None):
    if not object_text:
        return None
    
    text = normalize_entity_text(object_text)
    if not text:
        return None

    t = text.lower().strip()

     # keep clean single-word objects early
    if len(text.split()) == 1:
        if property_id in {"P106", "P39", "P27", "P19", "P20", "P569", "P570", "P166", "P136"}:
            return text

    # drop obvious bad phrases
    BAD_PATTERNS = [
        "network that", "person who", "killing of", "name ", "who were",
        "song on", "television network", "satellite television network"
    ]
    if any(p in t for p in BAD_PATTERNS):
        return None

    if property_id == "P106":
        occ = normalize_occupation_text(text)
        if occ:
            return occ

        # fallback: last word heuristic
        tokens = text.lower().split()
        if tokens:
            return tokens[-1]

        # fallback: extract last meaningful word
        tokens = t.split()
        if tokens:
            return tokens[-1]
        
        return None

    if property_id == "P39":
        pos = canonicalize_position_text(text)
        if pos:
            return pos

        tokens = t.split()
        if tokens:
            return tokens[-1]
        return None

    if property_id == "P27":
        nat = extract_nationality_from_text(text)
        if nat:
            return nat
        if text in NATIONALITY_TO_COUNTRY.values():
            return text
        # allow country-looking entity only if already canonical
        return text if len(text.split()) <= 3 else None

    if property_id in {"P569", "P570", "P19", "P20", "P131", "P159", "P166", "P264", "P136", "P463", "P361", "P26", "P127", "P495"}:
        return text

    if property_id in {"P57", "P58", "P112", "P161", "P178"}:
        # stricter: keep only short title/name-like objects
        if len(text.split()) <= 5:
            return text
        return None

    # if property is missing, don't keep random claim fragments for KG use
    if property_id is None:
        return None
    
    return text

def extract_claim(claim):
    cleaned = clean_claim(claim)
    doc = nlp(cleaned)
    root = get_root(doc)

    if root is None:
        return {
            "claim": claim,
            "subject": None,
            "predicate": None,
            "object": None,
            "property_id": None,
            "confidence": "low",
            "route": "semantic_search",
            "mapping_source": "none",
            "semantic_score": None,
            "relation_phrase": None,
        }

    app_subj, app_prop, app_obj = map_appositive_property(doc)
    if app_prop is not None:
        app_subj = normalize_entity_text(app_subj)
        app_obj = normalize_entity_text(app_obj)
        conf = get_confidence(app_prop, app_obj)
        return {
            "claim": claim,
            "subject": app_subj,
            "predicate": "appositive",
            "object": app_obj,
            "property_id": app_prop,
            "confidence": conf,
            "route": route_claim(app_subj, "appositive", app_obj, app_prop),
            "mapping_source": "rule",
            "semantic_score": None,
            "relation_phrase": None,
        }

    subject_tok = get_subject_token(doc, root)
    subject_text = resolve_subject_text(subject_tok)
    predicate = build_simple_predicate(root)

    property_id = None
    object_text = None
    subject_override = None
    mapping_source = "rule"
    semantic_score = None
    relation_phrase = None

    rule_functions = [
        map_birth_property,
        map_death_property,
        map_passive_property,
        map_active_transitive_property,
        map_have_property,
        map_award_property,
        map_marriage_property,
        map_member_of_property,
        map_copular_property,
        map_nominal_property,
        map_prepositional_relation,
        map_play_role_property,
        map_cast_property,
        map_record_label_property,
        map_location_property,
    ]

    for fn in rule_functions:
        pid, subj_over, obj = fn(doc, root, subject_text)
        if pid is not None:
            property_id = pid
            subject_override = subj_over
            object_text = obj
            break

    # semantic assist only if rules fail
    if property_id is None:
        pid, subj_over, obj, relation_phrase = map_semantic_relation(doc, root, subject_text)
        if pid is not None:
            property_id = pid
            subject_override = subj_over
            object_text = obj
            mapping_source = "semantic_assist"

    if subject_override:
        subject_text = subject_override

    # controlled fallback only if still missing object
    if object_text is None:
        if property_id == "P39":
            attr_tok = get_attr_token(root)
            if attr_tok is not None:
                object_text = canonicalize_position_text(get_best_entity_or_span(attr_tok))
        elif property_id == "P106":
            attr_tok = get_attr_token(root)
            if attr_tok is not None:
                object_text = normalize_occupation_text(get_best_entity_or_span(attr_tok))
        else:
            direct_obj = get_direct_object_token(root)
            if direct_obj is not None:
                object_text = improve_object_text(direct_obj)

    if object_text is None:
        attr_tok = get_attr_token(root)
        if attr_tok is not None and property_id not in {"P27"}:
            object_text = get_best_entity_or_span(attr_tok)

    # final normalization
    subject_text = normalize_entity_text(subject_text)


    confidence = get_confidence(property_id, object_text)
    route = route_claim(subject_text, predicate, object_text, property_id)

    return {
        "claim": claim,
        "subject": subject_text,
        "predicate": predicate,
        "object_raw" : object_text,
        "object_text" : finalize_object_for_kg(property_id, object_text, subject_text),
        "property_id": property_id,
        "confidence": confidence,
        "route": route,
        "mapping_source": mapping_source if property_id is not None else "none",
        "semantic_score": semantic_score,
        "relation_phrase": relation_phrase,
    }

# ============================================================
#  DIAGNOSTICS
# ============================================================

def diagnose_semantic(results_df):
    semantic_df = results_df[results_df["route"] == "semantic_search"].copy()
    if semantic_df.empty:
        print("\nNo semantic_search rows found.")
        return

    def safe_root_lemma(text):
        try:
            d = nlp(clean_claim(text))
            r = get_root(d)
            return r.lemma_.lower() if r else None
        except Exception:
            return None

    semantic_df["root_lemma"] = semantic_df["claim"].apply(safe_root_lemma)
    print("\nTop root lemmas in semantic_search:")
    print(semantic_df["root_lemma"].value_counts().head(30))

def summarize_results(results_df):
    print(f"Done. Processed {len(results_df)} claims.\n")

    print("Route distribution:")
    route_counts = results_df["route"].value_counts()
    for route, count in route_counts.items():
        print(f"  {route:<20} {count:>5}  ({100 * count / len(results_df):.1f}%)")

    print("\nConfidence distribution:")
    conf_counts = results_df["confidence"].value_counts()
    for conf, count in conf_counts.items():
        print(f"  {conf:<20} {count:>5}  ({100 * count / len(results_df):.1f}%)")

    if "mapping_source" in results_df.columns:
        print("\nMapping source distribution:")
        src_counts = results_df["mapping_source"].value_counts()
        for src, count in src_counts.items():
            print(f"  {src:<20} {count:>5}  ({100 * count / len(results_df):.1f}%)")

    if "property_id" in results_df.columns:
        print("\nTop properties:")
        print(results_df["property_id"].value_counts(dropna=False).head(20))

# ============================================================
#  BATCH RUNNER
# ============================================================

def process_claims_csv(input_csv, output_csv="extracted_claims_v7.csv", claim_col="claim"):
    df = pd.read_csv(input_csv)
    if claim_col not in df.columns:
        raise ValueError(f"Column '{claim_col}' not found. Available columns: {list(df.columns)}")

    records = []
    claims = df[claim_col].fillna("").tolist()

    for i, claim in enumerate(claims, 1):
        try:
            records.append(extract_claim(claim))
        except Exception as e:
            print(f"Error at row {i}: {claim}")
            raise
        if i % 100 == 0:
            print(f"Processed {i}/{len(claims)} claims...")
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)

    summarize_results(out_df)
    diagnose_semantic(out_df)
    print(f"\nSaved results to: {output_csv}")
    return out_df

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    INPUT_CSV = "D:/vidisha/STONY BROOK THINGSS/nlp/nlp_project/covered_claims_new.csv"
    OUTPUT_CSV = "extracted_claims_v4_new.csv"
    CLAIM_COL = "claim"
    process_claims_csv(INPUT_CSV, OUTPUT_CSV, CLAIM_COL)
