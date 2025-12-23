"""
The main directory should contain: candidates (containing cand, name_dict, neighbors, attributes) and test
Running this script will produce one output file in the outputs directory: eval.json
"""

import os
import json
import time
import traceback
from collections import defaultdict
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def get_llm_client():
    if OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        return client
    except Exception:
        return None


def llm_call(messages, temperature=0.2, max_tokens=512, model=MODEL_NAME):
    client = get_llm_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = getattr(resp, "usage", None)
        if usage:
            input_tok = getattr(usage, "prompt_tokens", 0)
            output_tok = getattr(usage, "completion_tokens", 0)
            total_tok = getattr(usage, "total_tokens", input_tok + output_tok)

            if not hasattr(llm_call, "token_usage"):
                llm_call.token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            llm_call.token_usage["total_tokens"] += total_tok

        return resp.choices[0].message.content
    except Exception:
        return None


def try_parse_json(txt):
    if not txt:
        return None
    txt = txt.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        buf = []
        in_code = False
        for ln in lines:
            if ln.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                buf.append(ln)
        txt = "\n".join(buf).strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            txt2 = txt.replace("\n", " ").replace("\t", " ")
            if txt2.endswith(","):
                txt2 = txt2[:-1]
            return json.loads(txt2)
        except Exception:
            return None

INPUT_DIR = "candidates"
OUTPUT_DIR = "outputs"
EVAL_PATH = os.path.join(OUTPUT_DIR, "eval.json")

CFG = {
    "w": {"emb": 0.2, "alias": 0.2, "attr": 0.2, "neigh": 0.2, "type": 0.2},
    "penalty_cap": 0.1,
    "expand_default": "expand",
}

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def id2name(ent_id, name_dict):
    return name_dict.get("ent", {}).get(str(ent_id), "")


def build_neighbor_names(ent_id, neighbors, name_dict, include_rel=True):
    out = []
    triples = neighbors.get(str(ent_id), [])
    rel_map = name_dict.get("rel", {})
    ent_map = name_dict.get("ent", {})
    for h, r, t in triples:
        rn = rel_map.get(str(r), str(r))
        if str(h) == str(ent_id):
            tn = ent_map.get(str(t), str(t))
            out.append(f"{rn}|{tn}" if include_rel else tn)
        else:
            hn = ent_map.get(str(h), str(h))
            out.append(f"{rn}|{hn}" if include_rel else hn)
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.5
    inter = len(sa & sb)
    union = len(sa | sb)
    return 0.0 if union == 0 else inter / union

PRIOR_KNOWLEDGE_CLAUSE = (
    " You can and should fully utilize your existing world knowledge and common sense (prior knowledge),"
    " to understand names/abbreviations/transliterations, infer types, interpret attributes, comprehend relational semantics and patterns,"
    " and identify common noise and ambiguities. Make judgments solely based on the given input and prior knowledge."
)


def build_alias_messages(src_name, cand_names):
    fewshot = [
        {
            "role": "system",
            "content": (
                    "You are an 'expert in alias equivalence judgment for entity alignment', proficient in recognizing, determining, and comparing entity aliases."
                    "Task: decide whether source_name and each candidate name refer to the same real-world entity."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "Prioritize: cross-lingual translations, historical or former names, abbreviations, multiple transliteration systems, stage names and nicknames, accent mark variations (São ↔ Sao), and common alternative names. If ambiguity remains based on names alone, use \"abstain\".\n\n"
                    "Only treat two names as cross-lingual translations if they refer to the *same real-world entity* expressed in different languages."
                    "Do NOT mark as cross-lingual translations if they merely belong to the same category, domain, or language family "
                    "(e.g., two different dialects, two distinct languages, or two organizations in similar fields).\n"
                    "Focus on the content of the name or alias rather than its format."
                    "Strictly output a JSON array: [{candidate_id, score, align, evidence}];"
                    "evidence ≤ 20 chars describing the rationale; score ∈ [0,1]; align ∈ {true, false, \"abstain\"}."
                    "Important: Judge solely based on names/aliases. No other information may influence your decision."
            )
        },
    ]
    user_task = {"source_name": src_name, "candidates": cand_names}
    return fewshot + [{"role": "user", "content": json.dumps(user_task, ensure_ascii=False)}]


def build_type_messages(src_pack, cand_packs):
    fewshot = [
        {
            "role": "system",
            "content": (
                    "You are an 'expert in type inference and consistency judgment for entity alignment', proficient in recognizing, determining, and comparing entity types. Task: determine the type of the source_entity and each candidate, and whether each candidate belongs to the same type as the source_entity."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "You should infer and choose the coarse type using entity name, entity attributes, neighbor token patterns, and your general prior knowledge."
                    "The entity type is coarsely divided in 5 types: {Person, Organization, Location, Event, Work(such as song, book, ...), Other}."
                    "Also you can decide your own types which are not provided such as {language}... , just depends on your confidence."
                    "If two entities are divided in same type, then they are \"same type\"."
                    "Then, provide a judgment on whether the source and candidate types are consistent: evidence(≤20 characters), score(0..1), align(true|false|\"abstain\")."
                    "\"same type\" means the source and candidate are of the same inferred type (e.g., both {Person}, both {Organization})."
                    "Scoring: same type = 1.0; related/near type ≈ 0.6-0.8; incompatible = 0.0."
                    "Strictly output a JSON array: [{candidate_id, score, align, evidence}]."
                    "Important:Focus ONLY on event type, no other information should influence your decision!"

                    "### Few-shot Abstract Examples ###\n\n"
                    "Example 1:\n"
                    "source_entity: '相模原市 [Location]'\n"
                    "candidates: ['Tang_Fei [Person]', 'Sagamihara [Location]', 'Sunderland_A.F.C. [Organization]']\n"
                    "Output: [\n"
                    "  {\"candidate_id\": 1, \"evidence\": \"location vs person\"}, \"score\": 0.0 \", align\": false\n"
                    "  {\"candidate_id\": 2, \"evidence\": \"same type: location\"}, \"score\": 1.0, \", align\": true,\n"
                    "  {\"candidate_id\": 3, \"evidence\": \"location vs organization\"}, \"score\": 0.0 \", align\": false\n"
                    "]\n\n"

                    "Example 2:\n"
                    "source_entity: '拜伦勋爵 [Person]'\n"
                    "candidates: ['Lord_Byron [Person]', 'University_of_Pisa [Location]', 'Bhutan_national_football_team [Organization]']\n"
                    "Output: [\n"
                    "  {\"candidate_id\": 1, \"evidence\": \"same type: person\"}, \"score\": 1.0 \", align\": true\n"
                    "  {\"candidate_id\": 2, \"evidence\": \"person vs location\"}, \"score\": 0.0, \", align\": false,\n"
                    "  {\"candidate_id\": 3, \"evidence\": \"person vs organization\"}, \"score\": 0.0 \", align\": false\n"
                    "]\n\n"

                    "Example 3:\n"
                    "source_entity: '伊斯法罕大學 [Organization]'\n"
                    "candidates: ['University_of_Isfahan [Organization]', 'John_Tukey [Person]', 'Alvin_Gentry [Person]']\n"
                    "Output: [\n"
                    "  {\"candidate_id\": 1, \"evidence\": \"same type: organization\"}, \"score\": 1.0 \", align\": true\n"
                    "  {\"candidate_id\": 2, \"evidence\": \"organization vs person\"}, \"score\": 0.0, \", align\": false,\n"
                    "  {\"candidate_id\": 3, \"evidence\": \"organization vs person\"}, \"score\": 0.0 \", align\": false\n"
                    "]\n\n"
                    "### End of Examples ###"
            )
        },
    ]

    def pack_for_llm(p):
        return {"name": p.get("name", ""),
                "attr_keys": list(p.get("attributes", {}).keys()),
                "neighbors": p.get("neighbors", [])}

    src_llm = pack_for_llm(src_pack)
    cand_llm = {cid: pack_for_llm(cinfo) for cid, cinfo in cand_packs.items()}
    user_task = {"source": src_llm, "candidates": cand_llm}
    return fewshot + [{"role": "user", "content": json.dumps(user_task, ensure_ascii=False)}]


def build_attr_messages(src_attr, cand_attrs):
    fewshot = [
        {
            "role": "system",
            "content": (
                    "You are an 'expert in attribute consistency adjudication for entity alignment', proficient in recognizing and comparing entity attributes."
                    "Task: determine whether the attributes of the source_entity and each candidate are consistent."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "Based on the entities' attributes, determine the consistency between the source and each candidate:"
                    "Consider format differences (e.g., YYYY vs. YYYY-MM-DD), close year values, missing/conflicting data, and common-sense co-occurrence of geographic attributes."
                    "\"fam\" and \"familycolor\" have the same meaning."
                    "When an entity's attribute value is unknown or unspecified, differences in that attribute should be disregarded."
                    "When the attribute value of a valid attribute is a statistical number, a certain degree of variation in the value is acceptable."
                    "Focus on the content of the attribute names or values, not on the language in which the attribute names or values are expressed."
                    "Differences in non-unique attributes between the source entity and candidate entities (such as workplaYou are an expert in neighborhood-structure and relational-pattern alignment for entity alignment. ce or works of the entity) are acceptable."
                    "When comparing attribute value, pay attention to the different expression of the same value, such as alias... ."
                    "Strictly output a JSON array: [{candidate_id, score, align, evidence}];"
                    "evidence ≤ 20 characters describing the rationale; score ∈ [0,1], depicts the score of align; align ∈ {true, false, \"abstain\"};."
                    "Important: Focus ONLY on attributes, no other information should influence your decision!"
                    "A score of 1.0 means you are very certain that the source entity and the candidate entity are definitely aligned; a score of 0 means you are very certain that the source entity and the candidate entity are definitely not aligned."
                    "The higher the score, the greater the likelihood you believe the two entities are aligned. When assigning a score, you should fully base it on your own evidence."
            )
        },
    ]
    user_task = {"source_attr": src_attr, "candidates_attr": cand_attrs}
    return fewshot + [{"role": "user", "content": json.dumps(user_task, ensure_ascii=False)}]


def build_neigh_messages(src_neighbors, cand_neighbors):
    fewshot = [
        {
            "role": "system",
            "content": (
                    "You are an 'expert in neighborhood structure and relational pattern alignment for entity alignment', proficient in domain relations and relational patterns within knowledge graphs."
                    "Task: determine whether the neighborhood structures and relational patterns of the source_entity and each candidate are consistent or highly similar."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "The input consists of structured neighbor tokens (e.g., 'located_in|France', 'member_of|UEFA')."
                    "You should use your prior understanding of common relation semantics and patterns to assess whether the neighborhoods are consistent or highly similar,"
                    "including: overlap of relation-value pairs, matching of key relations, and recognition of commonly equivalent relations, "
                    "rather than attributes."
                    "Differences in non-unique relations between the source entity and the candidate entities are acceptable."
                    "Strictly output a JSON array: [{candidate_id, score, align, evidence}];"
                    "evidence ≤ 20 chars describing the rationale; score ∈ [0,1]; align ∈ {true, false, \"abstain\"}."
                    "A score of 1.0 means you are very certain that the source entity and the candidate entity are definitely aligned; a score of 0 means you are very certain that the source entity and the candidate entity are definitely not aligned."
                    "Important: Focus ONLY from neighborhood structure and relational patterns, no other information should influence your decision!"
            )
        },
    ]
    user_task = {"source_neighbors": src_neighbors, "candidates_neighbors": cand_neighbors}
    return fewshot + [{"role": "user", "content": json.dumps(user_task, ensure_ascii=False)}]


def build_attack_messages(stage, main_outputs=None, inputs=None, history=None):
    fewshot = [
        {"role": "system",
         "content": (
                 "You are an 'expert in risk/weakness identification for entity alignment', proficient in identifying vulnerabilities in entity alignment."
                 + PRIOR_KNOWLEDGE_CLAUSE +
                 "Focus on the differences between the source entity and the candidate entities (e.g., they are of different types or have significant attribute discrepancies),"
                 "rather than on issues within a single entity's information (e.g., discontinuous dates or duplicated attributes)."
                 "The focus is on information inconsistency, not information omission."
                 "In cases where candidate entities have sparse or missing information, focus on the available data, and avoid penalizing candidates too harshly due to missing attributes."
                 "If only partial data is available, focus on identifying inconsistencies based on the existing attributes and try to quantify penalties appropriately."
                 "Integrate candidate information and (if available) outputs from other pathways to identify potential issues for each candidate and quantify penalties:"
                 "Strictly output a JSON array: [{candidate_id, issues, evidence, penalty, }]; penalty ∈ [0,0.1]; issues is an array of strings; evidence ≤ 50 characters."
         )
         }
    ]
    payload = {"stage": stage}
    if inputs is not None:
        payload["inputs"] = inputs
    if main_outputs is not None:
        payload["main_outputs"] = main_outputs
    if history is not None:
        payload["history"] = history
    return fewshot + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def build_judge_messages(stage, sims, alias_out, type_out, attr_out, neigh_out, attack_out, bucket, round_idx, inputs):
    fewshot = [
        {
            "role": "system",
            "content": (
                    "You are an expert in FINAL judgment synthesis for entity alignment. "
                    "Your role is to integrate the outputs of all expert agents and provide **minor score adjustments** and **alignment judgments** for each candidate."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "Based on various lines of evidence (embedding scores, alias, type, attribute, neighborhood, and attack penalties),"
                    "propose fine-tuning suggestions and alignment judgments for each candidate."
                    "Output JSON: {endorse: candidate_id, adjustments: [{candidate_id, delta, align, note}]}"
                    "where note ≤ 40 characters, delta ∈ [-0.1, 0.1] representing slight decreases or increases to each candidate's total score,"
                    "and align ∈ {true, false, \"abstain\"} representing whether you believe this candidate is the aligned entity."
                    "Note:Focus primarily on embedding scores, aliases and attributes."
                    "as they provide the most reliable basis for entity alignment confidence."
                    "The delta should be based on the scores and evidence provided by other experts, as well as your external knowledge."
                    "The higher the delta, the greater the likelihood of alignment."
                    "Provide an alignment judgment (align) for each candidate based on your overall assessment."
            )
        },
    ]
    payload = {
        "stage": stage, "bucket": bucket, "round": round_idx,
        "sims": sims, "alias": alias_out, "type": type_out,
        "attr": attr_out, "neigh": neigh_out, "attack": attack_out,
        "inputs": inputs
    }
    return fewshot + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def get_key_attributes_from_data(attributes_data, top_k=20):
    if not attributes_data:
        return {}
    attribute_freq = defaultdict(int)
    total_entities = 0

    USELESS_ATTRIBUTES = {'body', 'imageMap', 'imageSize'}

    for entity_id, attrs in attributes_data.items():
        if not attrs:
            continue
        total_entities += 1
        for attr_name in attrs.keys():
            if attr_name in USELESS_ATTRIBUTES:
                continue
            attribute_freq[attr_name] += 1

    attribute_coverage = {}
    for attr_name, count in attribute_freq.items():
        coverage = count / total_entities if total_entities > 0 else 0
        attribute_coverage[attr_name] = coverage

    sorted_attrs = sorted(attribute_coverage.items(), key=lambda x: x[1], reverse=True)

    key_attributes = [attr[0] for attr in sorted_attrs[:top_k]]

    if len(key_attributes) < top_k:
        common_attrs = [
            'birthDate', 'deathDate', 'founded', 'established', 'created',
            'latitude', 'longitude', 'country', 'city', 'location',
            'occupation', 'genre', 'type', 'category', 'industry',
            'population', 'area', 'elevation', 'members', 'size'
        ]
        for attr in common_attrs:
            if attr not in key_attributes and len(key_attributes) < top_k:
                key_attributes.append(attr)

    return set(key_attributes)


def get_key_relations_from_neighbors(neighbors_data, top_k=15):
    if not neighbors_data:
        return {}

    relation_freq = defaultdict(int)
    total_entities = 0

    for entity_id, triples in neighbors_data.items():
        if not triples:
            continue
        total_entities += 1
        for triple in triples:
            if len(triple) >= 2:
                relation_freq[triple[1]] += 1

    relation_coverage = {}
    for rel, count in relation_freq.items():
        coverage = count / total_entities if total_entities > 0 else 0
        relation_coverage[rel] = coverage

    sorted_rels = sorted(relation_coverage.items(), key=lambda x: x[1], reverse=True)

    key_relations = [rel[0] for rel in sorted_rels[:top_k]]

    if len(key_relations) < top_k:
        common_rels = {
            'located_in', 'part_of', 'member_of', 'child_of', 'spouse',
            'employer', 'founder', 'author', 'creator', 'capital_of',
            'contains', 'birth_place', 'death_place', 'educated_at',
            'works_at', 'participant_in', 'has_part'
        }
        for rel in common_rels:
            if rel not in key_relations and len(key_relations) < top_k:
                key_relations.append(rel)

    return set(key_relations)


KEY_ATTRIBUTES_CACHE = None
KEY_RELATIONS_CACHE = None


def compress_attributes(attributes, max_attrs=3):
    if not attributes:
        return {}

    global KEY_ATTRIBUTES_CACHE

    if KEY_ATTRIBUTES_CACHE is None:
        KEY_ATTRIBUTES_CACHE = {
            'birthDate', 'deathDate', 'founded', 'established', 'created',
            'latitude', 'longitude', 'country', 'city', 'location',
            'occupation', 'genre', 'type', 'category', 'industry',
            'population', 'area', 'elevation', 'members', 'size'
        }

    compressed = {}
    count = 0

    USELESS_ATTRIBUTES = {'body', 'imageMap', 'imageSize'}

    for key in KEY_ATTRIBUTES_CACHE:
        if key in attributes and count < max_attrs and key not in USELESS_ATTRIBUTES:
            compressed[key] = attributes[key]
            count += 1

    for key, value in attributes.items():
        if key not in compressed and count < max_attrs and key not in USELESS_ATTRIBUTES:
            value_str = str(value)
            if len(value_str) > 3 and len(value_str) < 100:
                compressed[key] = value
                count += 1

    return compressed


def compress_neighbors(neighbors, max_neighbors=3):
    if not neighbors:
        return []

    global KEY_RELATIONS_CACHE

    if KEY_RELATIONS_CACHE is None:
        KEY_RELATIONS_CACHE = {
            'located_in', 'part_of', 'member_of', 'child_of', 'spouse',
            'employer', 'founder', 'author', 'creator', 'capital_of',
            'contains', 'birth_place', 'death_place', 'educated_at',
            'works_at', 'participant_in', 'has_part'
        }

    important_neighbors = []
    other_neighbors = []

    for neighbor in neighbors:
        rel = neighbor.split('|')[0] if '|' in neighbor else ''
        if rel in KEY_RELATIONS_CACHE:
            important_neighbors.append(neighbor)
        else:
            other_neighbors.append(neighbor)

    result = important_neighbors + other_neighbors
    return result[:max_neighbors]


def initialize_key_elements(attributes_data, neighbors_data):
    global KEY_ATTRIBUTES_CACHE, KEY_RELATIONS_CACHE

    KEY_ATTRIBUTES_CACHE = get_key_attributes_from_data(attributes_data)
    KEY_RELATIONS_CACHE = get_key_relations_from_neighbors(neighbors_data)

def summarize_entity_for_basic_debate(entity_info):
    return {
        "id": entity_info["id"],
        "name": entity_info["name"],
        "attributes": compress_attributes(entity_info.get("attributes", {})),
        "neighbors": compress_neighbors(entity_info.get("neighbors", []))
    }


def build_proponent_messages_compressed(src_pack, cand_packs, round_idx, history=None):
    system_prompt = [
        {
            "role": "system",
            "content": (
                    "You are an objective alignment evaluator with a supportive perspective. "
                    "Task:Your task is to estimate how likely each candidate refers to the same real-world entity as the source."
                    "For each candidate, follow this reasoning procedure:"
                    "1. Identify concrete alignment signals from names, attributes, and relationships"
                    "2. assign a probability-like score based on all evidence"

                    "SCORING PRINCIPLES:\n"
                    "align_score represents the probability entities are the same:\n"
                    "0.9-1.0: Very high - strong consistent evidence\n"
                    "0.7-0.8: High - clear evidence with minor uncertainties\n"
                    "0.5-0.6: Moderate - mixed alignment indicators\n"
                    "0.3-0.4: Low - limited evidence with discrepancies\n"
                    "0.0-0.2: Very low - minimal convincing evidence\n\n"

                    "CRITICAL GUIDELINES:\n"
                    "- Explore alignment possibilities thoroughly\n"
                    "- Score objectively: higher score = higher probability\n"
                    "- Consider both supporting and contradictory evidence\n"
                    + PRIOR_KNOWLEDGE_CLAUSE +

                    "OUTPUT REQUIREMENTS:\n"
                    f"JSON array with ALL candidates: {list(cand_packs.keys())}\n"
                    "Format: [{\"candidate_id\": \"id\", \"align_score\": 0.x}, ...]\n"
                    "NO duplicate scores. Valid JSON required."
            )
        },
    ]

    compressed_cand_packs = {cid: summarize_entity_for_basic_debate(pack) for cid, pack in cand_packs.items()}
    compressed_src_pack = summarize_entity_for_basic_debate(src_pack)

    cand_ids = list(compressed_cand_packs.keys())
    cand_names = {cid: compressed_cand_packs[cid]["name"] for cid in compressed_cand_packs}
    cand_attrs = {cid: compressed_cand_packs[cid]["attributes"] for cid in compressed_cand_packs}
    cand_neighs = {cid: compressed_cand_packs[cid]["neighbors"] for cid in compressed_cand_packs}

    payload = {
        "src_name": compressed_src_pack["name"],
        "src_attr": compressed_src_pack["attributes"],
        "src_neigh": compressed_src_pack["neighbors"],
        "cand_ids": cand_ids,
        "cand_names": cand_names,
        "cand_attrs": cand_attrs,
        "cand_neighs": cand_neighs,
        "history": history
    }
    return system_prompt + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def build_opponent_messages_compressed(src_pack, cand_packs, round_idx, history=None):
    system_prompt = [
        {
            "role": "system",
            "content": (
                    "You are an objective alignment evaluator with a skeptical perspective."
                    "Task:Your task is to identify evidence that the source entity and each candidate are NOT the same real-world entity."
                    "For each candidate, follow this reasoning procedure:\n"
                    "1. Identify misalignment evidence from names, attributes, and relationships\n"
                    "2. Objectively assess alignment probability based on all evidence\n\n"

                    "SCORING PRINCIPLES:\n"
                    "align_score represents the probability entities are the same:\n"
                    "0.9-1.0: Very high - minimal discrepancies\n"
                    "0.7-0.8: High - strong evidence outweighs inconsistencies\n"
                    "0.5-0.6: Moderate - mixed evidence\n"
                    "0.3-0.4: Low - significant discrepancies\n"
                    "0.0-0.2: Very low - overwhelming misalignment\n\n"

                    "CRITICAL GUIDELINES:\n"
                    "- Identify potential discrepancies thoroughly\n"
                    "- Score objectively: higher score = higher probability\n"
                    "- Consider both alignment and misalignment evidence\n"
                    + PRIOR_KNOWLEDGE_CLAUSE +

                    "OUTPUT REQUIREMENTS:\n"
                    f"JSON array with ALL candidates: {list(cand_packs.keys())}\n"
                    "Format: [{\"candidate_id\": \"id\", \"align_score\": 0.x}, ...]\n"
                    "NO duplicate scores. Valid JSON required."
            )
        },
    ]

    compressed_cand_packs = {cid: summarize_entity_for_basic_debate(pack) for cid, pack in cand_packs.items()}
    compressed_src_pack = summarize_entity_for_basic_debate(src_pack)

    cand_ids = list(compressed_cand_packs.keys())
    cand_names = {cid: compressed_cand_packs[cid]["name"] for cid in compressed_cand_packs}
    cand_attrs = {cid: compressed_cand_packs[cid]["attributes"] for cid in compressed_cand_packs}
    cand_neighs = {cid: compressed_cand_packs[cid]["neighbors"] for cid in compressed_cand_packs}

    payload = {
        "src_name": compressed_src_pack["name"],
        "src_attr": compressed_src_pack["attributes"],
        "src_neigh": compressed_src_pack["neighbors"],
        "cand_ids": cand_ids,
        "cand_names": cand_names,
        "cand_attrs": cand_attrs,
        "cand_neighs": cand_neighs,
        "history": history
    }
    return system_prompt + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def build_referee_messages_for_debate_compressed(src_pack, cand_packs, round_idx, proponent_speech, opponent_speech,
                                                 history=None):
    system_prompt = [
        {
            "role": "system",
            "content": (
                    "You are the referee, responsible for assessing the alignment of entities based on the proponent and opponent's arguments. "
                    "Your task is to synthesize both sides' arguments and provide a balanced alignment score for each candidate entity."
                    + PRIOR_KNOWLEDGE_CLAUSE +
                    "CRITICAL GUIDELINES FOR REFEREE DECISION-MAKING:\n"
                    "1. CAREFULLY CONSIDER BOTH SIDES: You MUST carefully consider both the proponent's and opponent's scores and evidence for each candidate.\n"
                    "2. BALANCE THE EVIDENCE: Your align_score should reflect a balanced consideration of both viewpoints, not just one side.\n"
                    "3. HIGH PROPONENT + WEAK OPPONENT = HIGH SCORE: If proponent gives high score (≥0.7) with strong evidence and opponent provides weak or generic counter-evidence, assign a relatively high score (≥0.6).\n"
                    "4. STRONG OPPOSITION = LOWER SCORE: If opponent provides specific, factual counter-evidence (like 'different birth dates', 'conflicting locations'), even if proponent gives high score, you should lower the align_score appropriately.\n"
                    "5. BOTH WEAK = MODERATE SCORE: If both sides provide weak or generic evidence, assign a moderate score around 0.5.\n"
                    "6. CONSISTENT HIGH/LOW SCORES: If both proponent and opponent consistently give high/low scores to a candidate, your score should reflect this consensus.\n"
                    "7. EVIDENCE QUALITY MATTERS: Give more weight to specific, factual evidence over generic statements.\n\n"

                    "SCORING GUIDELINES:\n"
                    "- 0.8-1.0: Strong alignment - proponent provides strong evidence, opponent's objections are weak or irrelevant\n"
                    "- 0.6-0.7: Moderate alignment - proponent has good evidence, opponent has some valid concerns\n"
                    "- 0.4-0.5: Uncertain/Neutral - both sides have equally valid points, or evidence is insufficient\n"
                    "- 0.2-0.3: Weak alignment - opponent has strong counter-evidence, proponent's case is weak\n"
                    "- 0.0-0.1: No alignment - overwhelming counter-evidence from opponent\n\n"

                    "CRITICAL: You MUST output a JSON array containing EVERY candidate entity in the following list: " + str(
                list(cand_packs.keys())) + ".\n"
                                           "Do NOT omit any candidate. For each candidate, provide: candidate_id, align_score (0-1).\n"
                                           "Output format example: [{\"candidate_id\": \"123\", \"align_score\": 0.6}, ...]\n"
                                           "Ensure the output is valid JSON that can be parsed directly."
                                           "You MUST assign a different align_score to each candidate entity! No two candidates should have the same score!"
            )
        },
    ]

    compressed_cand_packs = {cid: summarize_entity_for_basic_debate(pack) for cid, pack in cand_packs.items()}
    compressed_src_pack = summarize_entity_for_basic_debate(src_pack)

    cand_ids = list(compressed_cand_packs.keys())
    cand_names = {cid: compressed_cand_packs[cid]["name"] for cid in compressed_cand_packs}
    cand_attrs = {cid: compressed_cand_packs[cid]["attributes"] for cid in compressed_cand_packs}
    cand_neighs = {cid: compressed_cand_packs[cid]["neighbors"] for cid in compressed_cand_packs}

    payload = {
        "src_name": compressed_src_pack["name"],
        "src_attr": compressed_src_pack["attributes"],
        "src_neigh": compressed_src_pack["neighbors"],
        "cand_ids": cand_ids,
        "cand_names": cand_names,
        "cand_attrs": cand_attrs,
        "cand_neighs": cand_neighs,
        "history": history,
        "proponent_arguments": proponent_speech,
        "opponent_arguments": opponent_speech
    }
    return system_prompt + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

def run_basic_debate_compressed(src_pack, cand_packs, round_count=3):
    final_scores = defaultdict(float)
    outs3 = []
    rounds_output = []

    all_candidate_ids = list(cand_packs.keys())

    for round_idx in range(1, round_count + 1):
        proponent_messages = build_proponent_messages_compressed(src_pack, cand_packs, round_idx, outs3)
        txt1 = llm_call(proponent_messages, temperature=0.1, max_tokens=800)
        parsed1 = try_parse_json(txt1) if txt1 else None
        outs1 = []

        if isinstance(parsed1, list):
            for rec in parsed1:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                outs1.append({"candidate_id": cid, "align_score": align_score})

            outs1 = ensure_all_candidates_covered(outs1, all_candidate_ids, 0.5, "Proponent not evaluated")
        else:
            for cid in all_candidate_ids:
                outs1.append({"candidate_id": cid, "align_score": 0.5})

        opponent_messages = build_opponent_messages_compressed(src_pack, cand_packs, round_idx, outs3)
        txt2 = llm_call(opponent_messages, temperature=0.1, max_tokens=800)
        parsed2 = try_parse_json(txt2) if txt2 else None
        outs2 = []

        if isinstance(parsed2, list):
            for rec in parsed2:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                outs2.append({"candidate_id": cid, "align_score": align_score})

            outs2 = ensure_all_candidates_covered(outs2, all_candidate_ids, 0.5, "Opponent not evaluated")
        else:
            for cid in all_candidate_ids:
                outs2.append({"candidate_id": cid, "align_score": 0.5})

        referee_messages = build_referee_messages_for_debate_compressed(src_pack, cand_packs, round_idx, outs1, outs2,
                                                                        outs3)
        txt3 = llm_call(referee_messages, temperature=0.1, max_tokens=1000)
        parsed3 = try_parse_json(txt3) if txt3 else None
        current_round_referee = []

        if isinstance(parsed3, list):
            for rec in parsed3:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                current_round_referee.append({
                    "candidate_id": cid,
                    "align_score": align_score
                })
                final_scores[cid] += align_score

            current_round_referee = ensure_all_candidates_covered(
                current_round_referee, all_candidate_ids, 0.5, "Referee not evaluated"
            )
        else:
            for cid in all_candidate_ids:
                current_round_referee.append({
                    "candidate_id": cid,
                    "align_score": 0.5
                })
                final_scores[cid] += 0.5

        outs3.extend(current_round_referee)

        rounds_output.append({
            "round": round_idx,
            "proponent": outs1,
            "opponent": outs2,
            "referee": current_round_referee
        })

    final_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "final_candidates": final_candidates,
        "rounds_output": rounds_output
    }


def ensure_all_candidates_covered(output_list, expected_candidate_ids, default_score=0.5, default_evidence="Not evaluated"):
    if not isinstance(output_list, list):
        output_list = []

    output_dict = {str(item.get("candidate_id", "")): item for item in output_list}
    complete_output = []

    for cid in expected_candidate_ids:
        cid_str = str(cid)
        if cid_str in output_dict:
            item = output_dict[cid_str]
            complete_output.append({
                "candidate_id": cid_str,
                "align_score": item.get("align_score", default_score)
            })
        else:
            complete_output.append({
                "candidate_id": cid_str,
                "align_score": default_score
            })

    return complete_output


def run_basic_debate(src_pack, cand_packs, round_count=3):
    final_scores = defaultdict(float)
    outs3 = []
    rounds_output = []

    all_candidate_ids = list(cand_packs.keys())

    for round_idx in range(1, round_count + 1):
        proponent_messages = build_proponent_messages_compressed(src_pack, cand_packs, round_idx, outs3)
        txt1 = llm_call(proponent_messages, temperature=0.1, max_tokens=800)
        parsed1 = try_parse_json(txt1) if txt1 else None
        outs1 = []

        if isinstance(parsed1, list):
            for rec in parsed1:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                outs1.append({"candidate_id": cid, "align_score": align_score})

            outs1 = ensure_all_candidates_covered(outs1, all_candidate_ids, 0.5, "Proponent not evaluated")
        else:
            for cid in all_candidate_ids:
                outs1.append({"candidate_id": cid, "align_score": 0.5})

        opponent_messages = build_opponent_messages_compressed(src_pack, cand_packs, round_idx, outs3)
        txt2 = llm_call(opponent_messages, temperature=0.1, max_tokens=800)
        parsed2 = try_parse_json(txt2) if txt2 else None
        outs2 = []

        if isinstance(parsed2, list):
            for rec in parsed2:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                outs2.append({"candidate_id": cid, "align_score": align_score})

            outs2 = ensure_all_candidates_covered(outs2, all_candidate_ids, 0.5, "Opponent not evaluated")
        else:
            for cid in all_candidate_ids:
                outs2.append({"candidate_id": cid, "align_score": 0.5})

        referee_messages = build_referee_messages_for_debate_compressed(src_pack, cand_packs, round_idx, outs1, outs2,
                                                                        outs3)
        txt3 = llm_call(referee_messages, temperature=0.1, max_tokens=1000)
        parsed3 = try_parse_json(txt3) if txt3 else None
        current_round_referee = []

        if isinstance(parsed3, list):
            for rec in parsed3:
                cid = str(rec.get("candidate_id"))
                align_score = rec.get("align_score", 0.5)
                try:
                    align_score = float(align_score)
                except Exception:
                    align_score = 0.5
                align_score = max(0.0, min(1.0, align_score))
                current_round_referee.append({
                    "candidate_id": cid,
                    "align_score": align_score
                })
                final_scores[cid] += align_score

            current_round_referee = ensure_all_candidates_covered(
                current_round_referee, all_candidate_ids, 0.5, "Referee not evaluated"
            )
        else:
            for cid in all_candidate_ids:
                current_round_referee.append({
                    "candidate_id": cid,
                    "align_score": 0.5
                })
                final_scores[cid] += 0.5

        outs3.extend(current_round_referee)

        rounds_output.append({
            "round": round_idx,
            "proponent": outs1,
            "opponent": outs2,
            "referee": current_round_referee
        })

    final_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "final_candidates": final_candidates,
        "rounds_output": rounds_output
    }


def check_if_enter_multirole_debate(top1_before_debate, top1_after_debate):
    if top1_before_debate == top1_after_debate:
        return False
    else:
        return True


def alias_agent(src_name, cand_names, history=None, judge_suggestion=None, attack_suggestion=None):
    messages = build_alias_messages(src_name, cand_names)
    txt = llm_call(messages, temperature=0.1, max_tokens=600)
    parsed = try_parse_json(txt)
    outs = []
    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        cand_set = set(cand_names.keys())
        for rec in parsed:
            cid = str(rec.get("candidate_id"))
            if cid not in cand_set:
                continue
            align = rec.get("align")
            if align not in (True, False, "abstain"):
                align = "abstain"
            score = rec.get("score", 0.5)
            try:
                score = float(score)
            except Exception:
                score = 0.5
            score = max(0.0, min(1.0, score))
            evidence = str(rec.get("evidence", ""))[:50]
            outs.append({"candidate_id": cid, "align": align, "score": round(score, 4), "evidence": evidence})
        miss = [cid for cid in cand_names if cid not in {x["candidate_id"] for x in outs}]
        outs += [{"candidate_id": cid, "align": "abstain", "score": 0.5, "evidence": "uncertain"} for cid in miss]
        return outs

    def norm(s):
        return str(s).lower().replace("·", "").replace("•", "").replace(" ", "")

    srcn = norm(src_name)
    for cid, cname in cand_names.items():
        cn = norm(cname)
        score = 1.0 if cn == srcn else (0.7 if (srcn in cn or cn in srcn) else 0.5)
        align = True if score >= 0.85 else (False if score <= 0.4 else "abstain")
        outs.append({"candidate_id": cid, "align": align, "score": round(score, 4), "evidence": "Fallback rule"})
    return outs


def type_agent(src_pack, cand_packs, history=None, judge_suggestion=None, attack_suggestion=None):
    messages = build_type_messages(src_pack, cand_packs)
    txt = llm_call(messages, temperature=0.1, max_tokens=800)
    parsed = try_parse_json(txt)
    outs = []
    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        cand_set = set(cand_packs.keys())
        for rec in parsed:
            cid = str(rec.get("candidate_id"))
            if cid not in cand_set:
                continue
            align = rec.get("align")
            if align not in (True, False, "abstain"):
                align = "abstain"
            score = rec.get("score", 0.6)
            try:
                score = float(score)
            except Exception:
                score = 0.6
            score = max(0.0, min(1.0, score))
            evidence = str(rec.get("evidence", ""))[:50]
            outs.append({"candidate_id": cid, "align": align, "score": round(score, 4), "evidence": evidence})
        miss = [cid for cid in cand_packs if cid not in {x["candidate_id"] for x in outs}]
        outs += [{"candidate_id": cid, "align": "abstain", "score": 0.6, "evidence": "uncertain"} for cid in miss]
        return outs

    def rough_type(attr_dict, name_str, neigh_list):
        keys = " ".join([k.lower() for k in attr_dict.keys()])
        nm = name_str
        neigh_s = " ".join(neigh_list).lower()
        if ("birthdate" in keys) or ("deathdate" in keys) or any(x in nm for x in ["先生", "女士", "总统", "总理"]):
            return "Person"
        if ("founded" in keys) or ("members" in keys) or ("club" in keys) or ("league" in keys) or (
                "serviceyears" in keys):
            return "Organization"
        if any(k in keys for k in
               ["latitude", "longitude", "latd", "latm", "longd", "longm", "areatotal", "poptotal", "postalcode"]) or (
                "located_in|" in neigh_s):
            return "Location"
        return "Other"

    src_t = rough_type(src_pack.get("attributes", {}), src_pack.get("name", ""), src_pack.get("neighbors", []))
    for cid, cinfo in cand_packs.items():
        cand_t = rough_type(cinfo.get("attributes", {}), cinfo.get("name", ""), cinfo.get("neighbors", []))
        if src_t == "Other" or cand_t == "Other":
            sc = 0.6
        elif src_t == cand_t:
            sc = 1.0
        else:
            sc = 0.0
        align = True if sc >= 0.85 else (False if sc <= 0.4 else "abstain")
        outs.append(
            {"candidate_id": cid, "align": align, "score": round(sc, 4), "evidence": f"{src_t} vs {cand_t} rule"})
    return outs


def attr_agent(src_attr, cand_attrs, history=None, judge_suggestion=None, attack_suggestion=None):
    messages = build_attr_messages(src_attr, cand_attrs)
    txt = llm_call(messages, temperature=0.1, max_tokens=800)
    parsed = try_parse_json(txt)
    outs = []
    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        cand_set = set(cand_attrs.keys())
        for rec in parsed:
            cid = str(rec.get("candidate_id"))
            if cid not in cand_set:
                continue
            align = rec.get("align")
            if align not in (True, False, "abstain"):
                align = "abstain"
            score = rec.get("score", 0.5)
            try:
                score = float(score)
            except Exception:
                score = 0.5
            score = max(0.0, min(1.0, score))
            evidence = str(rec.get("evidence", ""))[:50]
            outs.append({"candidate_id": cid, "align": align, "score": round(score, 4), "evidence": evidence})
        miss = [cid for cid in cand_attrs if cid not in {x["candidate_id"] for x in outs}]
        outs += [{"candidate_id": cid, "align": "abstain", "score": 0.5, "evidence": "uncertain"} for cid in miss]
        return outs
    for cid, attr in cand_attrs.items():
        s = 0.5
        ev = []
        sb = str(src_attr.get("birthDate", "")).strip()
        cb = str(attr.get("birthDate", "")).strip()
        if sb and cb:
            if sb == cb:
                s += 0.4;
                ev.append("DOB=1")
            elif sb[:4] == cb[:4]:
                s += 0.2;
                ev.append("DOB~year")
            else:
                s -= 0.3;
                ev.append("DOB≠")
        sd = str(src_attr.get("deathDate", "")).strip()
        cd = str(attr.get("deathDate", "")).strip()
        if sd and cd:
            if sd == cd:
                s += 0.2;
                ev.append("DOD=1")
            elif sd[:4] == cd[:4]:
                s += 0.1;
                ev.append("DOD~year")
            else:
                s -= 0.2;
                ev.append("DOD≠")
        src_geo = any(k in src_attr for k in
                      ["latitude", "latd", "latm", "longitude", "longd", "longm", "postalcode", "areatotal",
                       "poptotal"])
        cand_geo = any(k in attr for k in
                       ["latitude", "latd", "latm", "longitude", "longd", "longm", "postalcode", "areatotal",
                        "poptotal"])
        if src_geo and cand_geo:
            s += 0.1;
            ev.append("geo+")
        s = max(0.0, min(1.0, s))
        align = True if s >= 0.85 else (False if s <= 0.4 else "abstain")
        outs.append(
            {"candidate_id": cid, "align": align, "score": round(s, 4), "evidence": ",".join(ev) if ev else "No strong attributes"})
    return outs


def neigh_agent(src_neighbors, cand_neighbors, history=None, judge_suggestion=None, attack_suggestion=None):
    messages = build_neigh_messages(src_neighbors, cand_neighbors)
    txt = llm_call(messages, temperature=0.1, max_tokens=800)
    parsed = try_parse_json(txt)
    outs = []
    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        cand_set = set(cand_neighbors.keys())
        for rec in parsed:
            cid = str(rec.get("candidate_id"))
            if cid not in cand_set:
                continue
            align = rec.get("align")
            if align not in (True, False, "abstain"):
                align = "abstain"
            score = rec.get("score", 0.5)
            try:
                score = float(score)
            except Exception:
                score = 0.5
            score = max(0.0, min(1.0, score))
            evidence = str(rec.get("evidence", ""))[:50]
            outs.append({"candidate_id": cid, "align": align, "score": round(score, 4), "evidence": evidence})
        miss = [cid for cid in cand_neighbors if cid not in {x["candidate_id"] for x in outs}]
        outs += [{"candidate_id": cid, "align": "abstain", "score": 0.5, "evidence": "abstain"} for cid in miss]
        return outs
    sset = src_neighbors
    for cid, nlist in cand_neighbors.items():
        sc = jaccard(sset, nlist)
        align = True if sc >= 0.85 else (False if sc <= 0.4 else "abstain")
        overlaps = list(set(sset) & set(nlist))
        evidence = f"overlap={len(overlaps)}"
        outs.append({"candidate_id": cid, "align": align, "score": round(sc, 4), "evidence": evidence})
    return outs


def attack_agent(main_outputs=None, stage="R1", inputs=None, history=None):
    messages = build_attack_messages(stage, main_outputs=main_outputs, inputs=inputs, history=history)
    txt = llm_call(messages, temperature=0.1, max_tokens=800)
    parsed = try_parse_json(txt)
    outs = []
    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        for rec in parsed:
            cid = str(rec.get("candidate_id"))
            issues = rec.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            try:
                penalty = float(rec.get("penalty", 0.0))
            except Exception:
                penalty = 0.0
            penalty = max(0.0, min(CFG["penalty_cap"], penalty))
            evidence = str(rec.get("evidence", ""))[:60]
            outs.append({"candidate_id": cid, "issues": issues, "penalty": round(penalty, 4), "evidence": evidence})
        return outs
    idx = defaultdict(dict)
    if main_outputs:
        for ag, lst in main_outputs.items():
            for rec in lst:
                idx[rec["candidate_id"]][ag] = rec
    if not idx and inputs:
        sims_map = inputs.get("sims", {})
        for cid in inputs.get("cand_ids", []):
            issues = []
            penalty = 0.0
            simv = sims_map.get(cid, 0.0)
            if simv < 0.4:
                issues.append("low_sim");
                penalty += 0.05
            if not inputs.get("cand_names", {}).get(cid):
                issues.append("missing_name");
                penalty += 0.05
            attrs = inputs.get("cand_attrs", {}).get(cid, {})
            if not attrs:
                issues.append("attr_missing");
                penalty += 0.05
            penalty = min(CFG["penalty_cap"], penalty)
            outs.append({"candidate_id": cid, "issues": issues or ["ok"], "penalty": round(penalty, 4),
                         "evidence": "|".join(issues) if issues else "ok"})
        return outs
    for cid, per in idx.items():
        issues = []
        penalty = 0.0
        if "type" in per and per["type"]["score"] <= 0.2:
            issues.append("type_conflict");
            penalty += 0.05
        if "attr" in per and per["attr"]["score"] <= 0.3:
            issues.append("attr_weak");
            penalty += 0.05
        if "neigh" in per and per["neigh"]["score"] < 0.3:
            issues.append("near-zero_overlap");
            penalty += 0.05
        if "alias" in per and per["alias"]["score"] < 0.4:
            issues.append("alias_weak");
            penalty += 0.05
        penalty = min(CFG["penalty_cap"], penalty)
        outs.append({"candidate_id": cid, "issues": issues, "penalty": round(penalty, 4),
                     "evidence": "|".join(issues) if issues else "ok"})
    return outs


def _align_votes(main_outs_dict, candidate_id):
    agree = 0
    total = 0
    for k in ["alias", "type", "attr", "neigh"]:
        lst = main_outs_dict.get(k, [])
        m = next((x for x in lst if x["candidate_id"] == candidate_id), None)
        if m is None:
            continue
        total += 1
        if m["align"] is True:
            agree += 1
    return (agree, total)


def _apply_llm_judge_adjustments(stage, sims, alias_out, type_out, attr_out, neigh_out, attack_out, rows, bucket,
                                 round_idx, inputs):
    try:
        messages = build_judge_messages(stage, sims, alias_out, type_out, attr_out, neigh_out, attack_out, bucket,
                                        round_idx, inputs)
        txt = llm_call(messages, temperature=0.1, max_tokens=700)
        parsed = try_parse_json(txt)
        if not isinstance(parsed, dict):
            return rows, None, {}

        endor = str(parsed.get("endorse") or "")
        adjs = parsed.get("adjustments") or []

        adj_map = {}
        align_map = {}

        for it in adjs:
            cid = str(it.get("candidate_id") or "")

            try:
                delta = float(it.get("delta", 0.0))
            except Exception:
                delta = 0.0
            delta = max(-0.1, min(0.1, delta))
            adj_map[cid] = adj_map.get(cid, 0.0) + delta

            align = it.get("align")
            if align not in (True, False, "abstain"):
                align = "abstain"
            align_map[cid] = align

        for r in rows:
            cid = r["id"]
            if cid not in align_map:
                align_map[cid] = "abstain"
            if cid not in adj_map:
                adj_map[cid] = 0.0

        if adj_map:
            new_rows = []
            for r in rows:
                new_score = round(float(r["score"]) + adj_map.get(r["id"], 0.0), 6)
                new_rows.append({**r, "score": new_score, "delta": adj_map.get(r["id"], 0.0)})
            new_rows.sort(key=lambda x: x["score"], reverse=True)
            return new_rows, endor, align_map
        return rows, endor, align_map
    except Exception:
        return rows, None, {}


def judge_aggregate(stage, sims, alias_out, type_out, attr_out, neigh_out, attack_out, bucket, round_idx, inputs):
    w = CFG["w"]

    def to_map(lst):
        return {x["candidate_id"]: x for x in lst}

    alias_m = to_map(alias_out)
    type_m = to_map(type_out)
    attr_m = to_map(attr_out)
    neigh_m = to_map(neigh_out)
    attack_m = to_map(attack_out)
    cand_ids = list(sims.keys())
    rows = []
    for cid in cand_ids:
        emb = sims.get(cid, 0.0)
        alias = alias_m.get(cid, {"score": 0.5})["score"]
        attr = attr_m.get(cid, {"score": 0.5})["score"]
        neigh = neigh_m.get(cid, {"score": 0.5})["score"]
        typ = type_m.get(cid, {"score": 0.5})["score"]
        pen = attack_m.get(cid, {"penalty": 0.0})["penalty"]
        if type_m.get(cid, {}).get("align") is False:
            typ = 0.0
        score = w["emb"] * emb + w["alias"] * alias + w["attr"] * attr + w["neigh"] * neigh + w["type"] * typ - pen
        rows.append({"id": cid, "score": round(float(score), 6),
                     "components": {"emb": emb, "alias": alias, "attr": attr, "neigh": neigh, "type": typ,
                                    "penalty": pen}})
    rows.sort(key=lambda x: x["score"], reverse=True)

    rows, endorse_id, align_map = _apply_llm_judge_adjustments(stage, sims, alias_out, type_out, attr_out, neigh_out,
                                                               attack_out,
                                                               rows, bucket, round_idx, inputs)

    top1 = rows[0] if rows else {"id": "", "score": 0.0}
    top2 = rows[1] if len(rows) > 1 else {"id": "", "score": 0.0}
    margin = top1["score"] - top2["score"]
    main_outs = {"alias": alias_out, "type": type_out, "attr": attr_out, "neigh": neigh_out}
    agree_cnt, total_cnt = _align_votes(main_outs, top1["id"])
    agree_ratio = agree_cnt / max(1, total_cnt)
    risk = attack_m.get(top1["id"], {"penalty": 0.0})["penalty"]
    scores4 = []
    for k in ["alias", "type", "attr", "neigh"]:
        v = main_outs[k]
        m = next((x for x in v if x["candidate_id"] == top1["id"]), None)
        scores4.append(m["score"] if m else 0.5)
    mu = sum(scores4) / len(scores4)
    var = sum((x - mu) * (x - mu) for x in scores4) / len(scores4)
    alias_strong = 1.0 if alias_m.get(top1["id"], {"score": 0.0})["score"] >= 0.85 else (
        0.5 if alias_m.get(top1["id"], {"score": 0.0})["score"] >= 0.6 else 0.0)
    has_attr_signal = 1.0 if attr_m.get(top1["id"], {"score": 0.5})["score"] >= 0.6 else 0.0
    has_nei_overlap = 1.0 if neigh_m.get(top1["id"], {"score": 0.0})["score"] >= 0.6 else (
        0.5 if neigh_m.get(top1["id"], {"score": 0.0})["score"] >= 0.5 else 0.0)
    dens = (alias_strong + has_attr_signal + has_nei_overlap) / 3.0
    signal = {"margin": round(margin, 6), "agree": round(agree_ratio, 6), "risk": round(risk, 6), "var": round(var, 6),
              "dens": round(dens, 6)}
    if endorse_id:
        signal["endorse"] = endorse_id

    judge_align_top1 = align_map.get(top1["id"], "abstain")

    if stage == "R1":
        judge_endorses_top1 = (judge_align_top1 is True)
        if judge_endorses_top1 and (agree_ratio > 0.5 or margin > 0.05):
            return {"status": "final", "final_candidate": top1["id"], "scores_per_candidate": rows, "signal": signal,
                    "suggestion": "R1 early stop"}
        else:
            return {"status": "continue", "scores_per_candidate": rows, "signal": signal,
                    "suggestion": "Continue to R2, check weak dimensions"}
    elif stage == "R2":
        judge_endorses_top1 = (judge_align_top1 is True)

        if judge_endorses_top1 and (agree_ratio > 0.5 or margin > 0.05):
            return {
                "status": "final",
                "final_candidate": top1["id"],
                "scores_per_candidate": rows,
                "signal": signal,
                "suggestion": "R2 converge",
            }

        judge_rejects_top1 = (judge_align_top1 is False)
        if judge_rejects_top1 and agree_ratio <= 0.5 and top1["score"] < 0.5:
            return {
                "status": "expand",
                "scores_per_candidate": rows,
                "signal": signal,
                "suggestion": "R2 expand",
                "expanded_to": f"Top{CFG['expand_default']}",
            }

        return {
            "status": "continue",
            "scores_per_candidate": rows,
            "signal": signal,
            "suggestion": "R3 final judgment",
        }
    else:
        return {"status": "final", "final_candidate": top1["id"], "scores_per_candidate": rows, "signal": signal,
                "suggestion": "R3 converge/force single selection"}

def _classify_error(ref, cand_ids, init_top1, final_cand, debated):
    if ref is None:
        return None
    if final_cand == ref:
        return None
    if ref not in cand_ids or debated == False and final_cand != ref:
        return "embedding error"
    if init_top1 == ref and final_cand != ref:
        return "embedding correct but debate error"
    return "debate error"


def run_debate_for_source(source_id, cand_entry, name_dict, neighbors, attributes):
    try:
        src_id = str(source_id)
        src_name = id2name(src_id, name_dict)
        ref = str(cand_entry.get("ref")) if cand_entry.get("ref") is not None else None
        cand_ids = [str(x) for x in cand_entry.get("candidates", [])]
        sims = cand_entry.get("cand_sims", [])
        sim_map_full = {cid: float(sims[i]) if i < len(sims) else 0.0 for i, cid in enumerate(cand_ids)}
        init_top1 = cand_ids[0] if cand_ids else None

        margin = 0.0
        if len(sims) >= 2:
            margin = sims[0] - sims[1] if len(sims) >= 2 else 0.0

        if margin > 0.05:
            final_cand = init_top1
            is_correct = (ref is not None and final_cand == ref) if ref is not None else None
            err_type = _classify_error(ref, cand_ids, init_top1, final_cand, False)
            top5 = sorted(
                [{"id": cid, "score": sim_map_full.get(cid, 0.0)} for cid in cand_ids],
                key=lambda x: x["score"], reverse=True
            )[:5]

            return {
                "source_id": src_id,
                "ref": ref,
                "basic_debated": False,
                "debated": False,
                "bucket": "None",
                "final_candidate": final_cand,
                "confidence": 1.0,
                "top3": [{"candidate_id": final_cand, "score": sims[0]}],
                "flags": {"expanded": False, "low_confidence": False, "early_stop": True},
                "rounds": [],
                "metrics_local": {"margin": margin, "agree": 0.0, "risk": 0.0, "var": 0.0, "dens": 0.0},
                "is_correct_align": is_correct,
                "error_type": err_type,
                "top5_embedding_order": top5,
                "basic_debate_rounds": [],
                "basic_final_candidates": [],
                "basic_final_candidate": final_cand
            }
        elif margin > 0.025:
            src_pack = {
                "id": src_id,
                "name": src_name,
                "attributes": attributes.get(src_id, {}),
                "neighbors": build_neighbor_names(src_id, neighbors, name_dict, include_rel=True),
            }
            topk_ids = cand_ids[:20]
            cand_names = {cid: id2name(cid, name_dict) for cid in topk_ids}
            cand_attrs = {cid: attributes.get(cid, {}) for cid in topk_ids}
            cand_neighbors = {cid: build_neighbor_names(cid, neighbors, name_dict, include_rel=True) for cid in
                              topk_ids}
            cand_packs = {cid: {"id": cid, "name": cand_names[cid], "attributes": cand_attrs[cid],
                                "neighbors": cand_neighbors[cid]} for cid in topk_ids}

            basic_debate_result = run_basic_debate_compressed(src_pack, cand_packs, 1)
            final_candidates = basic_debate_result["final_candidates"]
            basic_debate_rounds = basic_debate_result["rounds_output"]

            basic_top1 = final_candidates[0][0] if final_candidates else None
            basic_final_candidates = [{"id": cid, "score": score} for cid, score in final_candidates]
            basic_final_candidate = basic_top1

            if basic_top1 == init_top1:
                final_cand = init_top1
                is_correct = (ref is not None and final_cand == ref) if ref is not None else None
                err_type = _classify_error(ref, cand_ids, init_top1, final_cand, False)
                top5 = sorted(
                    [{"id": cid, "score": sim_map_full.get(cid, 0.0)} for cid in cand_ids],
                    key=lambda x: x["score"], reverse=True
                )[:5]

                return {
                    "source_id": src_id,
                    "ref": ref,
                    "basic_debated": True,
                    "debated": False,
                    "bucket": "None",
                    "final_candidate": final_cand,
                    "confidence": 1.0,
                    "top3": [{"candidate_id": final_cand, "score": sims[0]}],
                    "flags": {"expanded": False, "low_confidence": False, "early_stop": True},
                    "rounds": [],
                    "metrics_local": {"margin": margin, "agree": 0.0, "risk": 0.0, "var": 0.0, "dens": 0.0},
                    "is_correct_align": is_correct,
                    "error_type": err_type,
                    "top5_embedding_order": top5,
                    "basic_debate_rounds": basic_debate_rounds,
                    "basic_final_candidates": basic_final_candidates,
                    "basic_final_candidate": basic_final_candidate
                }
            else:
                cand_ids = [cid for cid, score in final_candidates]
        else:
            basic_debate_rounds = []
            basic_final_candidates = []
            basic_final_candidate = init_top1
            src_pack = {
                "id": src_id,
                "name": src_name,
                "attributes": attributes.get(src_id, {}),
                "neighbors": build_neighbor_names(src_id, neighbors, name_dict, include_rel=True),
            }
            topk_ids = cand_ids[:20]
            cand_names = {cid: id2name(cid, name_dict) for cid in topk_ids}
            cand_attrs = {cid: attributes.get(cid, {}) for cid in topk_ids}
            cand_neighbors = {cid: build_neighbor_names(cid, neighbors, name_dict, include_rel=True) for cid in
                              topk_ids}
            cand_packs = {cid: {"id": cid, "name": cand_names[cid], "attributes": cand_attrs[cid],
                                "neighbors": cand_neighbors[cid]} for cid in topk_ids}

        rounds = []
        expanded_once = False
        final_obj = None
        last_rows = []
        for b in [5, 10, 15, 20]:
            cur_bucket = b
            topk_ids = cand_ids[:b]
            sims_k = {cid: sim_map_full.get(cid, 0.0) for cid in topk_ids}
            cand_names = {cid: id2name(cid, name_dict) for cid in topk_ids}
            cand_attrs = {cid: attributes.get(cid, {}) for cid in topk_ids}
            cand_neighbors = {cid: build_neighbor_names(cid, neighbors, name_dict, include_rel=True) for cid in
                              topk_ids}
            cand_packs = {cid: {"id": cid, "name": cand_names[cid], "attributes": cand_attrs[cid],
                                "neighbors": cand_neighbors[cid]} for cid in topk_ids}

            alias_out = alias_agent(src_pack["name"], cand_names)
            type_out = type_agent(src_pack, cand_packs)
            attr_out = attr_agent(src_pack["attributes"], cand_attrs)
            neigh_out = neigh_agent(src_pack["neighbors"], cand_neighbors)
            main_outputs_R1 = {"alias": alias_out, "type": type_out, "attr": attr_out, "neigh": neigh_out}

            r1_attack_inputs = {
                "src_name": src_pack["name"],
                "src_attr": src_pack["attributes"],
                "src_neighbor": src_pack["neighbors"],
                "cand_ids": topk_ids,
                "cand_names": cand_names,
                "cand_attrs": cand_attrs,
                "cand_neighbors": cand_neighbors,
                "sims": sims_k
            }
            judge_attack_inputs = {
                "src_name": src_pack["name"],
                "src_attr": src_pack["attributes"],
                "src_neighbor": src_pack["neighbors"],
                "cand_ids": topk_ids,
                "cand_names": cand_names,
                "cand_attrs": cand_attrs,
                "cand_neighbors": cand_neighbors
            }
            attack_out = attack_agent(main_outputs=None, stage="R1", inputs=r1_attack_inputs, history=None)

            judge_R1 = judge_aggregate("R1", sims_k, alias_out, type_out, attr_out, neigh_out, attack_out, b, 1,
                                       judge_attack_inputs)
            if judge_R1.get("scores_per_candidate"):
                last_rows = judge_R1.get("scores_per_candidate")

            rounds.append({
                "round": 1, "bucket": f"Top{b}",
                "alias": alias_out, "type": type_out, "attr": attr_out, "neigh": neigh_out,
                "attack": attack_out, "judge": judge_R1
            })
            if judge_R1["status"] == "final":
                final_obj = judge_R1
                break

            alias_out2 = alias_agent(src_pack["name"], cand_names, history=main_outputs_R1,
                                     judge_suggestion=judge_R1.get("suggestion"))
            type_out2 = type_agent(src_pack, cand_packs, history=main_outputs_R1,
                                   judge_suggestion=judge_R1.get("suggestion"))
            attr_out2 = attr_agent(src_pack["attributes"], cand_attrs, history=main_outputs_R1,
                                   judge_suggestion=judge_R1.get("suggestion"))
            neigh_out2 = neigh_agent(src_pack["neighbors"], cand_neighbors, history=main_outputs_R1,
                                     judge_suggestion=judge_R1.get("suggestion"))
            main_outputs_R2 = {"alias": alias_out2, "type": type_out2, "attr": attr_out2, "neigh": neigh_out2}

            attack_out2 = attack_agent(main_outputs=main_outputs_R2, stage="R2", inputs=judge_attack_inputs,
                                       history=attack_out)
            judge_R2 = judge_aggregate("R2", sims_k, alias_out2, type_out2, attr_out2, neigh_out2, attack_out2, b, 2,
                                       judge_attack_inputs)
            if judge_R2.get("scores_per_candidate"):
                last_rows = judge_R2.get("scores_per_candidate")

            rounds.append({
                "round": 2, "bucket": f"Top{b}",
                "alias": alias_out2, "type": type_out2, "attr": attr_out2, "neigh": neigh_out2,
                "attack": attack_out2, "judge": judge_R2
            })
            if judge_R2["status"] == "final":
                final_obj = judge_R2
                break

            if (judge_R2["status"] == "expand") and (not expanded_once):
                expanded_once = True

            if judge_R2["status"] == "continue":
                alias_out3 = alias_agent(src_pack["name"], cand_names, history=main_outputs_R2,
                                         judge_suggestion=judge_R2.get("suggestion"))
                type_out3 = type_agent(src_pack, cand_packs, history=main_outputs_R2,
                                       judge_suggestion=judge_R2.get("suggestion"))
                attr_out3 = attr_agent(src_pack["attributes"], cand_attrs, history=main_outputs_R2,
                                       judge_suggestion=judge_R2.get("suggestion"))
                neigh_out3 = neigh_agent(src_pack["neighbors"], cand_neighbors, history=main_outputs_R2,
                                         judge_suggestion=judge_R2.get("suggestion"))
                main_outputs_R3 = {"alias": alias_out3, "type": type_out3, "attr": attr_out3, "neigh": neigh_out3}

                attack_out3 = attack_agent(main_outputs=main_outputs_R3, stage="R3", inputs=judge_attack_inputs,
                                           history=attack_out2)
                judge_R3 = judge_aggregate("R3", sims_k, alias_out3, type_out3, attr_out3, neigh_out3, attack_out3, b,
                                           3, judge_attack_inputs)
                if judge_R3.get("scores_per_candidate"):
                    last_rows = judge_R3.get("scores_per_candidate")

                rounds.append({
                    "round": 3, "bucket": f"Top{b}",
                    "alias": alias_out3, "type": type_out3, "attr": attr_out3, "neigh": neigh_out3,
                    "attack": attack_out3, "judge": judge_R3
                })
                final_obj = judge_R3
                break

        if final_obj is None:
            if last_rows:
                chosen_top1 = last_rows[0]["id"]
                final_obj = {"final_candidate": chosen_top1, "scores_per_candidate": last_rows, "signal": {},
                             "suggestion": "fallback_from_last_rows"}
            else:
                if margin > 0.05:
                    final_cand = basic_final_candidate if basic_final_candidate else ""
                else:
                    final_cand = init_top1 if init_top1 else ""
                final_obj = {"final_candidate": final_cand,
                             "scores_per_candidate": [{"id": final_cand, "score": sim_map_full.get(final_cand, 0.0)}],
                             "signal": {}, "suggestion": "fallback_basic_debate_only"}

        final_rows = final_obj.get("scores_per_candidate", [])
        top3 = [{"candidate_id": r["id"], "score": r["score"]} for r in final_rows[:3]]
        sig = final_obj.get("signal", {})
        confidence = float(max(0.0, min(1.0, 0.5 + 0.5 * sig.get("margin", 0.0))))

        final_cand = final_obj.get("final_candidate")
        is_correct = (ref is not None and final_cand == ref) if ref is not None else None
        err_type = _classify_error(ref, cand_ids, init_top1, final_cand, True)

        return {
            "source_id": src_id,
            "ref": ref,
            "basic_debated": (margin > 0.05),
            "debated": True,
            "bucket": f"Top{cur_bucket}",
            "final_candidate": final_cand,
            "confidence": round(confidence, 4),
            "top3": top3,
            "flags": {"expanded": expanded_once, "low_confidence": confidence < 0.6},
            "rounds": rounds,
            "metrics_local": {
                "margin": sig.get("margin", 0.0),
                "agree": sig.get("agree", 0.0),
                "risk": sig.get("risk", 0.0),
                "var": sig.get("var", 0.0),
                "dens": sig.get("dens", 0.0)
            },
            "is_correct_align": is_correct,
            "error_type": err_type,
            "final_scores_per_candidate": final_rows,
            "basic_debate_rounds": basic_debate_rounds,
            "basic_final_candidates": basic_final_candidates,
            "basic_final_candidate": basic_final_candidate
        }
    except Exception as e:
        return {"source_id": str(source_id), "error": str(e), "trace": traceback.format_exc()}

def init_metrics(total, completed_ids, restore_metrics=None):
    if restore_metrics:
        metrics = {
            "num_total": total,
            "mrr": restore_metrics.get("mrr", 0.0),
            "hits@1": restore_metrics.get("hits@1", 0.0),
            "hits@5": restore_metrics.get("hits@5", 0.0),
            "hits@10": restore_metrics.get("hits@10", 0.0),
            "hits@20": restore_metrics.get("hits@20", 0.0),
            "num_debated": restore_metrics.get("num_debated", 0),
            "nfix": restore_metrics.get("num_embed_wrong_but_debate_correct", 0),
            "processed": len(completed_ids),

            "embed_wrong_but_debate_correct_ids": restore_metrics.get("embed_wrong_but_debate_correct_ids", []),
        }
        return metrics

    return {
        "num_total": total,
        "mrr": 0.0,
        "hits@1": 0.0,
        "hits@5": 0.0,
        "hits@10": 0.0,
        "hits@20": 0.0,
        "num_debated": 0,
        "nfix": 0,
        "processed": len(completed_ids),
        "embed_wrong_but_debate_correct_ids": [],
    }


def update_metrics(metrics, source_id, cand_entry, result_obj):
    metrics["processed"] += 1
    ref = str(cand_entry.get("ref")) if cand_entry.get("ref") is not None else None
    cand_ids = [str(x) for x in cand_entry.get("candidates", [])]
    sims = cand_entry.get("cand_sims", [])
    final_cand = str(result_obj.get("final_candidate"))
    debated = False

    if result_obj.get("debated"):
        debated = True
        metrics["num_debated"] += 1

    init_top1 = cand_ids[0] if cand_ids else None

    final_rows = result_obj.get("final_scores_per_candidate", [])
    if not final_rows:
        final_rows = sorted(
            [{"id": cid, "score": float(sims[i]) if i < len(sims) else 0.0} for i, cid in enumerate(cand_ids)],
            key=lambda x: x["score"], reverse=True
        )

    order = [r["id"] for r in final_rows] if final_rows else ([final_cand] if final_cand else [])

    embed_top20 = cand_ids[:20]
    order_extended = order[:20]
    order_set = set(order_extended)
    for cand in embed_top20:
        if cand not in order_set:
            order_extended.append(cand)
            order_set.add(cand)
        if len(order_extended) >= 20:
            break

    if ref is not None and final_cand == ref:
        metrics["hits@1"] += 1.0

    if ref in order_extended[:5]:
        metrics["hits@5"] += 1.0
    if ref in order_extended[:10]:
        metrics["hits@10"] += 1.0
    if ref in order_extended[:20]:
        metrics["hits@20"] += 1.0

    if ref in order_extended:
        rank = order_extended.index(ref) + 1
        metrics["mrr"] += 1.0 / rank

    if ref is not None and init_top1 is not None:
        if init_top1 != ref and final_cand == ref:
            metrics["nfix"] += 1
            metrics["embed_wrong_but_debate_correct_ids"].append(str(source_id))


def finalize_metrics(metrics):
    tot = max(1, metrics["processed"])
    return {
        "num_total": metrics["num_total"],
        "processed": metrics["processed"],
        "MRR": round(metrics["mrr"] / tot, 6),
        "Hits@1": round(metrics["hits@1"] / tot, 6),
        "Hits@5": round(metrics["hits@5"] / tot, 6),
        "Hits@10": round(metrics["hits@10"] / tot, 6),
        "Hits@20": round(metrics["hits@20"] / tot, 6),
        "num_debated": metrics["num_debated"],
        "num_embed_wrong_but_debate_correct": metrics["nfix"],
        "embed_wrong_but_debate_correct_ids": metrics["embed_wrong_but_debate_correct_ids"]
    }


def main():
    start_time = time.time()

    ensure_dirs()
    cand = load_json(os.path.join(INPUT_DIR, "cand"))
    name_dict = load_json(os.path.join(INPUT_DIR, "name_dict"))
    neighbors = load_json(os.path.join(INPUT_DIR, "neighbors"))
    attributes = load_json(os.path.join(INPUT_DIR, "attributes"))

    initialize_key_elements(attributes, neighbors)

    all_source_ids = list(cand.keys())
    completed_ids = set()
    left_ids = all_source_ids

    metrics = init_metrics(len(all_source_ids), set())

    pbar = tqdm(left_ids, desc="Debating")
    for sid in pbar:
        entry = cand[sid]
        try:
            result_obj = run_debate_for_source(sid, entry, name_dict, neighbors, attributes)
        except Exception as e:
            result_obj = {"source_id": sid, "error": str(e), "trace": traceback.format_exc()}

        update_metrics(metrics, sid, entry, result_obj)
        completed_ids.add(sid)

    summary = finalize_metrics(metrics)

    if hasattr(llm_call, "token_usage"):
        token_usage = llm_call.token_usage

        if isinstance(token_usage, dict):
            total_tokens = token_usage.get("total_tokens", 0)
        else:
            try:
                total_tokens = token_usage.total_tokens if hasattr(token_usage, "total_tokens") else 0
            except:
                total_tokens = 0
    else:
        total_tokens = 0

    summary["token_usage"] = {"total_tokens": total_tokens}

    end_time = time.time()
    elapsed_time = end_time - start_time
    summary["total_runtime_seconds"] = elapsed_time

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()