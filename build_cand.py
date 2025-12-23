"""
The main directory should contain: fused_embeddings.pkl, test, att.txt, triples_1, triples_2
Running this script will produce four output files in the candidates directory: cand、name_dict、attributes、neighbors
"""

import os
import re
import json
import pickle
import argparse
import numpy as np
import ast
from urllib.parse import unquote
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Dict, Iterable, Set


def load_embeddings(path_override=None):
    candidates = []
    if path_override:
        candidates.append(path_override)
    candidates += ["fused_embeddings.pkl", "fused_embedding.pkl"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                emb = pickle.load(f)
            if not isinstance(emb, dict):
                raise ValueError(f"File {p} is not in dictionary format.")
            emb = {str(k): np.asarray(v, dtype=np.float32) for k, v in emb.items()}
            return emb, p
    raise FileNotFoundError("fused_embeddings.pkl or fused_embedding.pkl not found.")


def read_test_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"[ERROR] test file format error: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs


def read_id_list(path):
    if not path or not os.path.exists(path):
        return []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def generate_cand(args):
    print("\n" + "=" * 60)
    print("Step 1: Generate cand file")
    print("=" * 60)

    emb, emb_path = load_embeddings(args.fused_path)

    gold = OrderedDict()
    if args.left_ids and args.right_ids:
        left_ids = read_id_list(args.left_ids)
        right_ids = read_id_list(args.right_ids)
    else:
        if not os.path.exists(args.test_path):
            raise FileNotFoundError(f"test file not found: {args.test_path}")
        pairs = read_test_pairs(args.test_path)
        for l, r in pairs:
            if l not in gold:
                gold[l] = r
        left_ids = list(gold.keys())
        right_ids = list(set(gold.values()))

    left_ids_in = [i for i in left_ids if i in emb]
    right_ids_in = [i for i in right_ids if i in emb]

    if len(right_ids_in) < 2:
        raise ValueError("[ERROR] Too few right candidate entities (<2), please check test file or right_ids source.")

    L = np.stack([emb[i] for i in left_ids_in]).astype(np.float32)
    R = np.stack([emb[i] for i in right_ids_in]).astype(np.float32)
    L = l2_normalize(L)
    R = l2_normalize(R)

    sim = np.matmul(L, R.T)
    topk = min(args.topk, len(right_ids_in))
    idx_topk = np.argsort(-sim, axis=1)[:, :topk]
    top_sims = np.take_along_axis(sim, idx_topk, axis=1)

    cand_json = OrderedDict()
    for i, lid in enumerate(left_ids_in):
        cand_list = [right_ids_in[j] for j in idx_topk[i]]
        sim_list = [float(top_sims[i, j]) for j in range(topk)]
        ref = gold.get(lid, "")
        ground_rank = topk
        if ref and ref in right_ids_in:
            try:
                ground_rank = cand_list.index(ref)
            except ValueError:
                ground_rank = topk
        cand_json[lid] = {
            "ref": ref,
            "ground_rank": int(ground_rank),
            "candidates": cand_list,
            "cand_sims": [round(s, 6) for s in sim_list]
        }

    out_dir = args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cand")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cand_json, f, ensure_ascii=False, indent=2)

    recall = sum(1 for v in cand_json.values() if v["ground_rank"] < topk) / max(1, len(cand_json))
    print(f"[OK] cand file generated: {out_path}")

    return out_path


def read_id_map(path):
    m = {}
    if path is None:
        return m
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            if len(parts) < 2:
                _id = str(parts[0])
                m[_id] = _id
            else:
                _id, label = parts[0], parts[1]
                m[str(_id)] = label
    return m


def to_pretty_name(raw: str) -> str:
    if not raw:
        return raw
    s = raw.strip()

    if (" " in s) and not s.startswith("http"):
        return s

    seg = s
    if "#" in seg:
        seg = seg.rsplit("#", 1)[-1]
    if "/" in seg:
        seg = seg.rsplit("/", 1)[-1]

    seg = unquote(seg)

    seg = seg.replace("_", " ")

    seg = re.sub(r"\s+\(", " (", seg)
    seg = re.sub(r"\(\s+", "(", seg)
    seg = re.sub(r"\s+\)", ")", seg)

    return seg if seg else s


def load_cand_entities_for_name_dict(cand_path):
    if not cand_path or not os.path.exists(cand_path):
        return set()
    with open(cand_path, "r", encoding="utf-8") as f:
        cand = json.load(f)
    ents = set()
    for l, info in cand.items():
        ents.add(str(l))
        for r in info.get("candidates", []):
            ents.add(str(r))
        ref = info.get("ref", "")
        if ref:
            ents.add(str(ref))
    return ents


def load_neighbors_entities_and_rels_for_name_dict(neigh_path):
    if not neigh_path or not os.path.exists(neigh_path):
        return set(), set()
    with open(neigh_path, "r", encoding="utf-8") as f:
        neigh = json.load(f)
    ents, rels = set(), set()
    for ent, triples in neigh.items():
        ents.add(str(ent))
        for tri in triples:
            if len(tri) >= 3:
                h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
                ents.add(h);
                ents.add(t);
                rels.add(r)
    return ents, rels


def generate_name_dict(args, cand_path, neighbors_path=None):
    print("\n" + "=" * 60)
    print("Step 2: Generate name_dict file")
    print("=" * 60)

    ent1 = read_id_map(args.ent_ids_1)
    ent2 = read_id_map(args.ent_ids_2)
    rel1 = read_id_map(args.rel_ids_1)
    rel2 = read_id_map(args.rel_ids_2)

    print(f"[INFO] ent_ids_1: {len(ent1)}  ent_ids_2: {len(ent2)}")
    print(f"[INFO] rel_ids_1: {len(rel1)}  rel_ids_2: {len(rel2)}")

    ent_map = {**ent1, **ent2}
    rel_map = {**rel1, **rel2}

    if args.pretty:
        ent_map = {k: to_pretty_name(v) for k, v in ent_map.items()}
        rel_map = {k: to_pretty_name(v) for k, v in rel_map.items()}

    need_ents = set()
    need_rels = set()

    need_ents |= load_cand_entities_for_name_dict(cand_path)
    if neighbors_path and os.path.exists(neighbors_path):
        ents2, rels2 = load_neighbors_entities_and_rels_for_name_dict(neighbors_path)
        need_ents |= ents2
        need_rels |= rels2

    missing_ent = 0
    for e in need_ents:
        if e not in ent_map:
            ent_map[e] = e
            missing_ent += 1
    missing_rel = 0
    for r in need_rels:
        if r not in rel_map:
            rel_map[r] = r
            missing_rel += 1

    if need_ents or need_rels:
        print(f"[INFO] ensured coverage from cand/neighbors: "
              f"ents_needed={len(need_ents)} (+{missing_ent} filled), "
              f"rels_needed={len(need_rels)} (+{missing_rel} filled)")

    name_dict = {
        "ent": ent_map,
        "rel": rel_map,
        "time": {}
    }

    out_path = os.path.join(args.data_dir, "name_dict")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(name_dict, f, ensure_ascii=False, indent=2)

    print(f"[OK] name_dict generated: {out_path}")
    print(f"[STATS] ent={len(name_dict['ent'])}  rel={len(name_dict['rel'])}  time={len(name_dict['time'])}")
    if args.pretty:
        print("[HINT] Using --pretty, URIs converted to readable names. To keep original URIs, remove this parameter.")

    return out_path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_literal_eval(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def strip_entity_prefix(item: str, ent_label: str):
    if not ent_label:
        return None
    prefix = ent_label.strip() + " "
    if item.startswith(prefix):
        return item[len(prefix):]
    return None


def parse_att_line(line: str, name_map_ent: dict, target_ids: set):
    line = line.rstrip("\n")
    if not line:
        return None, []

    if "\t" not in line:
        return None, []

    ent_id, raw_list = line.split("\t", 1)
    ent_id = ent_id.strip()
    if ent_id not in target_ids:
        return None, []

    att_list = safe_literal_eval(raw_list)
    if not isinstance(att_list, list):
        return ent_id, []

    ent_label = name_map_ent.get(ent_id, "").strip()
    results = []

    for item in att_list:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue

        trailing = strip_entity_prefix(s, ent_label)
        if trailing is None:
            continue

        if " " not in trailing:
            continue

        key, value = trailing.split(" ", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        IGNORE_KEYS = {}
        if key.lower() in IGNORE_KEYS:
            continue

        results.append((key, value))

    return ent_id, results


def generate_attributes(args, cand_path, name_dict_path):
    print("\n" + "=" * 60)
    print("Step 4: Generate attributes file")
    print("=" * 60)

    cand_data = load_json(cand_path)

    target_ids = set()
    for src_id, info in cand_data.items():
        target_ids.add(str(src_id))
        for cid in info.get("candidates", []):
            target_ids.add(str(cid))

    name_dict = load_json(name_dict_path)
    name_map_ent = {str(k): v for k, v in name_dict.get("ent", {}).items()}

    attributes = defaultdict(lambda: OrderedDict())

    if not os.path.exists(args.att_path):
        print(f"✖ File not found: {args.att_path}")
        return None

    skipped_no_prefix = 0
    total_items = 0
    kept_items = 0

    with open(args.att_path, "r", encoding="utf-8") as f:
        for line in f:
            ent_id, pairs = parse_att_line(line, name_map_ent, target_ids)
            if ent_id is None or not pairs:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    _id = parts[0].strip()
                    if _id in target_ids:
                        skipped_no_prefix += 1
                continue

            kv_store = defaultdict(list)
            for k, v in pairs:
                total_items += 1
                kv_store[k].append(v)
                kept_items += 1

            clean_att = OrderedDict()
            for k, vlist in kv_store.items():
                seen = set()
                uniq = []
                for x in vlist:
                    if x not in seen:
                        seen.add(x)
                        uniq.append(x)
                clean_att[k] = uniq[0] if len(uniq) == 1 else uniq

            attributes[ent_id] = clean_att

    out_path = os.path.join(args.data_dir, "attributes")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(attributes, f, ensure_ascii=False, indent=2)

    covered = len(attributes)
    print(f"Attributes file generated: {out_path}")

    return out_path


def parse_line_to_triple(line: str, keep_time: bool) -> List[str]:
    parts = line.strip().split()
    if len(parts) == 0:
        return None
    if parts[0].startswith("#"):
        return None

    if len(parts) == 3:
        h, r, t = parts
        return [h, r, t]
    elif len(parts) == 4:
        h, r, t, time_one = parts
        return [h, r, t, time_one] if keep_time else [h, r, t]
    elif len(parts) == 5:
        h, r, t, tb, te = parts
        return [h, r, t, tb, te] if keep_time else [h, r, t]
    else:
        raise ValueError(f"Bad triple format with {len(parts)} columns: {line.strip()}")


def stream_triples(path: str, keep_time: bool) -> Iterable[List[str]]:
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            triple = parse_line_to_triple(line, keep_time)
            if triple is None:
                continue
            cnt += 1
            yield triple


def load_cand_entities_for_neighbors(cand_path: str) -> Tuple[set, set]:
    with open(cand_path, "r", encoding="utf-8") as f:
        cand = json.load(f)

    left_ids = set()
    right_ids = set()

    for left, info in cand.items():
        left_ids.add(str(left))
        for rid in info.get("candidates", []):
            right_ids.add(str(rid))
        ref = info.get("ref", "")
        if ref:
            right_ids.add(str(ref))

    return left_ids, right_ids


def add_triple_to_maps(triple: List[str],
                       map_left: Dict[str, set],
                       map_right: Dict[str, set],
                       left_set: set,
                       right_set: set,
                       include_all: bool):
    h, r, t = triple[0], triple[1], triple[2]
    if include_all or (h in left_set or t in left_set):
        map_left[h].add(tuple(triple))
        map_left[t].add(tuple(triple))
    if include_all or (h in right_set or t in right_set):
        map_right[h].add(tuple(triple))
        map_right[t].add(tuple(triple))


def clamp_neighbors(neigh_map: Dict[str, set], max_per_entity: int) -> Dict[str, list]:
    out = {}
    for ent, triples in neigh_map.items():
        lst = list(triples)
        if max_per_entity > 0 and len(lst) > max_per_entity:
            lst = lst[:max_per_entity]
        out[ent] = [list(x) for x in lst]
    return out


def generate_neighbors(args, cand_path):
    print("\n" + "=" * 60)
    print("Step 3: Generate neighbors file")
    print("=" * 60)

    left_need, right_need = load_cand_entities_for_neighbors(cand_path)

    neigh_left: Dict[str, set] = defaultdict(set)
    neigh_right: Dict[str, set] = defaultdict(set)

    cnt_left = 0
    for tri in stream_triples(args.triples_left, keep_time=args.keep_time):
        add_triple_to_maps(tri, neigh_left, neigh_right, left_need, right_need, args.include_all)
        cnt_left += 1

    cnt_right = 0
    for tri in stream_triples(args.triples_right, keep_time=args.keep_time):
        add_triple_to_maps(tri, neigh_left, neigh_right, left_need, right_need, args.include_all)
        cnt_right += 1

    out_map = defaultdict(list)

    neigh_left_final = clamp_neighbors(neigh_left, args.max_per_entity)
    neigh_right_final = clamp_neighbors(neigh_right, args.max_per_entity)

    for k, v in neigh_left_final.items():
        out_map[k].extend(v)
    for k, v in neigh_right_final.items():
        out_map[k].extend(v)

    for lid in left_need:
        _ = out_map[lid]
    for rid in right_need:
        _ = out_map[rid]

    for ent, triples in out_map.items():
        uniq = []
        seen = set()
        for x in triples:
            t = tuple(x)
            if t not in seen:
                seen.add(t)
                uniq.append(x)
        if args.max_per_entity > 0 and len(uniq) > args.max_per_entity:
            uniq = uniq[:args.max_per_entity]
        out_map[ent] = uniq

    cov_left = sum(1 for k in left_need if len(out_map[k]) > 0)
    cov_right = sum(1 for k in right_need if len(out_map[k]) > 0)

    out_path = os.path.join(args.data_dir, "neighbors")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] neighbors written: {out_path}")

    return out_path


def main():
    # Main function: execute all steps in order
    parser = argparse.ArgumentParser(description="Integrated script: generate all required files for AgentEA")

    # Step 1 parameters
    parser.add_argument("--fused_path", type=str, default=None, help="Path to fused_embeddings.pkl (optional)")
    parser.add_argument("--test_path", type=str, default="test", help="Path to test file (left right)")
    parser.add_argument("--left_ids", type=str, default=None, help="Left entity ID file (optional)")
    parser.add_argument("--right_ids", type=str, default=None, help="Right entity ID file (optional)")
    parser.add_argument("--topk", type=int, default=20, help="Number of candidates per source entity (default 20)")

    # Step 2 parameters
    parser.add_argument("--ent_ids_1", type=str, help="Left entity mapping: ent_ids_1", default="dbp15k/zh_en/ent_ids_1")
    parser.add_argument("--ent_ids_2", type=str, help="Right entity mapping: ent_ids_2", default="dbp15k/zh_en/ent_ids_2")
    parser.add_argument("--rel_ids_1", type=str, help="Left relation mapping: rel_ids_1", default="dbp15k/zh_en/rel_ids_1")
    parser.add_argument("--rel_ids_2", type=str, help="Right relation mapping: rel_ids_2", default="dbp15k/zh_en/rel_ids_2")
    parser.add_argument("--pretty", default=True, action="store_true", help="Convert URIs to human-readable names (enabled by default)")

    # Step 3 parameters
    parser.add_argument("--att_path", type=str, default="att.txt", help="Path to attribute file")

    # Step 4 parameters
    parser.add_argument("--triples_left", help="Left KG triple file path", default="triples_1")
    parser.add_argument("--triples_right", help="Right KG triple file path", default="triples_2")
    parser.add_argument("--include_all", action="store_true",
                        help="Generate neighbors for all entities in KG (not only cand-related entities). Disabled by default to reduce file size.")
    parser.add_argument("--keep_time", action="store_true",
                        help="Keep time columns (if triples have 4/5 columns). Default only outputs [h,r,t].")
    parser.add_argument("--max_per_entity", type=int, default=0,
                        help="Maximum neighbors per entity (0 means unlimited).")

    # General parameters
    parser.add_argument("--data_dir", type=str, default="candidates", help="Output data directory")

    args = parser.parse_args()

    # Step 1: Generate cand file
    cand_path = generate_cand(args)

    # Step 2: Generate neighbors file
    neighbors_path = generate_neighbors(args, cand_path)

    # Step 3: Generate name_dict file (use cand and neighbors to ensure coverage)
    name_dict_path = generate_name_dict(args, cand_path, neighbors_path)

    # Step 4: Generate attributes file
    attributes_path = generate_attributes(args, cand_path, name_dict_path)

    print("\n" + "=" * 60)
    print("All files generated successfully!")
    print("=" * 60)
    print(f"cand: {cand_path}")
    print(f"name_dict: {name_dict_path}")
    print(f"attributes: {attributes_path}")
    print(f"neighbors: {neighbors_path}")


if __name__ == "__main__":
    main()