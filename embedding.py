"""
The main directory should contain: name, att, rel
Running this script will produce one output file in the current directory: fused_embeddings.pkl
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import torch
from numpy.linalg import norm
from llm2vec import LLM2Vec
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

Model_Path = os.getenv("Embedding_Model_Path", "")


def max_pooling_fusion(*features):
    return np.max(np.stack(features), axis=0)


def weight_fusion(*features):
    return np.average(np.stack(features), axis=0, weights=[1, 1, 1])


def concatenation_fusion(*features):
    return np.concatenate(features)


def mean_fusion(*features):
    return np.mean(np.stack(features), axis=0)


def load_data(directory, pkl_files):
    data_dict = {}
    for pkl_file in pkl_files:
        pkl_path = os.path.join(directory, pkl_file)
        with open(pkl_path, 'rb') as f:
            data_dict.update(pickle.load(f))
    return data_dict


def generate_fused_embeddings(directory, fusion_method, selected_dicts):
    loaded_data = {
        key: load_data(directory, [f'{key}_dict_left.pkl', f'{key}_dict_right.pkl'])
        for key in ['att', 'name', 'rel']
    }

    fused_embeddings = {}
    for key in loaded_data['att'].keys():
        features = [loaded_data[dict_key].get(key, []) for dict_key in selected_dicts]

        features = [
            f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f
            for f in features
        ]

        features = [f for f in features if np.size(f) > 0]
        if features:
            fused_embeddings[key] = fusion_method(*features)

    return fused_embeddings


def calculate_cosine_similarity(embed_left, embed_right):
    norm_left = norm(embed_left, axis=1, keepdims=True)
    norm_right = norm(embed_right, axis=1, keepdims=True)
    return np.dot(embed_left, embed_right.T) / (norm_left * norm_right.T)


def evaluate_predictions(sim_matrix, left_keys, right_keys, ground_truth, k_values):
    reciprocal_ranks = []
    hits = {k: 0 for k in k_values}
    ground_truth_dict = {left: right for left, right in ground_truth}

    candidates = {
        left_keys[i]: sorted(
            [(right_keys[j], sim_matrix[i, j]) for j in range(len(right_keys))],
            key=lambda x: x[1], reverse=True
        )
        for i in range(len(left_keys))
    }

    correct_predictions_at_1 = 0
    total_predictions = len(ground_truth)
    incorrect_hits1_candidates = {}

    for left_id, preds in candidates.items():
        if left_id not in ground_truth_dict:
            continue
        true_right_id = ground_truth_dict[left_id]

        rank = 0
        for i, (right_id, _) in enumerate(preds):
            if right_id == true_right_id:
                rank = i + 1
                if rank == 1:
                    correct_predictions_at_1 += 1
                break

        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        else:
            reciprocal_ranks.append(0.0)
            incorrect_hits1_candidates[left_id] = preds[:10]

    mrr = sum(reciprocal_ranks) / total_predictions if total_predictions > 0 else 0.0
    hits_results = {f'Hits@{k}': hits[k] / total_predictions for k in k_values}

    return {
        'MRR': mrr,
        **hits_results,
        'candidates': {k: v[:10] for k, v in candidates.items()},
        'correct_predictions_at_1': correct_predictions_at_1,
        'total_predictions': total_predictions,
        'incorrect_hits1_candidates': incorrect_hits1_candidates
    }


def load_alignments_from_txt(file_path):
    with open(file_path, 'r') as f:
        alignments = [line.strip().split() for line in f]
    return alignments

def generate_embeddings(input_file, feature_type):
    base_model_path = Model_Path

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    tokenizer.padding_side = "left"
    print(f"Set padding_side to: {tokenizer.padding_side}")

    config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True
    )

    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    att_values_left = {}
    att_values_right = {}

    print(f"\nProcessing {feature_type} features...")

    with open(input_file, 'r', encoding='utf-8') as txf:
        lines = txf.readlines()
        total_lines = len(lines)
        half_lines = total_lines // 2

        for i, line in enumerate(tqdm(lines[:half_lines], desc=f"Processing left entities ({feature_type})")):
            line = line.strip()
            if not line:
                continue

            if feature_type == 'name':
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ' ' in line:
                    parts = line.split(' ', 1)
                else:
                    continue

                if len(parts) == 2:
                    id, info = parts
                    att_values_left[id] = info
            else:
                id, info = line.strip().split('\t')
                info = eval(info)
                info = info[0]
                att_values_left[id] = info

        for i, line in enumerate(tqdm(lines[half_lines:], desc=f"Processing right entities ({feature_type})")):
            line = line.strip()
            if not line:
                continue

            if feature_type == 'name':
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ' ' in line:
                    parts = line.split(' ', 1)
                else:
                    continue

                if len(parts) == 2:
                    id, info = parts
                    att_values_right[id] = info
            else:
                id, info = line.strip().split('\t')
                info = eval(info)
                info = info[0]
                att_values_right[id] = info

    keys_left = list(att_values_left.keys())
    embed_left = l2v.encode(list(att_values_left.values()))
    value_left = np.array(embed_left)

    keys_right = list(att_values_right.keys())
    embed_right = l2v.encode(list(att_values_right.values()))
    value_right = np.array(embed_right)

    for i, key in enumerate(keys_left):
        att_values_left[key] = value_left[i]
    for i, key in enumerate(keys_right):
        att_values_right[key] = value_right[i]

    with open(f'{feature_type}_dict_left.pkl', 'wb') as f:
        pickle.dump(att_values_left, f)
    with open(f'{feature_type}_dict_right.pkl', 'wb') as f:
        pickle.dump(att_values_right, f)

    print(f"{feature_type} feature embeddings saved to {feature_type}_dict_left.pkl and {feature_type}_dict_right.pkl")


def main():
    print("Starting entity embedding processing...")

    # 1. Generate embeddings for name features
    generate_embeddings('name', 'name')

    # 2. Generate embeddings for att features
    generate_embeddings('att', 'att')

    # 3. Generate embeddings for rel features
    generate_embeddings('rel', 'rel')

    print("\nAll feature embeddings completed, starting feature fusion...")

    # 4. Feature fusion
    directory = ""
    fusion_method = concatenation_fusion
    selected_dicts = ['att', 'rel', 'name']
    embedding_dict = generate_fused_embeddings(directory, fusion_method, selected_dicts)

    # Save fused embeddings
    with open('fused_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)
    print("Fused embeddings saved to fused_embeddings.pkl")

    # 5. Load test data and evaluate
    data_path = 'test'
    dev_alignments = load_alignments_from_txt(data_path)

    left_ids = [itm[0] for itm in dev_alignments if itm[0] in embedding_dict]
    right_ids = [itm[1] for itm in dev_alignments if itm[1] in embedding_dict]
    known_pairs = [(left, right) for left, right in dev_alignments if
                   left in embedding_dict and right in embedding_dict]

    embed_left = np.array([embedding_dict[left_id] for left_id in left_ids])
    embed_right = np.array([embedding_dict[right_id] for right_id in right_ids])

    cos_sim = calculate_cosine_similarity(embed_left, embed_right)

    # 6. Evaluate results
    k_values = [1, 5, 10, 20]
    evaluation_results = evaluate_predictions(cos_sim, left_ids, right_ids, known_pairs, k_values)

    print(f"\n========== Evaluation Results ==========")
    print(f"Correct Predictions: {evaluation_results['correct_predictions_at_1']}")
    print(f"Total Predictions: {evaluation_results['total_predictions']}")
    print(f"MRR: {evaluation_results['MRR']:.4f}")
    for k in k_values:
        print(f"Hits@{k}: {evaluation_results[f'Hits@{k}']:.4f}")



if __name__ == "__main__":
    main()