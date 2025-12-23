"""
The main directory should contain: name.txt, att.txt, rel.txt
Running this script will produce three output files in the current directory: name, att, and rel.
"""

from tqdm import tqdm
import json
from openai import OpenAI
import os

API_KEY = ""
BASE_URL = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def save_result(filename, key, result):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps({key: result}, ensure_ascii=False) + '\n')


def save_all_results(filename, results_dict):
    with open(filename, 'w', encoding='utf-8') as f:
        for key, result in results_dict.items():
            f.write(json.dumps({key: result}, ensure_ascii=False) + '\n')


def jsonl_to_txt(jsonl_file_path, txt_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
            open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        for line in jsonl_file:
            try:
                data = json.loads(line)
                for key, value in data.items():
                    txt_file.write(f'{key}\t{value}\n')
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")


def translate_entity_names():
    print("=" * 50)
    print("Step 1: Starting entity name translation...")

    with open('name.txt', 'r', encoding='utf-8') as f:
        atts = f.readlines()

    att = {}
    for line in atts:
        entity_id, info = line.strip().split('\t')
        att[entity_id] = info

    prompt = """
    Translate the following Chinese entity names into English.
    You must remember that you can only give me the English entity name and cannot return any additional information.
    """

    att_ans = {}
    att_keys = list(att.keys())
    data_len = len(att_keys) // 2
    att_keys_chinese = att_keys[:data_len]
    att_keys_english = att_keys[data_len:]

    print(f"Need to translate {len(att_keys_chinese)} Chinese entity names...")

    for key in tqdm(att_keys_chinese, desc="Translating Chinese entity names"):
        info = att[key]
        user_prompt = f"\n{info}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                top_p=0.9,
                n=1
            )
            best_match_id = response.choices[0].message.content.strip()
            att_ans[key] = best_match_id
        except Exception as e:
            print(f"Error translating entity {key}: {e}")
            best_match_id = "Translation error"
            att_ans[key] = best_match_id

    for key in att_keys_english:
        att_ans[key] = att[key]

    try:
        with open("name_trans.jsonl", "w", encoding='utf-8') as f:
            for key, result in att_ans.items():
                f.write(json.dumps({key: result}, ensure_ascii=False) + "\n")
        print(f"Entity name translation completed! Total processed: {len(att_ans)} entities")
    except Exception as e:
        print(f"Error saving translation results: {e}")

    return att_ans


def process_attribute_entities():
    print("=" * 50)
    print("Step 2: Starting attribute entity processing...")

    with open('att.txt', 'r', encoding='utf-8') as f:
        atts = f.readlines()

    att = {}
    for line in atts:
        entity_id, info = line.strip().split('\t')
        att[entity_id] = eval(info)

    with open('rel.txt', 'r', encoding='utf-8') as f:
        rels = f.readlines()

    rel = {}
    for line in rels:
        entity_id, info = line.strip().split('\t')
        rel[entity_id] = eval(info)

    prompt = """
    You are an expert who can provide concise explanations based on entity information. I will give you the properties of an entity in the form of a triple (subject, predicate, object). Using this information along with your general knowledge, please provide a short description of the entity.
    - The explanation should be no longer than 100 words.
    - Focus on summarizing the entity based on the given information and your general knowledge.
    - Do not include unnecessary details or explanations beyond the entity description.
    - Every word in your answer must be English.
    Example:
    Entity Information: (Albert Einstein, profession, Physicist), (Albert Einstein, known for, Theory of Relativity)
    Explanation: Albert Einstein was a renowned physicist best known for developing the Theory of Relativity, a fundamental theory in modern physics.
    Now, please summarize the following entity information and return an desctription in English:
    """

    entity_keys = list(att.keys())
    print(f"Attribute entities: Total entities {len(entity_keys)}")

    results = {}

    for key in tqdm(entity_keys, desc="Processing attribute entities"):
        info = att[key]
        user_prompt = f"\n{info}\n"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=256,
                top_p=0.9,
                n=1
            )

            best_match_id = response.choices[0].message.content.strip()
            if best_match_id:
                result = [best_match_id]
                results[key] = result
            else:
                print(f"Warning: Returned empty content, entity: {key}")
                result = [""]
                results[key] = result

        except Exception as e:
            print(f"API call failed, entity: {key}, error: {e}")
            best_match_id = ""
            result = [best_match_id]
            results[key] = result

    save_all_results("att_summary.jsonl", results)

    print(f"Attribute entity processing completed! Processed {len(results)} entities")
    return results


def process_relation_entities():
    print("=" * 50)
    print("Step 3: Starting relation entity processing...")

    with open('rel.txt', 'r', encoding='utf-8') as f:
        rels = f.readlines()

    rel = {}
    for line in rels:
        entity_id, info = line.strip().split('\t')
        rel[entity_id] = eval(info)

    prompt = """
    You are an expert who can provide concise explanations based on entity information. I will give you the properties of an entity in the form of a triple (subject, predicate, object). Using this information along with your general knowledge, please provide a short description of the entity.
    - The explanation should be no longer than 100 words.
    - Focus on summarizing the entity based on the given information and your general knowledge.
    - Do not include unnecessary details or explanations beyond the entity description.
    - Every word in your answer must be English.
    Example:
    Entity Information: (Albert Einstein, profession, Physicist), (Albert Einstein, known for, Theory of Relativity)
    Explanation: Albert Einstein was a renowned physicist best known for developing the Theory of Relativity, a fundamental theory in modern physics.
    Now, please summarize the following entity information and return an desctription in English:
    """

    if os.path.exists("rel_summary.jsonl"):
        os.remove("rel_summary.jsonl")

    results = {}

    for key in tqdm(rel.keys(), desc="Processing relation entities"):
        info = rel[key]
        user_prompt = f"\n{info}\n"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=256,
                top_p=0.9,
                n=1
            )

            best_match_id = response.choices[0].message.content.strip()
            if best_match_id:
                result = [best_match_id]
                results[key] = result
                save_result("rel_summary.jsonl", key, result)
            else:
                print(f"Warning: Returned empty content, entity: {key}")
        except Exception as e:
            print(f"Error during API call for rel key {key}: {e}")
            best_match_id = ""
            result = [best_match_id]
            results[key] = result
            save_result("rel_summary.jsonl", key, result)

    print(f"Relation entity processing completed! Processed {len(results)} entities")
    return results


def convert_format():
    print("=" * 50)
    print("Step 4: Starting file format conversion...")

    print("Converting rel_summary.jsonl -> rel")
    jsonl_to_txt('rel_summary.jsonl', 'rel')

    print("Converting att_summary.jsonl -> att")
    jsonl_to_txt('att_summary.jsonl', 'att')

    print("Converting name_trans.jsonl -> name")
    jsonl_to_txt('name_trans.jsonl', 'name')

    print("File format conversion completed!")


def main():
    print("=" * 50)
    print("Starting preprocessing pipeline...")
    print("=" * 50)

    # Step 1: Translate entity names
    translate_entity_names()

    # Step 2: Process attribute entities
    process_attribute_entities()

    # Step 3: Process relation entities
    process_relation_entities()

    # Step 4: Convert format
    convert_format()

    print("=" * 50)
    print("All preprocessing steps completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()