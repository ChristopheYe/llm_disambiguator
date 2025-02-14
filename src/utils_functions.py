import ujson
import json
import sys
import ast
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import random
import faiss
import re
import openai
import time

from tqdm import tqdm
from typing import Optional

from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    add_deabbreviations,
    load_dataset_df,
    dataset_to_documents,
    dataset_to_df,
    load_dataset_df,
    resolve_abbreviation,
    dataset_unique_tax_ids,
)
from bioel.utils.solve_abbreviation.solve_abbreviations import create_abbrev
from bioel.ontology import BiomedicalOntology
from bioel.evaluate import Evaluate

from torch.utils.data import DataLoader

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_word_indices(doc, offsets):
    words = doc.split()
    char_count = 0
    start_char, end_char = offsets
    start_word_idx = None
    end_word_idx = None

    for idx, word in enumerate(words):
        word_start = char_count
        word_end = char_count + len(word)

        if word_start <= start_char < word_end:
            start_word_idx = idx
        if word_start < end_char <= word_end:
            end_word_idx = idx + 1

        char_count = word_end + 1  # +1 for the space

    # If the mention is at the end of the document
    if end_word_idx is None:
        end_word_idx = len(words)

    return start_word_idx, end_word_idx


def process_candidates_sapbert(sapbert_res, max_candidates=None):
    maxi = (
        max_candidates
        if max_candidates
        else max([len(entry["candidates"]) for entry in sapbert_res])
    )
    for entry in sapbert_res:
        seen_candidates = set()  # To track unique candidates
        new_candidates = []  # To store filtered candidates
        # Loop through the sublists of candidates
        for sublist in entry["candidates"]:
            # Iterate through each candidate in the sublist
            for candidate in sublist:
                if candidate not in seen_candidates:
                    seen_candidates.add(candidate)
                    new_candidates.append([candidate])
        # Pad the sublists with empty strings to match the maximum length
        new_candidates += [["ERROR_IGNORE_THIS"]] * (maxi - len(new_candidates))
        # Replace the original 'candidates' with the processed one
        entry["candidates"] = new_candidates
    return sapbert_res


### SECOND VERSION that incorporates sapbert case
def number_hit(df, column_name, hit_range):
    """
    Count the number of hits for each hit index specifically.

    Parameters:
    df (pd.DataFrame): DataFrame containing the hit index (position in the list where the prediction matches the gold CUI).
    column_name (str): Name of the column containing the hit index.
    hit_range (int): Number of candidates to consider (i.e., recall@k)

    Returns:
    dict: A dictionary where keys are indices (starting from 1) and values are counts of hits at each index.
    """
    # Initialize results dictionary
    results = {i + 1: 0 for i in range(hit_range)}

    for i in range(hit_range):
        results[i + 1] = len(df[df[column_name] == i])
    return results


### SECOND VERSION that incorporates sapbert case
def compute_recall(number_hits, hit_range, total_nb_mentions):
    """
    Function to compute the recall of the intial model (biencoder, crossencoder).
    ------
    number_hits : dict (Number of hits for each hit index)
    hit_range : int (Number of hits to consider = recall@k)
    total_nb_mentions : int (For unnormalized result = true performance)
    """
    unnormalized_recall = 0
    normalized_recall = 0
    results = []
    for hit_index in range(hit_range):
        res = number_hits[hit_index + 1]
        unnormalized_recall += res / total_nb_mentions
        normalized_recall += res / sum(number_hits.values())
        # Store the cumulative results for this hit_index
        results.append((unnormalized_recall, normalized_recall))
    return results


def retrieve_valid_mention_id(candidates_cuis, gold_cuis, hit_range):
    """
    Count the number of hits for each hit index specifically.

    Parameters:
    candidates_cuis (dict): {mention_id : list of predicted CUIs}
    gold_cuis (dict): {mention_id : list with the gold CUI}
    hit_range (int): Number of candidates to consider (i.e., recall@k)

    Returns:
    dict: A dictionary where keys are indices (starting from 1) and values are counts of hits at each index.
    """
    mentions_id_list = []
    # Loop through each mention and calculate hits for each index up to hit_range
    for mention_id in candidates_cuis:
        for idx, cui in enumerate(candidates_cuis[mention_id][:hit_range]):
            if (
                cui == gold_cuis[mention_id][0]
            ):  # Check if predicted CUI matches gold CUI
                mentions_id_list.append(mention_id)
                break  # Stop after the first hit, since we only count the first hit
    return mentions_id_list


def extract_cui(text):
    """
    Extracts the CUI from the text generated by the LLM.
    """
    # Expression pattern to match "MESH" or "OMIM"
    pattern = r"\b(UMLS|MESH|OMIM|NCBIGene):[A-Za-z0-9]+\b"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Return the matched string if found, otherwise return None
    return match.group(0) if match else None


def extract_last_cui(text):
    """
    Extracts the last CUI from the text generated by the LLM.
    """
    # Expression pattern to match "MESH", "OMIM"
    pattern = r"\b(?:UMLS|MESH|OMIM|NCBIGene):[A-Za-z0-9]+\b"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the last match if any are found, otherwise return None
    return matches[-1] if matches else None


def add_full_context(df, docs):
    """
    Add whole context to the dataset
    -------
    df : pd.DataFrame
    docs : dict {pmid : abstract}
    """
    contextualized_mentions = []
    for idx, row in df.iterrows():
        doc = docs[row["document_id"]]
        start = row["offsets"][0][0]  # start on the mention
        end = row["offsets"][-1][-1]  # end of the mention
        context_left = doc[:start]  # left context
        context_right = doc[end:]  # right context
        contextualized_mention = (
            context_left
            + "[ENTITY_START]"
            + row["deabbreviated_text"]
            + "[ENTITY_END]"
            + context_right
        )
        contextualized_mentions.append(contextualized_mention)

    df["contextualized_mention"] = contextualized_mentions


def add_context(df, docs, length=64):
    """
    Add surrounding context to the dataset
    Returns :
    - list of contextualized mentions
    - dictionary of mention_id to contextualized mention

    df : DataFrame
    docs : dict {pmid : abstract}
    """
    limited_contextualized_mentions = []
    contextMap = {}
    for idx, row in df.iterrows():
        doc = docs[row["document_id"]]
        start_word_idx, end_word_idx = get_word_indices(doc, row["offsets"][0])

        if start_word_idx is None or end_word_idx is None:
            limited_contextualized_mentions.append(doc)
            # print(idx)
            continue

        words = doc.split()
        mention_words = words[start_word_idx:end_word_idx]
        mention_length = len(mention_words)

        total_context_length = length - mention_length
        context_length = total_context_length // 2

        start_context_idx = max(0, start_word_idx - context_length)
        end_context_idx = min(len(words), end_word_idx + context_length)

        # Add the mention words back in their place
        limited_contextualized_mention_words = (
            words[start_context_idx:start_word_idx]
            + ["[ENTITY_START]"]
            + mention_words
            + ["[ENTITY_END]"]
            + words[end_word_idx:end_context_idx]
        )
        limited_contextualized_mention = " ".join(limited_contextualized_mention_words)

        limited_contextualized_mentions.append(limited_contextualized_mention)
        contextMap[row["mention_id"]] = limited_contextualized_mention

    df["limited_contextualized_mention"] = limited_contextualized_mentions
    return limited_contextualized_mentions, contextMap


def get_candidates_name(candidates, ontology):
    """
    Returns the name of the candidates
    ------
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    ontology : BiomedicalOntology object
    """
    candidates_name = {}
    for candidate in candidates:
        entity = ontology.entities.get(candidate)
        candidates_name[entity.cui] = entity.name
    return candidates_name


def get_candidates_data(candidates, ontology):
    """
    Returns the metadata of the candidates
    ------
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    ontology : BiomedicalOntology object
    """
    candidates_data = {}
    for candidate in candidates:
        entity = ontology.entities.get(candidate)
        if entity:
            entity_data = {
                "cui": entity.cui,
                "name": entity.name,
                "types": entity.types,
                "aliases": entity.aliases,
                "definition": entity.definition,
            }
            candidates_data[entity.cui] = entity_data
        else:
            candidates_data[candidate] = {
                "error": f"Entity for {candidate} not found, just ignore it."
            }
    return candidates_data


def get_candidates_data_v2(candidates, ontology):
    """
    Returns the metadata of the candidates (str : "name (aliases) [definition]")
    ------
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    ontology : BiomedicalOntology object
    """
    candidates_data = {}
    for candidate in candidates:
        entity = ontology.entities.get(candidate)
        if entity:
            entity_data = f"{entity.name}"
            if entity.aliases:
                entity_data += f" ({entity.aliases})"
            if entity.definition:
                entity_data += f" [{entity.definition}]"
            candidates_data[entity.cui] = entity_data
        else:
            candidates_data[candidate] = ""
    return candidates_data


def knn_query(model, index, query, k=5):
    """
    Find the top k most similar embeddings of the query from the corpus.
    ------
    model : SentenceTransformer model
    index : faiss index
    query : str (mention + surrounding context)
    k : int (number of similar embeddings to find)
    """
    # Generate embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().detach().numpy()
    query_embedding = query_embedding.reshape(1, -1)
    print(query_embedding.shape)
    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Perform the search
    distances, indices = index.search(query_embedding, k)

    # print("Query:", TestMap_cui2context[query])
    # print("\nTop 3 most similar sentences in the corpus:")

    # for i, idxs in enumerate(indices[0]):
    #     print(f"{i+1}. {corpus[idxs]} (Distance: {distances[0][i]})")

    return indices[0]


"""
Create a string for gpt prompt with : 
"Query : ... / Sentence (context+mention) : ... / Answer : cui
etc...
Query : ... / Sentence (context+mention) : ... / Answer : cui"
"""


def topk_examples(
    model,
    index,
    query,
    corpus,
    TrainMap_context2mention,
    train_mention2text,
    train_mention2gold,
    ontology,
    k=5,
):
    """
    Given a query (context sentence), returns the top k most similar contexts from the corpus.
    ------
    model : sentence embedding model
    index : faiss index for NN mention retrieval
    query : str (context sentence)
    corpus : list of str (all context sentences)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
    train_mention2gold : dict (mention_id to gold CUI)
    ontology : BiomedicalOntology object
    k : int (number of nearest neighbors)
    """
    indices = knn_query(model, index, query, k)
    result_list = []
    for i, idx in enumerate(indices):
        NN_mention = corpus[idx]
        # print("Nearest neighbor mention : ", NN_mention)
        if NN_mention not in TrainMap_context2mention:
            continue
        mention_id = TrainMap_context2mention[NN_mention]
        # print("mention_id :", mention_id)
        mention_text = train_mention2text[mention_id]
        cui = train_mention2gold[mention_id]
        # print("cui :", cui)
        gold = get_candidates_data(candidates=cui, ontology=ontology)
        # print("gold :", gold)
        result_list.append(
            f"Mention {i+1}: {mention_text} || Context: {NN_mention} || Correct CUI: {gold}"
        )

    res = "\n".join(result_list)

    return res


def topk_entities(
    model,
    index_entity,
    query,
    entity_corpus,
    entities_data2cui,
    k=5,
):
    """
    Given a query (mention text), returns the top k most similar entities from the entity corpus.
    ------
    model : sentence embedding model
    index_entity : faiss index for NN entity retrieval
    query : str (mention)
    entity_corpus : list of str (all context sentences)
    entities_data2cui : dict (entity data to CUI)
    k : int (number of nearest neighbors)
    """
    indices = knn_query(model, index_entity, query, k)
    result = []
    for i, idx in enumerate(indices):
        NN_data_entity = entity_corpus[idx]
        # print("Nearest neighbor mention : ", NN_entity)
        if NN_data_entity not in entities_data2cui:
            continue
        entity_cui = entities_data2cui[NN_data_entity]
        result.append(entity_cui)

    return result


def prompt_gpt(system_instructions, gpt_version, prompt):
    """
    system_instructions : str (instructions for the LLM)
    prompt : str (prompt text)
    gpt_version : Version of the GPT model
    """
    completion = openai.chat.completions.create(
        model=gpt_version,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0,
    )

    return completion.choices[0].message.content


def scoring(results, mention2gold):
    """
    Return the score of of the model
    -------
    results : dictionary {mention_id : {"predicted" : predicted CUI, "explanation" : explanation}}
    mention2gold : dictionary {mention_id : gold CUI}
    """
    score = 0
    errors = 0
    for key, value in results.items():
        if (
            key not in mention2gold or value["predicted"] is None
        ):  # e.g. If the model did not provide any answer because of token limitations
            errors += 1

        elif value["predicted"] in mention2gold[key]:
            score += 1
    nb_candidates = len(results) - errors
    return ((score / nb_candidates), score, nb_candidates, errors)


def recall_fn(results, mention2gold, ks):
    """
    Return a dictionary of recall@k scores for the model
    -------
    results : dictionary {mention_id : [CUI1, CUI2, etc...]}
    mention2gold : dictionary {mention_id : [gold CUI1, gold CUI2, etc...]}
    ks : list of top-k values to compute recall@k for
    """
    recall_scores = {f"recall@{k}": 0 for k in ks}
    errors = 0

    for mention_id, predicted_cuis_str in results.items():
        # If the result was already a list, no need to decode
        if isinstance(predicted_cuis_str, list):
            predicted_cuis = predicted_cuis_str
        else:
            # Clean up the string to ensure it's in a proper JSON format
            predicted_cuis_str = predicted_cuis_str.strip()

            # Remove "```json" and "```" markers if present
            if predicted_cuis_str.startswith("```json"):
                predicted_cuis_str = (
                    predicted_cuis_str.replace("```json", "").replace("```", "").strip()
                )

            # Handle newlines and extra characters like "None"
            predicted_cuis_str = predicted_cuis_str.replace("None", "null").replace(
                "\n", ""
            )

            # Try to parse as JSON first, then fall back to Python-like format
            try:
                # Attempt to load as JSON
                predicted_cuis = json.loads(predicted_cuis_str)
            except json.JSONDecodeError:
                # If JSON loading fails, attempt to parse using ast.literal_eval()
                try:
                    predicted_cuis = ast.literal_eval(predicted_cuis_str)
                    if not isinstance(predicted_cuis, list):
                        raise ValueError(
                            f"Expected a list but got {type(predicted_cuis)} for mention_id {mention_id}"
                        )
                except (ValueError, SyntaxError) as e:
                    print(f"Error decoding for mention_id {mention_id}: {e}")
                    errors += 1
                    continue  # Skip this mention if there's an error

        # Check if the predicted CUIs are in the correct format else skip this mention because the LLM can do so many different things, it's impossible to add all "if" condiditions to correct all of them
        if any(isinstance(item, (set, list, dict)) for item in predicted_cuis):
            print(
                f"Error: Found set, list, or dict in mention_id {mention_id}. Skipping this mention."
            )
            errors += 1
            continue

        if not isinstance(predicted_cuis, list):
            predicted_cuis = list(predicted_cuis)

        # Get the gold CUIs for this mention
        gold_cuis = mention2gold[mention_id]

        for k in ks:
            # Get the top-k predicted CUIs
            top_k_predictions = predicted_cuis[:k]

            # Calculate how many of the gold CUIs are in the top-k predictions
            correct_predictions = len(set(top_k_predictions) & set(gold_cuis))

            # Calculate recall@k for this mention
            recall_at_k = correct_predictions / len(gold_cuis)

            # Add the recall for this mention to the total recall for recall@k
            recall_scores[f"recall@{k}"] += recall_at_k

    # Average the recall scores over all mentions
    nb_candidates = len(results) - errors
    recall_scores = {k: v / nb_candidates for k, v in recall_scores.items()}

    return (recall_scores, nb_candidates, errors)


def error_analysis(results, ontology, mention2gold, mention2context):
    """
    Returns a dict of mention_id for mentions that were not correctly predicted
    Each dict contains the gold cui, the predicted cui, the context of the mention.
    -------
    results : dictionary {mention_id : predicted CUI}
    ontology : BiomedicalOntology object
    mention2gold : dictionary {mention_id : gold CUI}
    mention2context : dictionary {mention_id : context}
    """
    error_mentions = defaultdict(dict)
    for mention_id, predicted_cui in results.items():
        gold_cui = mention2gold[mention_id]
        if predicted_cui["predicted"] not in gold_cui:
            predicted_cui_metadata = get_candidates_data(
                candidates=[predicted_cui["predicted"]], ontology=ontology
            )
            gold_cui_metadata = get_candidates_data(
                candidates=[gold_cui[0]], ontology=ontology
            )
            mention_context = mention2context[mention_id]
            error_mentions[mention_id] = {
                "query_context": mention_context,
                "gold_cui": gold_cui_metadata,
                "predicted_cui": predicted_cui_metadata,
            }

    return error_mentions


def generate_prompt_text(
    mention,
    context,
    candidates,
    topk_examples=None,
    reasoning=False,
    recall=False,
    recall_k=5,
    analysis=None,
    analysis_version="v1",
    mention_id=None,
):
    """
    Generate prompt text based on the prompt version and reasoning flag.
    ----------------
    mention : str (name of the mention to be linked)
    context : str (context where the mention appears)
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    topk_examples : str (top k examples of similar contexts)
    reasoning : bool (whether to generate reasoning or not in the output)
    recall : bool (whether to generate a prompt for recall or not)
    recall_k : int (number of candidates to rank)
    analysis : list of dict of dict [{mention_id : {mention : CUI, analysis : str}},etc...] [Used in MoA]
    analysis_version : str : Candidates from proposer alone (v1) all candidates + analysis from proposers (v2)
    mention_id : str (mention_id) [Used for generating analysis text]
    """
    analysis_text = ""
    if analysis and analysis_version == "v2":
        analysis_text = (
            f"Those are the analysis made by {len(analysis)} other experts:\n"
        )
        analysis_text += "\n".join(
            [
                f"- Analysis of expert {i + 1}: {a[mention_id]}"
                for i, a in enumerate(analysis)
            ]
        )
    reasoning_clause = (
        "Use step-by-step reasoning."
        if reasoning
        else "Use step-by-step reasoning but do not add provide any explanations to me ! I only want the final answer"
    )
    text = ""
    if topk_examples:
        text += f"Here are a few examples:\n{topk_examples}\n"
    if recall:
        return (
            text
            + f"""
        This is the specific mention that needs to be linked to the correct entity: {mention}\n
        Here is the context where the mention appears::\n{context}\n
        Below are the candidate entities you can choose from::\n{candidates}\n
        {analysis_text}\n
        Your assistance in completing this task would be invaluable to me. 
        Please rank the top {recall_k} candidate entities from best to worst. {reasoning_clause}
        Return the results in JSON format as a list of CUIs ["CUI1", "CUI2", "CUI3", ...].
        For instance "["NCBIGene:0000000", "NCBIGene:1000000", "OMIM:000000, ...]" and "["NCBIGene:0000001", "MESH:D100001", "OMIM:000001", ...]" are examples of a valid format.
        """
        )
    return (
        text
        + f"""
    This is the specific mention that needs to be linked to the correct entity: {mention}\n
    This is the context where the mention appears:\n{context}\n
    These are the candidate entities to choose from:\n{candidates}\n
    {analysis_text}\n
    You MUST PROVIDE an ANSWER among the candidates. {reasoning_clause}
    Return the results in a JSON format containing only the CUI values (e.g. "MESH:D000000" is an example of a valid format).
    """
    )


# For nlm_gene
# Return the results in a JSON format containing only the CUI values (e.g., "NCBIGene:0000000" or "MESH:D000000" are examples of a valid format).


def prompt_llm_hf(system_instructions, llm, tokenizer, prompt):
    """
    system_instructions : str (instructions for the LLM)
    llm : Hugging Face model (e.g., AutoModelForCausalLM)
    tokenizer : Hugging Face tokenizer (e.g., AutoTokenizer)
    prompt : str (prompt text)
    """
    # Format the input as a chat prompt
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
    ]
    # Use tokenizer to format the chat template
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the input for the Hugging Face model
    model_inputs = tokenizer(chat_input, return_tensors="pt").to(llm.device)

    # Generate output tokens using the model
    generated_ids = llm.generate(
        **model_inputs, no_repeat_ngram_size=35, max_new_tokens=2048
    )

    # Decode the generated tokens into text
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return answer


def prompt_vllm(system_instructions, llm, tokenizer, sampling_params, prompt):
    """
    system_instructions : str (instructions for the LLM)
    llm : LLM model
    tokenizer : AutoTokenizer
    sampling_params : SamplingParams config
    prompt : str (prompt text)
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Decode the generated tokens into text
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    answer = outputs[0].outputs[0].text

    return answer


def extract_last_assistant_message(output):
    """
    Extract the last assistant message from a structured output.
    ----------
    - output : str
    The generated text including system, user, and assistant messages.
    Returns: (str) The last assistant's message.
    """
    # Split the output by role markers (system, user, assistant)
    chunks = output.split("\nassistant\n")
    # The last chunk is the assistant's final message
    if len(chunks) > 1:
        return chunks[-1].strip()  # Return the last assistant's message
    else:
        return output.strip()  # Fallback if no "assistant" marker exists


def extract_json_from_text(text):
    """
    Extract the JSON portion from a given text.
    ----------
    - text : str
      The input text containing a JSON block.
    Returns: (str) The JSON block if found, else an empty string.
    """
    # Use a regex pattern to match the JSON block
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()  # Return the JSON block
    else:
        return ""  # Return empty string if no JSON block is found


def parse_answer(answer):
    """
    Function to parse the answer as JSON, handling mismatched quotes, unescaped characters,
    smart quotes, trailing commas, and more.
    """
    # Clean the answer string by stripping unwanted artifacts
    answer = answer.strip("```json").strip("```").strip()

    try:
        # Attempt to parse as JSON directly
        return json.loads(answer)
    except json.JSONDecodeError:
        # Start cleaning known problematic patterns
        try:
            # Normalize smart quotes to standard quotes
            answer = re.sub(r"[“”]", '"', answer)

            # Fix mismatched or malformed quotes
            answer = re.sub(
                r"(?<!\\)'", '"', answer
            )  # Replace single quotes with double quotes

            # Fix keys with single quotes
            answer = re.sub(r"(?<=[{,])\s*'([a-zA-Z0-9_]+)'\s*:", r'"\1":', answer)

            # Remove invalid control characters
            answer = re.sub(r"[\t\r]", "", answer)

            # Remove trailing commas
            answer = re.sub(r",\s*([}\]])", r"\1", answer)

            # Fix unescaped Unicode characters (e.g., \u201c)
            answer = re.sub(
                r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), answer
            )

            # Attempt to parse again
            return json.loads(answer)
        except json.JSONDecodeError as e:
            # Log the remaining problematic string for debugging
            print("Failed to parse cleaned JSON:", str(e))
            # print("Problematic JSON snippet:", answer)
            return None


def evaluate(
    llm,
    nlp_model,
    tokenizer,
    index,
    system_instructions,
    mentions,
    ontology,
    corpus,
    mention2context,
    mention2candidates,
    mention2text,
    mention2gold,
    TrainMap_context2mention,
    train_mention2text,
    train_mention2gold,
    k,
    raw_result_path=False,
    sampling_params=None,
    reasoning=False,
    analysis_version="v1",
    analysis=None,
    recall=False,
    recall_k=None,
):
    """
    Run prompt function (prompt vllm, prompt gpt) for each mention in the list of mentions.
    Returns a dict of dict {mention_id : {predicted CUI, explanation}} for accuracy
    Returns a dict {mention_id : [CUI1, CUI2, etc...]} for recall
    -------
    llm : LLM model or GPT version (gpt-4o-mini, gpt-4o)
    nlp_model : SentenceTransformer model
    tokenizer : AutoTokenizer
    index : faiss index
    system_instructions : str (instructions for the LLM)
    mentions : list (mention_ids)
    ontology : BiomedicalOntology object
    corpus : list of str (all context sentences)
    mention2context : dict (mention_id : context)
    mention2candidates : dict (mention_id : list of candidate CUIs)
    mention2text : dict (mention_id : mention name)
    mention2gold : dict (mention_id : gold CUI)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
    train_mention2gold : dict (mention_id to gold CUI)
    k : int (number of nearest neighbors)
    raw_result_path : str (path to save the raw results)
    sampling_params : SamplingParams config
    reasoning : bool (whether to use reasoning or not)
    analysis_version : str (version of the analysis prompt) : Candidates from proposer alone ("v1") all candidates + analysis from proposers ("v2")
    analysis : list of dict of dict [{mention_id : {mention : CUI, analysis : str}},etc...] [Used in MoA]
    recall : bool (whether to generate a prompt for recall or not)
    recall_k : int (number of candidates to rank)
    """
    results = defaultdict(dict)
    raw_results = []
    for i in range(len(mentions)):
        mention_id = mentions[i]
        if i == 0:
            print("mention ID :", type(mention_id))
        mention_name = mention2text[mention_id]
        context = mention2context[mention_id]

        if analysis and analysis_version == "v1":
            candidates_cui = [cand[mention_id]["predicted"] for cand in analysis]
            candidates = get_candidates_data(candidates_cui, ontology)
        else:
            candidates = get_candidates_data(mention2candidates[mentions[i]], ontology)

        topk = topk_examples(
            model=nlp_model,  # sentence transformer model
            index=index,
            query=context,
            corpus=corpus,
            TrainMap_context2mention=TrainMap_context2mention,
            train_mention2text=train_mention2text,
            train_mention2gold=train_mention2gold,
            ontology=ontology,
            k=k,
        )

        prompt = generate_prompt_text(
            mention=mention_name,
            context=context,
            candidates=candidates,
            topk_examples=topk,
            reasoning=reasoning,
            recall=recall,
            recall_k=recall_k,
            analysis_version=analysis_version,
            analysis=analysis,
            mention_id=mention_id if analysis else None,
        )

        start_time = time.time()

        if llm in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            text = prompt_gpt(
                system_instructions=system_instructions,
                gpt_version=llm,
                prompt=prompt,
            )
        else:
            # output = prompt_llm_hf(
            #     system_instructions=system_instructions,
            #     llm=llm,
            #     tokenizer=tokenizer,
            #     prompt=prompt,
            # )
            # text = extract_last_assistant_message(output)

            text = prompt_vllm(
                prompt=prompt,
                system_instructions=system_instructions,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
            )

        end_time = time.time()
        # Measure and display elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        raw_result = {"pmid": mention_id, "llm_output": text}
        raw_results.append(raw_result)

        if raw_result_path:
            with open(f"{raw_result_path}.jsonl", "a") as raw_results_f:
                raw_results_f.write(json.dumps(raw_result) + "\n")

        if recall:
            print(
                "mention ID :",
                mention_id,
                "|| gold : ",
                mention2gold[mention_id],
                "|| LLM answer :",
                text,
            )
            results[mention_id] = text

        else:
            cand = extract_last_cui(text)
            print(
                "mention ID :",
                mention_id,
                "|| gold : ",
                mention2gold[mention_id],
                "|| LLM answer :",
                cand,
            )
            results[mention_id] = {
                "predicted": cand,
                "gold": mention2gold[mention_id],
                "explanation": text,
            }

        if i % 20 == 0:
            print(f"i = {i}")

    return results


def mcnemar_test(target, base_predictions, llm_predictions):
    """
    Function to perform McNemar's test for comparing two models.
    ------
    Parameters:
    - target: dict (mention_id : gold CUI)
    - base_predictions: dict (mention_id : predicted CUI from base model)
    - llm_predictions: dict (mention_id : predicted CUI from base model + LLM)
    """
    both_correct = 0
    base_correct_llm_incorrect = 0
    llm_correct_base_incorrect = 0
    both_incorrect = 0
    # Iterate through each example in the dictionaries
    for cui, truth in target.items():
        pred1 = base_predictions.get(cui)
        pred2 = llm_predictions.get(cui)

        if pred1 == truth and pred2 == truth:
            both_correct += 1
        elif pred1 == truth and pred2 != truth:
            base_correct_llm_incorrect += 1
        elif pred1 != truth and pred2 == truth:
            llm_correct_base_incorrect += 1
        elif pred1 != truth and pred2 != truth:
            both_incorrect += 1

    # Create the contingency table
    # Contingency table format:
    # [[both correct, model1 correct and model2 incorrect],
    #  [model2 correct and model1 incorrect, both incorrect]]
    contingency_table = np.array(
        [
            [both_correct, base_correct_llm_incorrect],
            [llm_correct_base_incorrect, both_incorrect],
        ]
    )

    # Apply McNemar's test
    result = mcnemar(
        contingency_table, exact=True
    )  # Use exact=True for small sample sizes
    return result
