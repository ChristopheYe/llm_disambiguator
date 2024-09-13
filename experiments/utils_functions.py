import pickle
import ujson
import json
import sys
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import random
import faiss
import re
import openai

from tqdm import tqdm
from collections import defaultdict
from typing import Optional

from bioel.utils.umls_utils import UmlsMappings
from bioel.utils.bigbio_utils import (
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    DATASET_NAMES,
    VALIDATION_DOCUMENT_IDS,
)
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
from bioel.models.arboel.biencoder.data.data_utils import process_ontology
from bioel.evaluate import Evaluate

from torch.utils.data import DataLoader


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


def process_ontology(
    ontology: BiomedicalOntology,
    # data_path: str,
    tax2name_filepath: str = None,
):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - tax2name_filepath : str
        Path to the taxonomy to name file
    """

    # Check if equivalent CUIs are present for the first entity
    first_entity_cui = next(iter(ontology.entities))
    equivalant_cuis = bool(ontology.entities[first_entity_cui].equivalant_cuis)
    print("equivalant cuis :", equivalant_cuis)
    # "If dictionary already processed, load it else process and load it"
    # entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    # if os.path.isfile(entity_dictionary_pkl_path):
    #     print("Loading stored processed entity dictionary...")
    #     with open(entity_dictionary_pkl_path, "rb") as read_handle:
    #         entities = pickle.load(read_handle)

    #     return entities, equivalant_cuis

    if tax2name_filepath:
        with open(tax2name_filepath, "r") as f:
            tax2name = ujson.load(f)

    ontology_entities = []
    for cui, entity in tqdm(ontology.entities.items()):
        new_entity = {}

        # if ontology.name.lower() in ["umls"]:
        #     with open(os.path.join(data_path, "tui2type_hierarchy.json"), "r") as f:
        #         type2geneology = ujson.load(f)
        #     entity.types = get_type_gcd(entity.types, type2geneology)

        new_entity["cui"] = entity.cui
        new_entity["title"] = entity.name
        new_entity["types"] = f"{entity.types}"

        if entity.aliases:
            if entity.definition:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types} : {entity.aliases} ) [{entity.definition}]"
                    )

                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types} : {entity.aliases} ) [{entity.definition}]"
                    )

            else:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types} : {entity.aliases} )"
                    )
                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types} : {entity.aliases} )"
                    )

        else:
            if entity.definition:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types}) [{entity.definition}]"
                    )

                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types}) [{entity.definition}]"
                    )
            else:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types})"
                    )
                else:
                    new_entity["description"] = f"{entity.name} ({entity.types})"

        if hasattr(entity, "metadata") and entity.metadata:
            new_entity["description"] += f" {entity.metadata}"

        if equivalant_cuis:
            new_entity["cuis"] = entity.equivalant_cuis

        ontology_entities.append(new_entity)

    return ontology_entities, equivalant_cuis


def extract_cui(text):
    """
    Extracts the CUI from the text generated by the LLM.
    """
    # Define the regular expression pattern to match "MESH" or "OMIM"
    pattern = r"\b(MESH|OMIM|NCBIGene):[A-Za-z0-9]+\b"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Return the matched string if found, otherwise return None
    return match.group(0) if match else None


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
            candidates_data[candidate] = {"error": f"Entity for {candidate} not found"}
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
    query : str (context sentence)
    corpus : list of str (all context sentences)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
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


def prompt_gpt(
    mention, context, candidates, system_instructions, topk_examples, llm="gpt-4o-mini"
):
    """
    mention : str (name of the mention to be linked)
    context : str (context where the mention appears)
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    system_instructions : str (instructions for the LLM)
    topk_examples : str (top k examples of similar contexts)
    model : str (name of the model to use)
    """
    completion = openai.chat.completions.create(
        model=llm,
        messages=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": f""" 
                Here are a few examples : \n
                {topk_examples} \n

                This is the specific mention that needs to be linked to the correct entity : {mention} \n
                
                This is the context where the mention appears : {context} \n
                
                These are the candidate entities to choose from: {candidates} \n
                You must provide an answer among the candidates. \n
                
                Return the answer in the following format : CUI \n
                
                Do not add any explanations !
                """,
            },
        ],
        max_tokens=2048,
        temperature=0,
    )
    # Do not add any explanations !
    # Do not return anything except the correct CUI from the list of candidates.
    # If there are multiple possible entities, you can return multiple IDs.

    # Be careful not to mix it with another mention that could appear in the context !

    # This is the specific mention that needs to be linked to the correct entity : {mentions[i]}
    return completion.choices[0].message.content


def evaluate_gpt(
    mentions,
    mention2context,
    mention2biencoder_candidates,
    ontology,
    k,
    corpus,
    TrainMap_context2mention,
    train_mention2text,
    train_mention2gold,
    system_instructions,
    index,
    nlp_model,
    llm="gpt-4o-mini",
):
    """
    Run "prompt" function for each mention in the list of mentions.
    Returns a dictionary {mention_id : predicted CUI}
    -------
    mentions : list (mention_ids)
    ontology : BiomedicalOntology object
    mention2context : dict (mention_id : context)
    mention2biencoder_candidates : dict (mention_id : list of candidate CUIs)
    k : int (number of nearest neighbors)
    corpus : list of str (all context sentences)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
    train_mention2gold : dict (mention_id to gold CUI)
    system_instructions : str (instructions for the LLM)
    index : faiss index
    nlp_model : sentence-transformers model
    llm : str (name of the model)
    """
    results = {}
    for i in range(len(mentions)):
        mention = mentions[i]
        context = mention2context[mention]
        candidates = get_candidates_data(
            candidates=mention2biencoder_candidates[mentions[i]], ontology=ontology
        )
        # candidates = get_candidates_data_v2(mention2crossencoder_candidates[mention])
        topk = topk_examples(
            model=nlp_model,
            index=index,
            query=context,
            corpus=corpus,
            TrainMap_context2mention=TrainMap_context2mention,
            train_mention2text=train_mention2text,
            train_mention2gold=train_mention2gold,
            ontology=ontology,
            k=k,
        )

        text = prompt_gpt(
            mention=mention,
            context=context,
            system_instructions=system_instructions,
            candidates=candidates,
            topk_examples=topk,
            llm=llm,
        )

        cand = extract_cui(text)
        results[mention] = cand

        if i % 20 == 0:
            print(f"i = {i}")

    return results


def scoring(results, mention2gold):
    """
    Return the score of of the model
    -------
    results : dictionary {mention_id : predicted CUI}
    mention2gold : dictionary {mention_id : gold CUI}
    """
    score = 0
    for key, value in results.items():
        if value in mention2gold[key]:
            score += 1

    return score / len(results)


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
        if predicted_cui not in gold_cui:
            predicted_cui_metadata = get_candidates_data(
                candidates=[predicted_cui], ontology=ontology
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


def prompt_vllm(
    mention,
    context,
    system_instructions,
    candidates,
    topk_examples,
    llm,
    tokenizer,
    sampling_params,
):
    """
    mention : str (name of the mention to be linked)
    context : str (context where the mention appears)
    system_instructions : str (instructions for the LLM)
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    topk_examples : str (top k examples of similar contexts)
    llm : LLM model
    tokenizer : AutoTokenizer
    sampling_params : SamplingParams config
    """
    prompt_text = f"""
    Here are a few examples: \n
    {topk_examples} \n

    This is the specific mention that needs to be linked to the correct entity: {mention} \n

    This is the context where the mention appears: \n
    {context} \n
    
    These are the candidate entities to choose from: \n
    {candidates} \n
    
    You MUST PROVIDE an ANSWER among the candidates. \n

    Return the answer in the following format: CUI
    For instance : "MESH:D000000" "OMIM:000000" are valid answers. \n
    Reason step by step but do not add provide any explanations to me ! I only want the final answer.
    """
    # Do not add provide any explanations ! But you must give AN answer.

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt_text},
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Decode the generated tokens into text
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    answer = outputs[0].outputs[0].text

    return answer


def evaluate_vllm(
    llm,
    nlp_model,
    tokenizer,
    index,
    system_instructions,
    mentions,
    ontology,
    corpus,
    mention2context,
    mention2biencoder_candidates,
    mention2text,
    TrainMap_context2mention,
    train_mention2text,
    train_mention2gold,
    k,
    sampling_params,
):
    """
    Run "prompt" function for each mention in the list of mentions.
    Returns a dictionary {mention_id : predicted CUI}
    -------
    llm : LLM model
    nlp_model : SentenceTransformer model
    tokenizer : AutoTokenizer
    index : faiss index
    system_instructions : str (instructions for the LLM)
    mentions : list (mention_ids)
    ontology : BiomedicalOntology object
    corpus : list of str (all context sentences)
    mention2context : dict (mention_id : context)
    mention2biencoder_candidates : dict (mention_id : list of candidate CUIs)
    mention2text : dict (mention_id : mention name)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
    train_mention2gold : dict (mention_id to gold CUI)
    k : int (number of nearest neighbors)
    sampling_params : SamplingParams config
    """
    results = {}
    for i in range(len(mentions)):
        mention_id = mentions[i]
        mention_name = mention2text[mention_id]
        context = mention2context[mention_id]
        candidates = get_candidates_data(
            mention2biencoder_candidates[mentions[i]], ontology
        )
        # candidates = get_candidates_data_v2(mention2crossencoder_candidates[mention])
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

        text = prompt_vllm(
            mention=mention_name,
            context=context,
            system_instructions=system_instructions,
            candidates=candidates,
            topk_examples=topk,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        print("mention ID :", mention_id, "|| LLM answer :", text)
        cand = extract_cui(text)
        results[mention_id] = cand
        if i % 20 == 0:
            print(f"i = {i}")

    return results


def prompt_vllm_aqlm(
    mention,
    context,
    system_instructions,
    candidates,
    topk_examples,
    llm,
    tokenizer,
    sampling_params,
):
    """
    mention : str (name of the mention to be linked)
    context : str (context where the mention appears)
    system_instructions : str (instructions for the LLM)
    candidates : list of list of CUIs : [[cui1], [cui2, cui3], ...]
    topk_examples : str (top k examples of similar contexts)
    llm : LLM model
    tokenizer : AutoTokenizer
    sampling_params : SamplingParams config
    """

    prompt_text = f"""
    System Instructions: {system_instructions} \n
    
    Here are a few examples: \n
    {topk_examples} \n

    This is the specific mention that needs to be linked to the correct entity: {mention} \n

    This is the context where the mention appears: \n
    {context} \n
    
    These are the candidate entities to choose from: \n
    {candidates} \n
    
    You MUST PROVIDE an ANSWER among the candidates. \n

    Return the answer in the following format: CUI
    For instance : "MESH:D000000" "OMIM:000000" are valid answers. \n
    Do not add provide any explanations ! But you MUST give ONE answer.
    """
    conversations = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
    )

    # Decode the generated tokens into text
    outputs = llm.generate(
        [conversations], sampling_params=sampling_params, use_tqdm=False
    )
    answer = outputs[0].outputs[0].text

    return answer


def evaluate_vllm_aqlm(
    llm,
    nlp_model,
    tokenizer,
    index,
    system_instructions,
    mentions,
    ontology,
    corpus,
    mention2context,
    mention2biencoder_candidates,
    mention2text,
    TrainMap_context2mention,
    train_mention2text,
    train_mention2gold,
    k,
    sampling_params,
):
    """
    Run "prompt" function for each mention in the list of mentions.
    Returns a dictionary {mention_id : predicted CUI}
    -------
    llm : LLM model
    nlp_model : SentenceTransformer model
    tokenizer : AutoTokenizer
    index : faiss index
    system_instructions : str (instructions for the LLM)
    mentions : list (mention_ids)
    ontology : BiomedicalOntology object
    corpus : list of str (all context sentences)
    mention2context : dict (mention_id : context)
    mention2biencoder_candidates : dict (mention_id : list of candidate CUIs)
    mention2text : dict (mention_id : mention name)
    TrainMap_context2mention : dict (context sentence to mention_id)
    train_mention2text : dict (mention_id to mention name)
    train_mention2gold : dict (mention_id to gold CUI)
    k : int (number of nearest neighbors)
    sampling_params : SamplingParams config
    """
    results = {}
    for i in range(len(mentions)):
        mention_id = mentions[i]
        mention_name = mention2text[mention_id]
        context = mention2context[mention_id]
        candidates = get_candidates_data(
            mention2biencoder_candidates[mentions[i]], ontology
        )
        # candidates = get_candidates_data_v2(mention2crossencoder_candidates[mention])
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
        text = prompt_vllm_aqlm(
            mention=mention_name,
            context=context,
            system_instructions=system_instructions,
            candidates=candidates,
            topk_examples=topk,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        print("mention ID :", mention_id, "|| LLM answer :", text)
        cand = extract_cui(text)
        results[mention_id] = cand
        if i % 20 == 0:
            print(f"i = {i}")

    return results
