import pickle
import json
import sys
import os
import time

import pandas as pd
import numpy as np
import torch
import random
import faiss

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
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from ids import open_ai_api_key

from utils_functions import *

import openai

openai.api_key = open_ai_api_key
import re
import ujson
import logging
from collections import Counter, defaultdict


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

device_1 = torch.device("cuda:0")

sampling_params = SamplingParams(
    temperature=0, top_p=0.9, max_tokens=5000, stop=["<|eot_id|>"]
)

set_seed(12)

# Start time
start_time = time.time()

############ I) LOAD THE DATA (ONTOLOGY + MENTIONS) ############

# ----------------- START ARGUMENTS ----------------- #

# # ncbi_disease
# dataset_name = "ncbi_disease"
# ontology_dir = "/mitchell/entity-linking/kbs/medic.tsv"
# name = "medic"
# ontology2 = BiomedicalOntology.load_medic(filepath=ontology_dir, name=name)

# gnormplus & nlm_gene
dataset_name = "nlm_gene"
# dataset_name = "gnormplus"
entrez_dict = {
    "name": "entrez",
    "filepath": "/mitchell/entity-linking/el-robustness-comparison/data/gene_info.tsv",
    "dataset": f"{dataset_name}",
}
ontology2 = BiomedicalOntology.load_entrez(**entrez_dict)

# # nlm_chem
# dataset_name = "nlmchem"
# mesh_dict = {"name": "mesh", "filepath": "/mitchell/entity-linking/2017AA/META/"}
# ontology2 = BiomedicalOntology.load_mesh(**mesh_dict)

# # mm_st21pv
# dataset_name = "medmentions_st21pv"
# umls_dict_st21pv = {
#     "name": "umls",
#     "filepath": "/mitchell/entity-linking/2017AA/META/",
#     "path_st21pv_cui": "/home2/cye73/data_test2/arboel/medmentions_st21pv/umls_cuis_st21pv.json",
# }
# ontology2 = BiomedicalOntology.load_umls(**umls_dict_st21pv)


number_candidates = 40
print("number of candidates to consider :", number_candidates)

EL_model = "arboel"
# EL_model = "sapbert"

device = device_1

analysis = None
analysis_version = "v1"  # v1 for default, v2 for MoA
recall = False
recall_k = 5
k = 3

llm_model = "mistralai/Mistral-Nemo-Instruct-2407"
llm_subname = "Mistral-Nemo-Instruct-2407"

# llm_subname = "gpt-4o-2024-08-06"

# ----------------- END ARGUMENTS ----------------- #

path_to_abbrev = "/home2/cye73/data_test2/abbreviations.json"
dataset = load_bigbio_dataset(dataset_name)
dataset = add_deabbreviations(dataset, path_to_abbrev)

dataset_df = dataset_to_df(dataset)
test_df = dataset_df[dataset_df["split"] == "test"]
train_df = dataset_df[dataset_df["split"] == "train"]

docs = dataset_to_documents(dataset)

add_full_context(df=test_df, docs=docs)
add_full_context(df=train_df, docs=docs)

_, TestMap_mention2context = add_context(df=test_df, docs=docs)
corpus, TrainMap_mention2context = add_context(df=train_df, docs=docs)
_, _ = add_context(df=dataset_df, docs=docs)
TrainMap_context2mention = {v: k for k, v in TrainMap_mention2context.items()}


############ II) CREATE THE RAG-LIKE SYSTEM FOR BETTER ICL IN THE PROMPT ############
model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # It's worse
model.to(device)
# Generate embeddings for the corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
corpus_embeddings = corpus_embeddings.cpu().detach().numpy()

embedding_dimension = corpus_embeddings.shape[1]

# Create the HNSW index with the correct arguments
M = 32  # Number of neighbors in the HNSW graph
index = faiss.IndexHNSWFlat(embedding_dimension, M)

# Normalize the corpus embeddings if using cosine similarity
faiss.normalize_L2(corpus_embeddings)

# Add the embeddings to the index
index.add(corpus_embeddings)


dataset_names = [f"{dataset_name}"]
model_names = [f"{EL_model}"]

path_to_result = {f"{dataset_name}": {}}
if EL_model == "arboel":  # arboel
    path_to_result[dataset_name][
        EL_model
    ] = f"candidates/{dataset_name}/arboel_{dataset_name}.json"
elif EL_model == "sapbert":
    if dataset_name == ("nlm_gene" or "gnormplus"):
        path_to_result[dataset_name][
            EL_model
        ] = f"candidates/{dataset_name}/{EL_model}_{dataset_name}_real2.json"  # _real2.json
    else:
        path_to_result[dataset_name][
            EL_model
        ] = f"candidates/{dataset_name}/{EL_model}_{dataset_name}_real.json"

eval_strategies = ["basic"]
evaluator = Evaluate(
    dataset_names=dataset_names,
    model_names=model_names,
    path_to_result=path_to_result,
    eval_strategies=eval_strategies,
    abbreviations_path=path_to_abbrev,
)
evaluator.load_results()
evaluator.process_datasets()
evaluator.evaluate()

results = evaluator.full_results["basic"][f"{dataset_name}"]

# Base columns common to all models
base_cols = [
    "document_id",
    "offsets",
    "deabbreviated_text",
    "db_ids",
    "mention_id",
    "joined_offsets",
]

# Add model-specific columns based on EL_model

model_cols = [f"{EL_model}_resolve_abbrev", f"{EL_model}_resolve_abbrev_min_hit_index"]
rename_map = {
    f"{EL_model}_resolve_abbrev": f"{EL_model}_candidates",
    f"{EL_model}_resolve_abbrev_min_hit_index": f"{EL_model}_hit_index",
}

# Combine base and model-specific columns
cols = base_cols + model_cols
print(results.columns)
# Select and rename columns in results
filtered_results = results[cols].rename(columns=rename_map)
# Filter results based on hit index
filtered_results = filtered_results[
    filtered_results[f"{EL_model}_hit_index"] < number_candidates
]


############ III) FILTER THE CANDIDATES ############
filtered_results = filtered_results[
    filtered_results[f"{EL_model}_hit_index"] < number_candidates
]
total_mentions = len(results)
total_mentions_with_hit_index = len(filtered_results)

print("total mentions :", total_mentions)
print(
    "total mentions with hit index = number of mentions to evaluate:",
    total_mentions_with_hit_index,
)


############ IV) DATA PREPROCESSING FOR THE PROMPT USED IN THE LLM ############
train_mentions = []
train_mention2context = {}
train_mention2gold = {}
train_mention2text = {}
for idx, row in train_df.iterrows():
    train_mention2gold[row["mention_id"]] = row["db_ids"]
    train_mentions.append(row["mention_id"])
    train_mention2text[row["mention_id"]] = row["deabbreviated_text"]
    train_mention2context[row["mention_id"]] = row["limited_contextualized_mention"]


mention2context = {}
for idx, row in test_df.iterrows():
    mention2context[row["mention_id"]] = row["limited_contextualized_mention"]

mentions = []
mention2gold = {}
mention2text = {}
mention2arboel_candidates = {}
mention2sapbert_candidates = {}
mention2hit = {}
for idx, row in filtered_results.iterrows():
    # Only consider row if hit_index < max number of candidates
    if row[f"{EL_model}_hit_index"] < number_candidates:
        if EL_model == "arboel":
            mention2arboel_candidates[row["mention_id"]] = [
                el[0] for el in row["arboel_candidates"][:number_candidates]
            ]
        elif EL_model == "sapbert":
            mention2sapbert_candidates[row["mention_id"]] = [
                el[0] for el in row["sapbert_candidates"][:number_candidates]
            ]
        mention2gold[row["mention_id"]] = row["db_ids"]
        mentions.append(row["mention_id"])
        mention2text[row["mention_id"]] = row["deabbreviated_text"]
        mention2hit[row["mention_id"]] = row[f"{EL_model}_hit_index"]


############ V) COMPUTE THE RESULTS FROM THE ORIGINAL ENTITY-LINKING MODELS ############

number_hits_arboel = 0
number_hits_sapbert = 0


#### ARBOEL
if EL_model == "arboel":
    number_hits_arboel = number_hit(
        mention2arboel_candidates, mention2gold, number_candidates
    )
    print("number hits arboel :", number_hits_arboel)

    arboel_results = compute_recall(
        number_hits_arboel, number_candidates, total_mentions
    )
    print("Crossencoder:")
    for i, (unnormalized, normalized) in enumerate(arboel_results):
        print(
            f"recall {i+1}: Normalized = {normalized:.4f}, Unnormalized = {unnormalized:.4f}"
        )

#### SAPBERT
if EL_model == "sapbert":
    number_hits_sapbert = number_hit(
        mention2sapbert_candidates, mention2gold, number_candidates
    )
    print("number hits sapbert :", number_hits_sapbert)

    sapbert_results = compute_recall(
        number_hits_sapbert, number_candidates, total_mentions
    )
    print("Sapbert:")
    for i, (unnormalized, normalized) in enumerate(sapbert_results):
        print(
            f"recall {i+1}: Normalized = {normalized:.4f}, Unnormalized = {unnormalized:.4f}"
        )


assert len(mentions) == len(
    filtered_results
), f"Length mismatch: mentions has {len(mentions)} elements, but it should have {len(filtered_results)} elements. Check the condition 'row['sapbert/arboel_hit_index'] < number_candidates' for both filtered_result."

number_hits = {
    "arboel": number_hits_arboel,
    "sapbert": number_hits_sapbert,
}

############ VI) RUN THE LLM ############

system_instructions_simple = """You are a professional data annotator and curator.
Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities."""

system_instructions_moa = """You are a professional data annotator and curator.
Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities.
You will provided with the analysis of different professional annotators and you have to provide the final decision based on the analysis."""

system_instructions_recall = """You are a professional data annotator and curator.
Your task is to rank the candidate entities from best to worst for a given mention based on the provided context and the descriptions of each candidate entities."""


mention2candidates_dict = {
    "arboel": mention2arboel_candidates,
    "sapbert": mention2sapbert_candidates,
}

mention2candidates = mention2candidates_dict[EL_model]

system_instructions_dict = {
    "recall": system_instructions_recall,
    "simple": system_instructions_simple,
}

mode = "recall" if recall else "simple"
system_instructions = system_instructions_dict[mode]
repo = "recall" if recall else "accuracy"


llm = LLM(
    model=llm_model,
    tensor_parallel_size=1,
    dtype="half",
    gpu_memory_utilization=0.85,  # % of memory of the gpu that KV caching will take (allows for higher "max_model_len").
    max_logprobs=1000,
    device=device,
    max_model_len=30000,
)

tokenizer = llm.get_tokenizer()

# Path to the results
directory = f"results/{dataset_name}/{EL_model}/{repo}"
filename = f"{llm_subname}_k={k}_cands={number_candidates}_recall={recall}{recall_k}_true_results2.json"
path = os.path.join(directory, filename)

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

print("path :", path)

results = evaluate(
    llm=llm,
    tokenizer=tokenizer,
    # llm="gpt-4o-2024-08-06",
    # tokenizer=None,
    # llm="gpt-4o-mini",
    nlp_model=model,
    index=index,
    system_instructions=system_instructions,
    mentions=mentions,
    ontology=ontology2,
    corpus=corpus,
    mention2context=mention2context,
    mention2candidates=mention2candidates,
    mention2text=mention2text,
    TrainMap_context2mention=TrainMap_context2mention,
    train_mention2text=train_mention2text,
    train_mention2gold=train_mention2gold,
    k=k,
    sampling_params=sampling_params,
    reasoning=False,
    analysis_version=analysis_version,
    analysis=analysis,
    recall=recall,
    recall_k=recall_k,
)


with open(path, "w") as f:
    json.dump(results, f, indent=4)

if recall:
    recall_r, nb_cands, errors = recall_fn(
        results=results, mention2gold=mention2gold, ks=list(range(1, recall_k + 1))
    )
    print("Number of mentions:", total_mentions)
    # Number of candidates - the number of mentions where the answer is 'None' because of token limits imposed by the LLM
    print("recall :", recall_r)
    number_cands = sum(
        [number_hits[f"{EL_model}"][i] for i in range(1, number_candidates + 1)]
    )
    print(
        "Number of evaluated candidates (only one with correct CUI in the candidates) :",
        number_cands,
    )
    print(
        "Number of mentions being ignored because not correctly formatted by the LLM:",
        errors,
    )
    normalized_recall = {
        f"normalized_{k}": v * number_cands / total_mentions
        for k, v in recall_r.items()
    }
    print("Normalized_recall :", normalized_recall)
else:
    score, match, nb_cands, errors = scoring(results=results, mention2gold=mention2gold)
    print("Number of mentions:", total_mentions)
    # Number of candidates - the number of mentions where the answer is 'None' because of token limits imposed by the LLM
    print(
        "Number of evaluated candidates (only one with correct CUI in the candidates):",
        nb_cands,
    )
    print(
        "Number of mentions being ignored because not correctly formatted by the LLM:",
        errors,
    )
    print(
        "Number of correct predictions:",
        match,
    )
    print("Accuracy :", score)
    print("Normalized accuracy:", score * nb_cands / total_mentions)


end_time = time.time()
running_time = end_time - start_time
hours, rem = divmod(running_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script executed in: {int(hours)}h,{int(minutes)}mins,{seconds:.2f}s")
print(f"Results saved in: {path}")
