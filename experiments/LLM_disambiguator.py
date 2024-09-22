import pickle
import ujson
import json
import sys
import os
import time

from collections import defaultdict

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

from peft import PeftModel
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
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_1 = torch.device("cuda:0")  # First GPU (GPU 0)
device_2 = torch.device("cuda:1")  # Second GPU (GPU 1)
sampling_params = SamplingParams(
    temperature=0, top_p=0.9, max_tokens=5000, stop=["<|eot_id|>"]
)

set_seed(12)

# Start time
start_time = time.time()

### 1) Load the data (ontology + mentions)
ontology_dir = "/mitchell/entity-linking/kbs/medic.tsv"
name = "medic"
ontology2 = BiomedicalOntology.load_medic(filepath=ontology_dir, name=name)

dataset_name = "ncbi_disease"
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


### 2) Create the RAG-like method for better ICL in the prompt
model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
model.to(device_2)
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

dataset_names = ["ncbi_disease"]
model_names = ["arboel_biencoder", "arboel_crossencoder"]
path_to_result = {
    "ncbi_disease": {
        "arboel_biencoder": "/home2/cye73/results2/arboel/ncbi_disease/biencoder_output_eval.json",
        "arboel_crossencoder": "/home2/cye73/results2/arboel/ncbi_disease/crossencoder_output_eval.json",
    }
}

abbreviations_path = "/home2/cye73/data_test/abbreviations.json"

evaluator = Evaluate(dataset_names, model_names, path_to_result, abbreviations_path)
evaluator.load_results()
evaluator.process_datasets()
evaluator.evaluate(eval_strategies=["basic"])

results = evaluator.full_results["basic"]["ncbi_disease"]

cols = [
    "document_id",
    "offsets",
    "deabbreviated_text",
    "db_ids",
    "mention_id",
    "joined_offsets",
    "arboel_biencoder_resolve_abbrev",
    "arboel_biencoder_resolve_abbrev_min_hit_index",
    "arboel_crossencoder_resolve_abbrev",
    "arboel_crossencoder_resolve_abbrev_min_hit_index",
]
filtered_results = results[cols].rename(
    columns={
        "arboel_biencoder_resolve_abbrev": "biencoder_candidates",
        "arboel_crossencoder_resolve_abbrev": "crossencoder_candidates",
        "arboel_biencoder_resolve_abbrev_min_hit_index": "biencoder_hit_index",
        "arboel_crossencoder_resolve_abbrev_min_hit_index": "crossencoder_hit_index",
    }
)

### 3) Compute the results from the original entity-linking models
number_candidates = 20
filtered_results = filtered_results[
    filtered_results["crossencoder_hit_index"] < number_candidates
]

number_hits_biencoder = number_hit(
    filtered_results, "biencoder_hit_index", number_candidates
)
number_hits_crossencoder = number_hit(
    filtered_results, "crossencoder_hit_index", number_candidates
)

total_mentions = len(results)
total_mentions_with_hit_index = len(filtered_results)

print("total mentions :", total_mentions)
print(
    "total mentions with hit index = number of mentions to evaluate:",
    total_mentions_with_hit_index,
)
print("number hits biencoder :", number_hits_biencoder)
print("number hits crossencoder :", number_hits_crossencoder)

biencoder_results = compute_recall(
    filtered_results, "biencoder_hit_index", 5, total_mentions
)
print("Biencoder:")
for i, (unnormalized, normalized) in enumerate(biencoder_results):
    print(
        f"recall {i+1}: Normalized = {normalized:.4f}, Unnormalized = {unnormalized:.4f}"
    )

crossencoder_results = compute_recall(
    filtered_results, "crossencoder_hit_index", 5, total_mentions
)
print("Crossencoder:")
for i, (unnormalized, normalized) in enumerate(crossencoder_results):
    print(
        f"recall {i+1}: Normalized = {normalized:.4f}, Unnormalized = {unnormalized:.4f}"
    )


### 4) Data preprocessing for the prompt used in the LLM
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
mention2biencoder_candidates = {}
mention2crossencoder_candidates = {}
mention2gold = {}
mention2hit = {}
mention2text = {}
for idx, row in filtered_results.iterrows():
    # Only consider row if hit_index < max number of candidates
    if row["crossencoder_hit_index"] < number_candidates:
        mention2biencoder_candidates[row["mention_id"]] = [
            el[0] for el in row["biencoder_candidates"][:number_candidates]
        ]
        mention2gold[row["mention_id"]] = row["db_ids"]
        mentions.append(row["mention_id"])
        mention2text[row["mention_id"]] = row["deabbreviated_text"]
        mention2hit[row["mention_id"]] = row["biencoder_hit_index"]

assert len(mentions) == len(
    filtered_results
), f"Length mismatch: mentions has {len(train_mentions)} elements, but it should have {len(filtered_results)} elements. Check the condition 'row['biencoder/crossencoder_hit_index'] < number_candidates' for both filtered_result."

### 5) Run the LLM
system_instructions = """You are a professional data annotator and curator.
Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities."""

system_instructions_moa = """You are a professional data annotator and curator.
Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities.
You will provided with the analysis of different professional annotators and you have to provide the final decision based on the analysis."""

system_instructions_recall = """You are a professional data annotator and curator.
Your task is to rank the candidate entities from best to worst for a given mention based on the provided context and the descriptions of each candidate entities."""


with open(
    "data/biencoder/default2/Meta-Llama-3.1-8B-Instruct_k=3_results.json", "r"
) as f:
    analysis1 = json.load(f)

with open(
    "data/biencoder/default2/Mistral-7B-Instruct-v0.3_k=3_results.json", "r"
) as f:
    analysis2 = json.load(f)

with open(
    "data/biencoder/default2/Mistral-Nemo-Instruct-2407_k=3_results.json", "r"
) as f:
    analysis3 = json.load(f)

with open("data/biencoder/default2/Qwen2.5-7B-Instruct_k=3_results.json", "r") as f:
    analysis4 = json.load(f)

with open("data/biencoder/default2/Qwen2.5-14B-Instruct_k=3_results.json", "r") as f:
    analysis5 = json.load(f)

#######################################################################################

# with open(
#     "data/biencoder/reasoning2/Meta-Llama-3.1-8B-Instruct_k=3_reasoning_results.json", "r"
# ) as f:
#     analysis1 = json.load(f)

# with open(
#     "data/biencoder/reasoning2/Mistral-7B-Instruct-v0.3_k=3_reasoning_results.json", "r"
# ) as f:
#     analysis2 = json.load(f)

# with open(
#     "data/biencoder/reasoning2/Mistral-Nemo-Instruct-2407_k=3_reasoning_results.json", "r"
# ) as f:
#     analysis3 = json.load(f)

# with open("data/biencoder/reasoning2/Qwen2.5-7B-Instruct_k=3_reasoning_results.json", "r") as f:
#     analysis4 = json.load(f)

# with open("data/biencoder/reasoning2/Qwen2.5-14B-Instruct_k=3_reasoning_results.json", "r") as f:
#     analysis5 = json.load(f)

analysis = [analysis1, analysis2, analysis3, analysis4, analysis5]
analysis = None
analysis_version = "v1"  # v1 for default, v2 for MoA
recall = True
recall_k = 5

llm = LLM(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    tensor_parallel_size=1,
    dtype="half",
    gpu_memory_utilization=0.9,  # % of memory of the gpu that KV caching will take (allows for higher "max_model_len").
    max_logprobs=1000,
    device=device_2,
    max_model_len=30000,
)

tokenizer = llm.get_tokenizer()

results = evaluate_vllm(
    llm=llm,
    nlp_model=model,
    tokenizer=tokenizer,
    index=index,
    system_instructions=system_instructions_recall,
    mentions=mentions,
    ontology=ontology2,
    corpus=corpus,
    mention2context=mention2context,
    mention2biencoder_candidates=mention2biencoder_candidates,
    mention2text=mention2text,
    TrainMap_context2mention=TrainMap_context2mention,
    train_mention2text=train_mention2text,
    train_mention2gold=train_mention2gold,
    k=3,
    sampling_params=sampling_params,
    reasoning=False,
    analysis_version=analysis_version,
    analysis=analysis,
    recall=recall,
    recall_k=recall_k,
)

with open(
    "data/crossencoder/recall/Mistral-Nemo-Instruct-2407_k=3_results_test1.json", "w"
) as f:
    json.dump(results, f, indent=4)

if recall:
    recall_r = recall_fn(
        results=results, mention2gold=mention2gold, ks=list(range(1, recall_k + 1))
    )
    print("recall :", recall_r)
else:
    score = scoring(results=results, mention2gold=mention2gold)
    print("score :", score)


end_time = time.time()
running_time = end_time - start_time
hours, rem = divmod(running_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script executed in: {int(hours)}h,{int(minutes)}mins,{seconds:.2f}s")
