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
    temperature=0, top_p=0.9, max_tokens=1000, stop=["<|eot_id|>"]
)

set_seed(12)

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

model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
model.to(device_2)
# Generate embeddings for the corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
corpus_embeddings = corpus_embeddings.cpu().detach().numpy()

# Assuming corpus_embeddings is already a NumPy array
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
filtered_results = filtered_results[filtered_results["biencoder_hit_index"] < 20]

biencoder_res = 0
for idx, row in filtered_results.iterrows():
    if row["biencoder_hit_index"] == 0:
        biencoder_res += 1

print("biencoder results :", biencoder_res / len(filtered_results))

crossencoder_res = 0
for idx, row in filtered_results.iterrows():
    if row["crossencoder_hit_index"] == 0:
        crossencoder_res += 1

print("crossencoder results :", crossencoder_res / len(filtered_results))


train_mentions = []
train_mention2context = {}
train_mention2gold = {}
train_mention2text = {}
for idx, row in train_df.iterrows():
    train_mention2gold[row["mention_id"]] = row["db_ids"]
    train_mentions.append(row["mention_id"])
    train_mention2text[row["mention_id"]] = row["deabbreviated_text"]
    train_mention2context[row["mention_id"]] = row["limited_contextualized_mention"]

number_candidates = 20

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
    # print(idx)
    # Only consider row if hit_index < max number of candidates
    if row["biencoder_hit_index"] < number_candidates:
        mention2biencoder_candidates[row["mention_id"]] = [
            el[0] for el in row["biencoder_candidates"][:number_candidates]
        ]
        # mention2crossencoder_candidates[row['mention_id']] = [el[0] for el in row['crossencoder_candidates'][:number_candidates]]
        mention2gold[row["mention_id"]] = row["db_ids"]
        mentions.append(row["mention_id"])
        mention2text[row["mention_id"]] = row["deabbreviated_text"]
        mention2hit[row["mention_id"]] = row["biencoder_hit_index"]

system_instructions = """You are a professional data annotator and curator.
Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities."""


llm = LLM(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    tensor_parallel_size=1,
    dtype="half",
    gpu_memory_utilization=0.85,
    max_logprobs=1000,
    device=device_2,
    max_model_len=20000,
)

tokenizer = llm.get_tokenizer()

results = evaluate_vllm(
    llm=llm,
    nlp_model=model,
    tokenizer=tokenizer,
    index=index,
    system_instructions=system_instructions,
    mentions=mentions,
    ontology=ontology2,
    corpus=corpus,
    mention2context=mention2context,
    mention2biencoder_candidates=mention2biencoder_candidates,
    mention2text=mention2text,
    TrainMap_context2mention=TrainMap_context2mention,
    train_mention2text=train_mention2text,
    train_mention2gold=train_mention2gold,
    k=10,
    sampling_params=sampling_params,
)

score = scoring(results=results, mention2gold=mention2gold)

print("score :", score)

with open("Mistral-Nemo-Instruct-2407_k=10_reasoning_results.json", "w") as f:
    json.dump(results, f, indent=4)
