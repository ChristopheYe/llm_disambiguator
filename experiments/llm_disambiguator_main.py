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
import gzip
import argparse

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from ids import open_ai_api_key
from utils_functions import *
from collections import defaultdict
import openai
import re
import ujson
import logging
from collections import Counter, defaultdict

openai.api_key = open_ai_api_key

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    """
    Parse the input arguments
    """
    print("Starting argument parsing...")
    parser = argparse.ArgumentParser(description="LLM Disambiguator")
    # Required
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ncbi_disease",
        help="Name of the evaluated dataset",
    )
    parser.add_argument(
        "--EL_model",
        type=str,
        default="sapbert",
        help="Name of the entity linking model",
    )

    parser.add_argument(
        "--recall",
        action="store_true",
        help="Set True if recall is needed, else accuracy",
    )

    parser.add_argument(
        "--recall_k",
        type=int,
        default=5,
        help="Number of candidates to rerank for recall",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Name of the LLM model to use for entity disambiguation",
    )

    parser.add_argument(
        "--llm_subname",
        type=str,
        default="Qwen2.5-7B-Instruct",
        help="Used for the name of the results file",
    )

    parser.add_argument(
        "--number_candidates",
        type=int,
        default=20,
        help="Number of candidates in the prompt",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of examples to include in the prompt",
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Default to GPU if available
        help="Device to run the model on: 'cpu', 'cuda', or 'cuda:0', etc.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="seed for reproducibility",
    )

    parser.add_argument(
        "--analysis",
        type=str,
        default=None,
        help="With or without analysis for Mixture of agent reasoning",
    )

    parser.add_argument(
        "--analysis_version",
        type=str,
        default="v1",
        help="v1 for default, v2 for MoA",
    )

    return parser.parse_args()


def main(config):
    print("Parsed Arguments:", config)
    # Start time
    start_time = time.time()
    # ----------------- START ARGUMENTS ----------------- #
    dataset_name = config.dataset_name

    EL_model = config.EL_model

    number_candidates = config.number_candidates
    print("number of candidates to consider :", number_candidates)

    device = config.device

    analysis = config.analysis
    analysis_version = config.analysis_version  # v1 for default, v2 for MoA
    recall = config.recall
    recall_k = config.recall_k
    k = config.k

    llm_model = config.llm_model
    llm_subname = config.llm_subname

    sampling_params = SamplingParams(
        temperature=0, top_p=0.9, max_tokens=5000, stop=["<|eot_id|>"]
    )
    set_seed(config.seed)

    # ----------------- END ARGUMENTS ----------------- #

    ############ I) LOAD THE DATA (ONTOLOGY + MENTIONS) ############

    with gzip.open("ontology/medic_ontology.pkl.gz", "rb") as f:
        ontology2 = pickle.load(f)

    # Load the df from a CSV file
    dataset_df = pd.read_csv(
        "datasets/ncbi_disease.csv", dtype={"document_id": str, "mention_id": str}
    )
    dataset_df["offsets"] = dataset_df["offsets"].apply(ast.literal_eval)
    dataset_df["db_ids"] = dataset_df["db_ids"].apply(ast.literal_eval)
    dataset_df["type"] = dataset_df["type"].apply(ast.literal_eval)
    test_df = dataset_df[dataset_df["split"] == "test"]
    train_df = dataset_df[dataset_df["split"] == "train"]
    corpus = [
        train_df.iloc[i]["limited_contextualized_mention"] for i in range(len(train_df))
    ]

    TrainMap_mention2context = {}
    for idx, row in train_df.iterrows():
        TrainMap_mention2context[row["mention_id"]] = row[
            "limited_contextualized_mention"
        ]
    TrainMap_context2mention = {v: k for k, v in TrainMap_mention2context.items()}

    # Load the DataFrame from CSV, ensuring 'document_id' and 'mention_id' are strings
    filtered_results = pd.read_csv(
        "datasets/ncbi_disease_final_sapbert.csv",
        dtype={"document_id": str, "mention_id": str},
    )
    # Convert the necessary columns back to lists
    filtered_results = pd.read_csv(
        "datasets/ncbi_disease_final_sapbert.csv",
        dtype={"document_id": str, "mention_id": str},
    )
    filtered_results["sapbert_candidates"] = filtered_results[
        "sapbert_candidates"
    ].apply(json.loads)
    filtered_results["offsets"] = filtered_results["offsets"].apply(ast.literal_eval)
    filtered_results["db_ids"] = filtered_results["db_ids"].apply(ast.literal_eval)
    filtered_results = filtered_results[
        filtered_results[f"{EL_model}_hit_index"] < config.number_candidates
    ]

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

    total_mentions = len(test_df)
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
        mention2context[(str(row["mention_id"]))] = row[
            "limited_contextualized_mention"
        ]

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

    print("mentions :", mentions)

    ############ V) COMPUTE THE RESULTS FROM THE ORIGINAL ENTITY-LINKING MODELS ############

    number_hits_arboel = 0
    number_hits_sapbert = 0
    print("length mention :", len(mentions))
    print("length filtered results :", len(filtered_results))
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
    ), f"Length mismatch: mentions has {len(mentions)} elements, but it should have {len(filtered_results)} elements. Check the condition 'row['sapbert/arboel_hit_index'] < number_candidates', the number_candidates is probably too low"

    number_hits = {
        "arboel": number_hits_arboel,
        "sapbert": number_hits_sapbert,
    }

    ############ VI) RUN THE LLM ############

    system_instructions_simple = """You are a professional data annotator and curator.
    Your task is to identify the correct entity for a given mention based on the provided context and the descriptions of {number_candidates} candidate entities."""

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

    if "gpt" in llm_model:
        llm = llm_model
        tokenizer = None
    else:
        llm = LLM(
            model=llm_model,
            tensor_parallel_size=1,
            dtype="half",
            gpu_memory_utilization=0.75,  # % of memory of the gpu that KV caching will take (allows for higher "max_model_len").
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
        score, match, nb_cands, errors = scoring(
            results=results, mention2gold=mention2gold
        )
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


if __name__ == "__main__":
    print("Script started...")
    config = parse_args()
    main(config)
