import pickle
import ujson
import json
import sys
import os

import pandas as pd
import numpy as np
import torch
import random

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
from peft import PeftModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import random

load_dotenv(override=True)

import openai
import json
import re
import logging
from collections import Counter, defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device_1 = torch.device("cuda:0")  # First GPU (GPU 0)
device_2 = torch.device("cuda:1")  # Second GPU (GPU 1)


llm_api_test = tg.get_engine(
    engine_name="vllm-meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype="half",
    enforce_eager=True,
    device=device_1,
    gpu_memory_utilization=0.8,
)


llm_api_eval = tg.get_engine(
    engine_name="mistralai/Mistral-7B-Instruct-v0.3",
    dtype="half",
    enforce_eager=True,
    device=device_2,
    gpu_memory_utilization=0.8,
)
