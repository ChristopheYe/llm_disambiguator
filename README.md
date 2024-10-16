# LLM as a Reasoning Entity Disambiguator.

## Installation

In order to run this, please install the following conda environment : 

```bash
conda env create -f llm_disambiguator_env.yaml
conda activate llm_disambiguator
```

## Example Usage: Run Inference and Evaluation on NCBI-Disease
```
python llm_disambiguator_main.py --dataset_name ncbi_disease --EL_model arboel --llm_model "Qwen/Qwen2.5-7B-Instruct" --llm_subname "Qwen2.5-7B-Instruct"
```

## Candidates

"Candidates for the NCBI-Dataset from SapBERT and ArboEL are available in `src/candidates`, and the corresponding MEDIC ontology is located in `src/ontology`."
