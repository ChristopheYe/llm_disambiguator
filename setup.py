from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))


setup(
    name="llm_disambiguator",
    description="LLM for named entity disambiguation",
    # long_description=long_description,
    url="",
    author="Anonymous",
    author_email="",
    keywords=[
        "biomedical",
        "entity-linking",
        "biomedical-entity-linking",
        "named entity disambiguation",
    ],
    packages=find_packages(),
    python_requires=">= 3.9",
    install_requires=[
        "tqdm",
        "pandas",
        "numpy",
        "matplotlib",
        "ujson",
        "torch",
        "transformers",
        "faiss-gpu",
        "sentence_transformers",
        "vllm",
        "openai",
        "statsmodels",
    ],
)
