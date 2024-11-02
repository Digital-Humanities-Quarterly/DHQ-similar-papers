"""
This module provides scripts to generate SPECTER2 embeddings for the DHQ journal, using
paper titles and abstracts (i.e., a "proximity" task natively supported by SPECTER2).

The implementation is heavily inspired by:
- https://github.com/bcglee/DHQ-similar-papers
- https://huggingface.co/allenai/specter2
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "CC0"
__version__ = "0.0.5"


import math
from time import time
from typing import Dict, List

import numpy as np
import torch
from adapters import AutoAdapterModel
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import (SPCTR_TSV_PATH, get_metadata, sort_then_save,
                   validate_metadata)

MODEL = "allenai/specter2_base"
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings(texts: List[str], tokenizer, model) -> np.ndarray:
    """
    Generate SPECTER embeddings for a list of texts.

    Args:
        texts: List of title and abstract separated with a sep_token.
        tokenizer: The tokenizer instance to use for SPECTER.
        model: The model instance to use for SPECTER.

    Returns:
        A numpy array of embeddings.
    """

    def chunk(file_list, n_chunks):
        chunk_size = math.ceil(float(len(file_list)) / n_chunks)
        return [
            file_list[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_chunks - 1)
        ] + [file_list[(n_chunks - 1) * chunk_size :]]

    batches = chunk(texts, math.ceil(len(texts) / BATCH_SIZE))

    embeddings_batches = []
    for batch in tqdm(batches):
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=300
        ).to(device)
        with torch.no_grad():
            result = model(**inputs)
        embeddings_batches.append(result.last_hidden_state[:, 0, :].cpu())

    embeddings = torch.cat(embeddings_batches, 0)
    return embeddings.detach().numpy()


def find_most_similar_papers(
    recs: List[Dict], vecs: np.ndarray, top_n: int = 10
) -> List[Dict]:
    """
    Find the top N most similar papers for each paper using cosine similarity.

    Args:
        recs: List of dictionaries containing paper recommendations with prefilled fields.
        vecs: List of vectors in the same order as recs (sorted by paper_id).
        top_n: Number of top similar papers to find.

    Returns:
        A list of dictionaries containing paper recommendations.
    """
    pairwise_cos_dists = pdist(vecs, "cosine")
    cos_sim = 1 - squareform(pairwise_cos_dists)

    for i, rec in enumerate(recs):
        similarity_scores = cos_sim[i]
        similarity_scores[i] = -np.inf  # ignore self-similarity

        sorted_indices = np.argsort(-similarity_scores)[:top_n]
        similar_papers = [recs[idx]["Article ID"] for idx in sorted_indices]

        for j, paper_id in enumerate(similar_papers):
            rec[f"Recommendation {j + 1}"] = paper_id

    return recs


if __name__ == "__main__":
    print("*" * 80)
    print(f"Generating paper recommendations based on {MODEL} using {device}...")
    start = time()
    metadata = get_metadata()
    metadata, recs = validate_metadata(metadata)

    # generate embeddings using SPECTER2
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoAdapterModel.from_pretrained(MODEL)
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )
    model.to(device)

    # combine title and abstract separated with a sep_token for SPECTER input
    title_abstracts = [
        m.get("title", "") + tokenizer.sep_token + m.get("abstract", "")
        for m in metadata
    ]

    vecs = generate_embeddings(title_abstracts, tokenizer, model)

    # find most similar papers for each paper
    recs = find_most_similar_papers(recs, vecs)

    # output recommendations
    sort_then_save(recs, SPCTR_TSV_PATH)

    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have\n"
        f"been successfully saved to {SPCTR_TSV_PATH}. {len(recs)} papers are in the\n"
        f"recommendation list. This used {round(time() - start)} seconds."
    )
    print("*" * 80)
