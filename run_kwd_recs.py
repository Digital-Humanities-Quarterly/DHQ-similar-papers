"""
This module contains scripts to find the most similar papers based on DHQ Classification
Scheme assignments (i.e., DHQ keyword).
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "CC0"
__version__ = "0.0.5"


import random
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np

from utils import KWD_TSV_PATH, get_metadata, sort_then_save, validate_metadata


def construct_binary_matrix(metadata: List[Dict]) -> np.ndarray:
    """
    Construct a binary matrix representing the presence of keywords in papers.

    Args:
        metadata: A list of dictionaries containing paper metadata.

    Returns:
        A binary matrix where rows represent papers and columns represent keywords.
    """
    dhq_keywords = set(chain(*[m["dhq_keywords"] for m in metadata]))
    dhq_keywords.discard("")
    dhq_keywords = list(dhq_keywords)

    keyword_to_index = {keyword: idx for idx, keyword in enumerate(dhq_keywords)}

    paper_ids = [paper["paper_id"] for paper in metadata]

    paper_to_row = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

    binary_matrix = np.zeros((len(paper_ids), len(dhq_keywords)), dtype=int)

    for paper in metadata:
        row_idx = paper_to_row[paper["paper_id"]]
        for keyword in paper["dhq_keywords"]:
            if keyword in keyword_to_index:
                col_idx = keyword_to_index[keyword]
                binary_matrix[row_idx, col_idx] = 1

    print(
        f"Total: {len(dhq_keywords)} unique DHQ keywords and {len(paper_ids)} "
        f"papers are under consideration."
    )
    return binary_matrix


def calculate_normalized_similarity(binary_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate a normalized similarity matrix for papers based on their binary keyword
    vectors.

    Args:
        binary_matrix: A binary matrix where rows represent papers and columns represent
    keywords.

    Returns:
        A normalized similarity matrix.
    """
    sim_raw = np.dot(binary_matrix, binary_matrix.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.nan_to_num(sim_raw / np.sum(binary_matrix, axis=1)[:, np.newaxis])

    return sim


def find_similar_papers_by_paper_id(
    query_paper_id: str,
    similarity_matrix: np.ndarray,
    paper_to_row: Dict[str, int],
    paper_ids: List[str],
    metadata: List[Dict[str, List[str]]],
    top_n: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> List[Tuple[List[str], str, float]]:
    """
    Find the most similar papers to a given paper based on a precomputed similarity
    matrix, randomly choosing among ties.
    """
    # set the seed for reproducibility
    random.seed(random_state)

    # find the index of the given paper
    row_index = paper_to_row[query_paper_id]

    # get the similarity scores for the given paper
    similarity_scores = similarity_matrix[row_index, :]
    similarity_scores[row_index] = -np.inf  # ignore self-similarity

    # find all scores that could be in the top N
    sorted_indices = np.argsort(-similarity_scores)
    sorted_scores = similarity_scores[sorted_indices]

    # find unique scores and the cutoff for top_n, accounting for ties
    n_th_index = np.where(sorted_scores >= sorted_scores[top_n - 1])[0][-1]

    # now adjust potential_indices to include all indices up to n_th_index
    potential_indices = sorted_indices[: n_th_index + 1]
    if len(potential_indices) > top_n:
        chosen_indices = random.sample(list(potential_indices), top_n)
    else:
        chosen_indices = potential_indices

    # sort chosen_indices by score to maintain descending order
    chosen_indices = sorted(chosen_indices, key=lambda x: -similarity_scores[x])

    # get the paper IDs and titles corresponding to the chosen indices
    if verbose:
        similar_papers = [
            (
                [m for m in metadata if m["paper_id"] == paper_ids[i]].pop(),
                paper_ids[i],
                similarity_scores[i],
            )
            for i in chosen_indices
        ]
    else:
        similar_papers = [paper_ids[i] for i in chosen_indices]

    # post hoc sanity check  # todo: unit tests
    if not len(similar_papers) == 10:
        raise RuntimeError(f"Less than {top_n} papers recommended.")

    return similar_papers


if __name__ == "__main__":
    print("*" * 80)
    print("Generating papers recommendations based on the DHQ Classification Scheme...")
    metadata = get_metadata()
    metadata, recs = validate_metadata(metadata)

    binary_matrix = construct_binary_matrix(metadata)
    similarity_matrix = calculate_normalized_similarity(binary_matrix)

    paper_ids = [m["paper_id"] for m in metadata]
    paper_to_row = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

    # Add recommendations iteratively
    for idx, rec in enumerate(recs):
        row_index = paper_to_row[rec["Article ID"]]
        similarity_scores = similarity_matrix[row_index, :]
        similarity_scores[row_index] = -np.inf  # ignore self-similarity

        sorted_indices = np.argsort(-similarity_scores)[:10]
        similar_papers = [paper_ids[i] for i in sorted_indices]

        for i, paper_id in enumerate(similar_papers):
            rec[f"Recommendation {i + 1}"] = paper_id

    sort_then_save(recs, KWD_TSV_PATH)

    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have\n"
        f"been successfully saved to {KWD_TSV_PATH}. {len(recs)} papers are in the\n"
        f"keyword-based recommendation."
    )
    print("*" * 80)
