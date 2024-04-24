"""
This module contains scripts to find the most similar papers based on DHQ Classification
Scheme assignments (i.e., DHQ keyword).
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.2"

import csv
import os
import random
from itertools import chain
from typing import Dict, List, Set, Tuple

import numpy as np

from utils import (extract_article_folders, extract_relevant_elements,
                   get_articles_in_editorial_process)


def construct_binary_matrix(metadata: List[Dict]) -> Tuple:
    """
    Construct a binary matrix representing the presence or absence of keywords in
    papers.

    Args:
        metadata: A list of dictionaries containing paper metadata.

    Returns:
        A tuple containing the binary matrix, list of paper IDs, list of DHQ
        keywords, and a mapping of paper ID to row index.
    """
    # get all dhq_keywords
    dhq_keywords: Set[str] = set(chain(*[m["dhq_keywords"] for m in metadata]))
    dhq_keywords.discard("")
    dhq_keywords = list(dhq_keywords)

    # create a mapping of keyword to column index
    keyword_to_index: Dict[str, int] = {
        keyword: idx for idx, keyword in enumerate(dhq_keywords)
    }

    # extract unique paper IDs and sort them
    paper_ids: List[str] = sorted({paper["paper_id"] for paper in metadata})

    # create a mapping of paper ID to row index
    paper_to_row: Dict[str, int] = {
        paper_id: idx for idx, paper_id in enumerate(paper_ids)
    }

    # initialize a blank matrix of zeros with the correct shape
    binary_matrix: np.ndarray = np.zeros((len(paper_ids), len(dhq_keywords)), dtype=int)

    # fill the matrix
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
    return binary_matrix, paper_ids, dhq_keywords, paper_to_row


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
    print("Generating similar papers based on the DHQ Classification Scheme...")
    # 730 articles are included in the recommendation system as per Apr 2024
    xml_folders = extract_article_folders("dhq-journal/articles")

    # remove articles in editorial process (should not be considered in recommendation)
    xml_to_remove = [
        os.path.join("dhq-journal/articles", f)
        for f in get_articles_in_editorial_process()
    ]
    xml_folders = [f for f in xml_folders if f not in xml_to_remove]

    metadata = []
    for xml_folder in xml_folders:
        paper_id = xml_folder.split("/").pop()
        paper_path = os.path.join(xml_folder, f"{paper_id}.xml")
        if os.path.exists(paper_path):
            metadata.append(extract_relevant_elements(xml_folder))

    binary_matrix, paper_ids, keywords, paper_to_row = construct_binary_matrix(metadata)
    similarity_matrix = calculate_normalized_similarity(binary_matrix)

    # uncomment below and check if output is reasonable
    # print(find_similar_papers_by_paper_id('000448',
    #                                       similarity_matrix,
    #                                       paper_to_row,
    #                                       paper_ids,
    #                                       metadata,
    #                                       verbose=True))

    # computing results
    recommends = []
    for m in metadata:
        # pick up the naming used in an early repo
        recommend = {
            "Article ID": m["paper_id"],
            "Pub. Year": m["publication_year"],
            "Volume and Issue": m["volume_and_issue"],
            "Authors": m["authors"],
            "Affiliations": m["affiliations"],
            "Title": m["title"],
            "Abstract": m["abstract"],
        }
        raw_recommend = find_similar_papers_by_paper_id(
            m["paper_id"], similarity_matrix, paper_to_row, paper_ids, metadata
        )
        for i, r in enumerate(raw_recommend):
            recommend[f"Recommendation {i + 1}"] = r

        # Add the URL at the end
        recommend["url"] = m["url"]
        recommends.append(recommend)
    # sort list based on 'Article ID'
    recommends = sorted(recommends, key=lambda x: x["Article ID"])

    # output
    header = list(recommends[0].keys())
    # move 'url' to the end to follow naming conventions is a previous repo
    header.append(header.pop(header.index("url")))

    tsv_path = "dhq-recs-zfill.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in recommends:
            writer.writerow(row)
    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have\n"
        f"been successfully saved to {tsv_path}."
    )
    print("*" * 80)
