"""
This module contains scripts to find the most similar papers based on DHQ Classification
Scheme assignments.
"""

__author__ = "The Digital Humanities Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.1"

import csv
import os
import random
import re
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from bs4 import BeautifulSoup

# fmt: off
# fixme: a temporary solution to account for missing year
# hard-coded publication years for papers missing this data in .xml
MISSING_YEAR = {'000573': '2021', '000580': '2021', '000646': '2022', '000664': '2023',
                '000673': '2023', '000680': '2023', '000681': '2023', '000695': '2023',
                '000709': '2023', '000710': '2023', '000714': '2023', '000716': '2023',
                '000720': '2023', '000735': '2024', '000737': '2024'}

# some known removed papers
IGNORE_PAPER_IDS = ['000424', '000432', '000488', '000492']
# fmt: on


def extract_article_folders(directory: str) -> List[str]:
    """
    Extract folders that match DHQ article folder naming pattern, excluding the example
    article '000000'.

    Args:
        directory: The directory path to scan for article folders (in the cloned DHQ
            repo).

    Returns:
        A list of paths to folders that match the DHQ article naming convention.
    """
    fld_reg = re.compile(r"^00\d{4}$")
    filtered_folders = [
        entry.path
        for entry in os.scandir(directory)
        if entry.is_dir() and re.match(fld_reg, entry.name) and entry.name != "000000"
    ]

    return filtered_folders


def remove_excessive_space(t: str) -> str:
    """
    Remove redundant space and Zero Width Space from extracted text.
    Args:
        t: an extracted field

    Returns:
        a string properly spaced
    """
    # \u200B is the unicode for Zero Width Space
    t = t.replace("\u200B", "")
    t = re.sub(r"\s+", " ", t)

    return t


def extract_relevant_elements(paper_id: str) -> Dict[str, Optional[str]]:
    """
    Extract relevant elements from a DHQ article XML file, including paper_id,
    publication_year, volume_and_issue, authors (pipe concatenated if multiple),
    affiliations (pipe concatenated if multiple), title, abstract, url, and
    dhq_keywords.

    Args:
        paper_id: A 6-digit id associated with a paper, e.g., '000622'.

    Returns:
        A dictionary containing the extracted information.
    """
    article_path = os.path.join(xml_folder, f"{paper_id}.xml")

    with open(article_path, "r", encoding="utf-8") as file:
        xml = file.read()
        soup = BeautifulSoup(xml, "xml")

    # extract title
    title = remove_excessive_space(soup.find("title").text)

    # extract publication year, volume, and issue
    publication_date = soup.find("date", {"when": True})
    publication_year = publication_date["when"][:4] if publication_date else None
    # fixme: a temporary solution to account for missing year
    if publication_year is None and paper_id in MISSING_YEAR:
        publication_year = MISSING_YEAR[str(paper_id)]
    volume = (
        soup.find("idno", {"type": "volume"}).text
        if soup.find("idno", {"type": "volume"})
        else None
    )
    # trim leading 0s
    volume = volume.lstrip("0")
    issue = (
        soup.find("idno", {"type": "issue"}).text
        if soup.find("idno", {"type": "issue"})
        else None
    )
    volume_and_issue = f"{volume}.{issue}" if volume and issue else None

    # extract authors and affiliations
    authors_tag = []
    affiliations_tag = []
    for author_info in soup.find_all("dhq:authorInfo"):
        author_name_tag = author_info.find("dhq:author_name")
        if author_name_tag:
            # extract the full name as text, including proper spacing
            full_name = " ".join(author_name_tag.stripped_strings)
            authors_tag.append(full_name)
        else:
            authors_tag.append("")
        affiliation_tag = author_info.find("dhq:affiliation")
        affiliation = affiliation_tag.get_text(strip=True) if affiliation_tag else ""
        affiliations_tag.append(affiliation)

    authors = " | ".join([remove_excessive_space(name) for name in authors_tag])
    affiliations = " | ".join([remove_excessive_space(aff) for aff in affiliations_tag])

    # extract abstract
    abstract_tag = soup.find("dhq:abstract")
    if abstract_tag:
        paragraphs = abstract_tag.find_all("p")
        abstract = " ".join(p.get_text(strip=True) for p in paragraphs)
        abstract = remove_excessive_space(abstract)
    else:
        abstract = ""

    # extract DHQ keywords
    dhq_keywords_tags = soup.find_all("term", {"corresp": True})
    dhq_keywords = (
        [term["corresp"].lower().strip() for term in dhq_keywords_tags]
        if dhq_keywords_tags
        else [""]
    )
    # heuristically construct url
    url = (
        f"https://digitalhumanities.org/dhq/vol/"
        f"{volume}/{issue}/{paper_id}/{paper_id}.html"
    )

    return {
        "paper_id": paper_id,
        "title": title,
        "publication_year": publication_year,
        "volume_and_issue": volume_and_issue,
        "authors": authors,
        "affiliations": affiliations,
        "abstract": abstract,
        "url": url,
        "dhq_keywords": dhq_keywords,
    }


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
    # 737 out of 739 article folders have a .xml, as per Feb 2024
    # 000678 and 000722 are not associated with .xml files
    xml_folders = extract_article_folders("dhq-journal/articles")

    metadata = []
    for xml_folder in xml_folders:
        paper_id = xml_folder.split("/")[-1]
        paper_path = os.path.join(xml_folder, f"{paper_id}.xml")
        if os.path.exists(paper_path):
            # two indices are wasted, as Feb 2024
            metadata.append(extract_relevant_elements(paper_id))

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
