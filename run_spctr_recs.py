"""
This module provides scripts to generate SPECTER2 embeddings for the DHQ journal, using
paper titles and abstracts (i.e., a "proximity" task natively supported by SPECTER2).
It identifies the most similar papers based on these embeddings.

The implementation is heavily inspired by:
- https://github.com/bcglee/DHQ-similar-papers
- https://huggingface.co/allenai/specter2
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.3"

import csv
import json
import os
from typing import List

import math

import numpy as np
import torch
from annoy import AnnoyIndex
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import (extract_article_folders, extract_relevant_elements,
                   get_articles_in_editorial_process)

D_MODEL = 768
ANNOY_INDEX_PATH = "specter2.ann"
EMBEDDINGS_PATH = "specter2_embeddings.npy"
PAPER_ID_LOOKUP_PATH = "specter2_paper_ids.json"
BATCH_SIZE = 4


def load_embeddings_and_index():
    """
    Load existing embeddings and AnnoyIndex from disk.
    """
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(
            ANNOY_INDEX_PATH) and os.path.exists(PAPER_ID_LOOKUP_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
        ann_index = AnnoyIndex(D_MODEL, 'angular')
        ann_index.load(ANNOY_INDEX_PATH)
        with open(PAPER_ID_LOOKUP_PATH, 'r') as f:
            paper_id_lookup = json.load(f)
    else:
        embeddings = np.empty((0, D_MODEL))
        ann_index = AnnoyIndex(D_MODEL, 'angular')
        paper_id_lookup = {}

    return embeddings, ann_index, paper_id_lookup


def save_embeddings_and_index(embeddings, ann_index, paper_id_lookup):
    """
    Save embeddings and AnnoyIndex to disk.
    """
    np.save(EMBEDDINGS_PATH, embeddings)
    ann_index.save(ANNOY_INDEX_PATH)
    with open(PAPER_ID_LOOKUP_PATH, 'w') as f:
        json.dump(paper_id_lookup, f)


def generate_specter_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate SPECTER embeddings for a list of texts.

    Args:
        texts: List of title and abstract seperated with a sep_token.

    Returns:
        A numpy array of embeddings.
    """

    def chunk(file_list, n_chunks):
        chunk_size = math.ceil(float(len(file_list)) / n_chunks)
        return [file_list[i * chunk_size:(i + 1) * chunk_size] for i in
                range(n_chunks - 1)] + [file_list[(n_chunks - 1) * chunk_size:]]

    batches = chunk(texts, math.ceil(len(texts) / 4))

    embeddings_batches = []
    for batch in tqdm(batches):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt",
                           max_length=300)
        result = model(**inputs)
        embeddings_batches.append(result.last_hidden_state[:, 0, :])

    embeddings = torch.cat(embeddings_batches, 0)
    return embeddings.detach().numpy()


def find_top_similar_papers(ann_index, embeddings, metadata, top_n=10):
    """
    Find the top N most similar papers for each paper using AnnoyIndex.

    Args:
        ann_index: AnnoyIndex object containing the embeddings.
        embeddings: Numpy array of embeddings.
        metadata: List of dictionaries containing paper metadata.
        top_n: Number of top similar papers to find.

    Returns:
        A list of dictionaries containing paper recommendations.
    """
    recommends = []
    for i, m in enumerate(metadata):
        recommend = {
            "Article ID": m["paper_id"],
            "Pub. Year": m["publication_year"],
            "Authors": m["authors"],
            "Affiliations": m["affiliations"],
            "Title": m["title"],
        }

        # find top N similar papers (excluding self)
        similar_papers = ann_index.get_nns_by_vector(embeddings[i], top_n + 1)[1:]
        recommend.update(
            {f"Recommendation {j + 1}": metadata[idx]["paper_id"] for j, idx in
             enumerate(similar_papers)})
        recommend["url"] = m["url"]
        recommends.append(recommend)

    return recommends


if __name__ == "__main__":
    print("*" * 80)
    print("Generating paper recommendations based on SPECTER2 (base)...")

    # get all xml files
    xml_folders = extract_article_folders("dhq-journal/articles")

    # remove articles in editorial process (should not be considered in recommendation)
    xml_to_remove = [
        os.path.join("dhq-journal/articles", f)
        for f in get_articles_in_editorial_process()
    ]
    xml_folders = [f for f in xml_folders if f not in xml_to_remove]

    # load existing embeddings and index
    embeddings, ann_index, paper_id_lookup = load_embeddings_and_index()

    metadata = []
    new_papers = []
    for xml_folder in xml_folders:
        paper_id = xml_folder.split("/").pop()
        paper_path = os.path.join(xml_folder, f"{paper_id}.xml")
        if os.path.exists(paper_path):
            m = extract_relevant_elements(xml_folder)
            has_zero_length_value = False
            for key, value in m.items():
                if value == "":
                    print(
                        f"{m['paper_id']}'s {key} is missing. "
                        f"Will not be included in the recommendations."
                    )
                    has_zero_length_value = True
            if not has_zero_length_value:
                metadata.append(m)
                if paper_id not in paper_id_lookup:
                    new_papers.append(m)

    # sort metadata before embedding computation
    metadata = sorted(metadata, key=lambda x: x["paper_id"])

    if new_papers:
        # initialize SPECTER model and tokenizer only if there are new papers
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        model = AutoModel.from_pretrained('allenai/specter2_base')

        # combine title and abstract separated with a sep_token for SPECTER input
        new_title_abstracts = [
            m.get('title', '') + tokenizer.sep_token + m.get('abstract', '') for m in
            new_papers]

        # generate embeddings for new papers
        new_embeddings = generate_specter_embeddings(new_title_abstracts)
        new_paper_id_lookup = {m["paper_id"]: i + len(paper_id_lookup) for i, m in
                               enumerate(new_papers)}

        # update embeddings and index
        embeddings = np.vstack((embeddings, new_embeddings))
        for i, embedding in enumerate(new_embeddings):
            ann_index.add_item(len(paper_id_lookup) + i, embedding)

        paper_id_lookup.update(new_paper_id_lookup)
        ann_index.build(10)
        save_embeddings_and_index(embeddings, ann_index, paper_id_lookup)

    # find top similar papers for each paper
    recommends = find_top_similar_papers(ann_index, embeddings, metadata)

    # output recommendations
    header = list(recommends[0].keys())
    header.append(header.pop(header.index("url")))

    tsv_path = "dhq-recommendations-specter2.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in recommends:
            writer.writerow(row)

    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have been"
        f"successfully saved to {tsv_path}. {len(recommends)} papers are in the "
        f"BM25-based recommendation using title, abstract, and body text.")
    print("*" * 80)

