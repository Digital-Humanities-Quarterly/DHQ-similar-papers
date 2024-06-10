"""
This module provides scripts to generate SPECTER2 embeddings for the DHQ journal, using
paper titles and abstracts (i.e., a "proximity" task natively supported by SPECTER2).

The implementation is heavily inspired by:
- https://github.com/bcglee/DHQ-similar-papers
- https://huggingface.co/allenai/specter2
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.4"

import csv
import math
import os
from time import time
from typing import List

import numpy as np
import torch
from adapters import AutoAdapterModel
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import (extract_article_folders, extract_relevant_elements,
                   get_articles_in_editorial_process,
                   NO_RECOMMEDATIONS)

MODEL = 'allenai/specter2_base'
BATCH_SIZE = 4
tsv_path = "dhq-recs-zfill-spctr.tsv"
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


def find_most_similar_papers(metadata, vecs, top_n=10):
    """
    Find the top N most similar papers for each paper using AnnoyIndex.

    Args:
        metadata: List of dictionaries containing paper metadata.
        vecs: List of vectors in the same order of as metadata (sorted by paper_id).
        top_n: Number of top similar papers to find.

    Returns:
        A list of dictionaries containing paper recommendations.
    """
    pairwise_cos_dists = pdist(vecs, "cosine")
    cos_sim = 1 - squareform(pairwise_cos_dists)

    def find_most_similar(index, similarity_matrix, top_n=top_n):
        similar_indices = np.argsort(similarity_matrix[index])[-top_n - 1:-1][
                          ::-1]  # exclude the first one as it is the abstract itself
        return similar_indices

    recommends = []
    for i, m in enumerate(metadata):
        recommend = {
            "Article ID": m["paper_id"],
            "Pub. Year": m["publication_year"],
            "Authors": m["authors"],
            "Affiliations": m["affiliations"],
            "Title": m["title"],
        }

        # find top 10 similar papers (excluding self)
        similar_papers = find_most_similar(i, cos_sim)
        recommend.update(
            {
                f"Recommendation {j + 1}": metadata[idx]["paper_id"]
                for j, idx in enumerate(similar_papers)
            }
        )
        recommend["url"] = m["url"]
        recommends.append(recommend)

    return recommends


if __name__ == "__main__":
    print("*" * 80)
    print(
        f"Generating paper recommendations based on {MODEL} using {device}..."
    )
    start = time()
    # get all xml files
    xml_folders = extract_article_folders("dhq-journal/articles")

    # remove articles in editorial process (should not be considered in recommendation)
    xml_to_remove = [
        os.path.join("dhq-journal/articles", f)
        for f in get_articles_in_editorial_process()
    ]
    xml_to_remove.extend(NO_RECOMMEDATIONS)
    xml_folders = [f for f in xml_folders if f not in xml_to_remove]

    metadata = []
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

    # sort metadata before embedding computation
    metadata = sorted(metadata, key=lambda x: x["paper_id"])

    # generate embeddings using recommended method
    # https://huggingface.co/allenai/specter2
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoAdapterModel.from_pretrained(MODEL)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2",
                       set_active=True)
    model.to(device)

    # combine title and abstract separated with a sep_token for SPECTER input
    title_abstracts = [
        m.get("title", "") + tokenizer.sep_token + m.get("abstract", "")
        for m in metadata
    ]

    vecs = generate_embeddings(title_abstracts, tokenizer, model)

    # find most similar papers for each paper
    recommends = find_most_similar_papers(metadata, vecs)

    # output recommendations
    header = list(recommends[0].keys())
    header.append(header.pop(header.index("url")))
    with open(tsv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in recommends:
            writer.writerow(row)

    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have "
        f"been successfully saved to {tsv_path}. {len(recommends)} papers are in the "
        f"recommendation list. This used {round(time() - start)} seconds."
    )
    print("*" * 80)
