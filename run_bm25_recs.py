"""
This module contains scripts to find the most similar papers based on the full text
(title, abstract, and body text) using BM25 similarity.
"""

__author__ = "The Digital Humanities Quarterly Data Analytics Team"
__license__ = "MIT"
__version__ = "0.0.5"


from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import BM25_TSV_PATH, get_metadata, sort_then_save, validate_metadata


def compute_bm25_scores(
    documents: List[str], queries: List[str], k1: float = 1.5, b: float = 0.75
) -> sp.csr_matrix:
    """
    Compute the BM25 scores for a set of queries against a set of documents.

    This function configures a TfidfVectorizer without using sublinear tf scaling to
    approximate BM25's term frequency (TF) component. It then applies the BM25
    modifications to the TF component and adjusts for document frequency and document
    length normalization.

    Args:
        documents: A list of document strings.
        queries: A list of query strings, each representing a single query.
        k1: The BM25 k1 parameter, controlling the scaling of the term frequency
            (default 1.5).
        b: The BM25 b parameter, which controls the degree of document length
            normalization (default 0.75).

    Returns:
        A sparse matrix of shape (len(queries), len(documents)) where each
        element [i, j] is the BM25 score of the i-th query against the j-th
        document.
    """
    # tokenizer adopted from
    # https://github.com/karpathy/arxiv-sanity-lite/blob/master/compute.py
    v = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
        ngram_range=(1, 2),
        max_features=2000,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        max_df=0.1,
        min_df=5,
    )
    X = v.fit_transform(documents)  # shape: (documents, terms)
    Q = v.transform(queries)  # shape: (queries, terms)

    idf = v.idf_ - 1  # length: terms
    idf = np.expand_dims(idf, axis=0)  # shape: (1, terms)

    doc_lens = X.sum(axis=1)  # shape: (documents, 1)
    avg_dl = np.mean(doc_lens)

    # compute BM25 score adjustments for document length normalization
    len_norm = (1 - b + b * doc_lens / avg_dl).A1  # shape: (documents,)
    # apply the BM25 term frequency adjustment with k1 adjustment
    denominator = X + k1 * len_norm.reshape(-1, 1)  # shape: (documents, terms)
    numerator = X.multiply(k1 + 1)  # shape: (documents, terms)
    bm25 = numerator / denominator  # shape: (documents, terms)
    # multiply BM25 scores by IDF values to get final scores
    bm25_idf = bm25.multiply(idf)

    # compute the dot product of the query and document matrices
    scores = Q * bm25_idf.T  # shape: (queries, documents)

    return scores


if __name__ == "__main__":
    print("*" * 80)
    print("Generating paper recommendations based on the full text using BM25...")

    metadata = get_metadata()
    metadata, recs = validate_metadata(metadata)

    # combine title, abstract, and body text for BM25 input
    docs = [
        f"{m.get('title', '')} {m.get('abstract', '')} {m.get('body_text', '')}"
        for m in metadata
    ]
    # compute similarity in a doc * doc matrix
    scores = compute_bm25_scores(docs, docs)

    # add recommendations iteratively
    for idx, rec in enumerate(recs):
        sim_paper_indices = np.argsort(scores.toarray()[idx, :])[::-1][1:11]
        rec.update(
            {
                f"Recommendation {i + 1}": recs[paper_idx]["Article ID"]
                for i, paper_idx in enumerate(sim_paper_indices)
            }
        )

    # sort and save
    sort_then_save(recs, BM25_TSV_PATH)
    print(
        f"Each paper's top 10 similar papers, along with additional metadata, have\n"
        f"been successfully saved to {BM25_TSV_PATH}. {len(recs)} papers are in the\n"
        f"BM25-based recommendation using title, abstract, and body text."
    )
    print("*" * 80)
