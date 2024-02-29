# DHQ-similar-papers

This repository contains scripts for creating a paper recommendation system for the journal Digital Humanities Quarterly (DHQ). Recommendations are based on the DHQ Classification Scheme, an editor-assigned controlled vocabulary comprising 88 terms, such as `#gender` and `#machine_learning`.

The workflow is as follows:
1. Initialize the official DHQ repository as a submodule.
2. Extract relevant elements from DHQ papers in TEI format, primarily focusing on `dhq_keywords`.
3. Construct a similarity matrix for generating recommendations (refer to `run.py` for more details).
4. Retrieve the most similar papers from the similarity matrix, utilizing a random seed to handle ties in rankings.

## Reproduction

To reproduce the setup and run the scripts, use the following commands:

```bash
git submodule update --init
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m run
```

## Alternative Implementation
- [Ben's solution using SPECTER 2 embeddings.](https://github.com/bcglee/DHQ-similar-papers)

## Known Issues
1. Papers with indices 000424, 000432, 000488, and 000492 are excluded.
2. The Volume and Issue information is missing for paper 000664.
3. Papers with indices 000678 and 000722 do not have associated .xml files.
4. Papers with indices 000710 and 000716 lack dhq_keywords associations (thus, they will not appear in recommendations).
5. The publication year is missing for 15 papers (refer to run.py for a comprehensive list).

## To-Do List:
1. Write unit tests.
2. Implement a CI/CD process to automatically update the zfill spreadsheet.
