# Two DHQ Article Recommendation Systems


This repository contains scripts for creating two separate paper recommendation systems for the journal 
*Digital Humanities Quarterly* (DHQ):

- **Keyword-based Recommendations**: Recommendations are based on the DHQ Classification Scheme, an editor-assigned 
controlled vocabulary comprising 88 terms, such as `#gender` and `#machine_learning`.
- **Full text-based Recommendations**: Recommendations are based on the full text (i.e., a concatenation of title, 
abstract, and body text without references) using the BM25 algorithm.


## Workflow
1. Initialize the [official DHQ repository](https://github.com/Digital-Humanities-Quarterly/dhq-journal) as a submodule.
2. Extract relevant elements from DHQ papers in TEI format, with the keyword-based recommendation system primarily 
focusing on `dhq_keywords`, and the full text-based recommendation system extracting the title, abstract, and body text 
as well. Papers are in the editorial process are not considered.
3. Construct a similarity matrix for generating recommendations (refer to `run_keyword_based_recommendation.py` and 
`run_full_text_bm25_recommendation.py` for more details).
4. Retrieve the most similar papers from the similarity matrix, utilizing a random seed to handle ties in rankings.

## Reproduction

To reproduce the setup and run the scripts, use the following commands:

```bash
# Clone the repository and navigate into the directory
git clone https://github.com/Wang-Haining/DHQ-similar-papers.git
cd DHQ-similar-papers

# Initialize and update submodules (dhq-journal)
git submodule update --init --remote

# Set up a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Execute the keyword-based recommendation system
python -m run_keyword_based_recommendation

# Execute the full text-based recommendation system
python -m run_full_text_bm25_recommendation
```

The ten most similar article IDs for each of the systems are documented in `dhq-recs-zfill-keyword.tsv` and 
`dhq-recs-zfill-full-text.tsv`. Each time the DHQ repository is updated, the recommendation scripts are expected to 
run again to ensure the most up-to-date recommendations.

## Alternative Implementation
- [Ben's solution using SPECTER 2 embeddings.](https://github.com/bcglee/DHQ-similar-papers)

## To-Do List:
1. Write unit tests.
2. Implement a CI/CD process to automatically update the zfill spreadsheet.

## License
MIT

## Author
The Digital Humanities Quarterly Data Analytics Team

## Contribution
Please open an issue for any suggestions, thank you!

## History
- v0.0.1: 
  - Implemented the keyword-based recommendation system.
- v0.0.2:
  - Implemented the full text-based recommendation system.
  - Included logic for removing papers in the editorial process.
  - Refactored the keyword-based recommendation system.
  - Updated data files for both systems.
