# DHQ Recommendation Systems

This repository contains scripts for creating three separate paper recommendation systems for the journal 
*Digital Humanities Quarterly* (DHQ):

- **[Keyword-based Recommendations](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/run_kwd_recs.py)**: 
Recommendations are based on the DHQ Classification Scheme, an editor-assigned controlled vocabulary comprising 88 
terms, such as `#gender` and `#machine_learning`.
- **[BM25 Recommendations](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/run_bm25_recs.py)**: 
Recommendations are based on the full text (i.e., a concatenation of title, abstract, and body text without references) 
using the BM25 algorithm.
- **[SPECTER2-based Recommendations](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/run_spctr_recs.py)**: 
Recommendations are generated using the hidden states from [SPECTER2](https://huggingface.co/allenai/specter2), based on
the paper's title and abstract.

## Use
To get a set of the most updated recommendations when changes have been introduced to the DHQ-journal repository, please 
click `Run workflow` at 
**[GitHub Actions](https://github.com/Wang-Haining/DHQ-similar-papers/actions/workflows/update_tsvs_now.yml)**.
It will also automatically update the tsv files every 1st and 15th at midnight if 
[dhq-journal](https://github.com/Digital-Humanities-Quarterly/dhq-journal) repository changes.

The ten most similar article IDs for each of the systems are documented in
- [`dhq-recs-zfill-kwd.tsv`](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/dhq-recs-zfill-kwd.tsv)
- [`dhq-recs-zfill-bm25.tsv`](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/dhq-recs-zfill-bm25.tsv), and
- [`dhq-recs-zfill-spctr.tsv`](https://github.com/Wang-Haining/DHQ-similar-papers/blob/main/dhq-recs-zfill-spctr.tsv).


## Workflow
Manual or automatic recommendation works as follows:

1. Initialize the [official DHQ repository](https://github.com/Digital-Humanities-Quarterly/dhq-journal) as a submodule.
2. Extract relevant elements from DHQ papers in TEI format, with the keyword-based recommendation system primarily 
focusing on `dhq_keywords`, and the full text-based recommendation system extracting the title, abstract, and body text 
as well. Papers in the editorial process are not considered.
3. Construct a similarity matrix for generating recommendations.
4. Retrieve the most similar papers from the similarity matrix, utilizing a random seed to handle ties.

## Reproduction
<details>
  <summary>Click to expand</summary>
To reproduce the recommendations on your own machine (not recommended for production), please use the following 
commands:

```bash
# clone the repository and navigate into the directory
git clone https://github.com/Wang-Haining/DHQ-similar-papers.git
cd DHQ-similar-papers

# initialize and update submodules (dhq-journal)
git submodule update --init --remote

# set up a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# install dependencies
python -m pip install -r requirements.txt

# execute the keyword-based recommendation
python -m run_kwd_recs

# execute the BM25 recommendation
python -m run_bm25_recs

# execute the SPECTER2 recommendation
python -m run_spctr_recs
```
</details>

## License
This code and recommendation files are dedicated to the public domain under the CC0 1.0 Universal Public Domain Dedication, 
allowing unrestricted use, modification, and distribution.


## Author
The Digital Humanities Quarterly Data Analytics Team

## Contribution
Please open a ticket for any issues or suggestions, thank you!


## History
<details>
  <summary>Click to expand</summary>

- v0.0.5
  - Streamlined utilities.
  - Added a pipeline for recs recalculation regardless of submodule updates.
- v0.0.4
  - Ignored remembrance pieces in recommendations.
  - Added unit tests.
  - Dumped `annoy` for spctr method.
  - Added a rule to Actions to run tests before commit.
- v0.0.3
  - Merged Ben's SPECTER method.
  - Added CI pipeline with Actions.
  - Improved module/var naming.
  - Updated data files.
- v0.0.2:
  - Implemented the full text-based recommendation system.
  - Included logic for removing papers in the editorial process.
  - Refactored the keyword-based recommendation system.
  - Updated data files for both systems.
- v0.0.1: 
  - Implemented the keyword-based recommendation system.

</details>
