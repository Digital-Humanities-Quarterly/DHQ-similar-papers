# DHQ-similar-papers

This is a repository for collaborative work surrounding DHQ similar paper recommendation. 

DHQ is beginning an initiative to generate paper recommendations on the DHQ website in order to improve discoverability and user engagement on the site. This work is a collaboration between the general editors, the managing editors, the data analytics team, and the collaborative development team. To accomplish this, we are exploring three approaches to paper recommendation:

- Using the manually-curated paper keywords.
- Utilizing the topic model being developed by managing editor Benjamin Grey.
- Experimenting with paper embeddings from large language models.

Each of these three approaches have distinct advantages, ranging from clearly interpretable reasons for why papers are being recommended to utilizing state-of-the-art approaches from machine learning. Ideally, users will be able to control how the paper recommendations are being surfaced by selecting which approach they are interested in. To accomplish this, we will accompany the paper recommendations with helpful modals, tooltips, and an FAQ page detailing how the recommendations are being generated. We plan to eeventually conduct informal user testing surrounding the paper recommendation quality before deploying this affordance on the live site.

More details on each of the three approaches described above will be provided in this documentation as they are implemented.

## Ben's approach:
Ben's approach to similar paper recommendation involves the construction of embeddings using transformers, namely, AI2's [https://github.com/allenai/specter](SPECTER) ("Document-level Representation Learning using Citation-informed Transformers"). In particular, SPECTER has been pre-trained on scientific papers using a similar papers task in order to produce high-performing, domain-specific embeddings. Utilizing the pre-trained [HuggingFace implementation](https://huggingface.co/allenai/specter), it is straightforward to generate embeddings for our similar papers task -- namely, the title and abstract for each paper is concatenated and treated as the textual input. A technical overview of SPECTER can be found in [this ArXiv paper](https://arxiv.org/abs/2004.07180).

## Meeting notes:
- [Paper Recommendations Meetings Notes](https://drive.google.com/drive/folders/1N3-368_BLbl5exN62npnUPpcPIpS1CWW?usp=sharing)


## Additional relevant resources:
- [SPECTER](https://github.com/allenai/specter)

