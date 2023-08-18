# DHQ-similar-papers

This is a repository for collaborative work surrounding DHQ similar paper recommendation. 

DHQ is beginning an initiative to generate paper recommendations on the DHQ website in order to improve discoverability and user engagement on the site. This work is a collaboration between the general editors, the managing editors, the data analytics team, and the collaborative development team. To accomplish this, we are exploring three approaches to paper recommendation:

- Using the manually-curated paper keywords.
- Utilizing the topic model being developed by managing editor Benjamin Grey.
- Experimenting with paper embeddings from large language models.

Each of these three approaches have distinct advantages, ranging from clearly interpretable reasons for why papers are being recommended to utilizing state-of-the-art approaches from machine learning. Ideally, users will be able to control how the paper recommendations are being surfaced by selecting which approach they are interested in. To accomplish this, we will accompany the paper recommendations with helpful modals, tooltips, and an FAQ page detailing how the recommendations are being generated. We plan to eeventually conduct informal user testing surrounding the paper recommendation quality before deploying this affordance on the live site.

More details on each of the three approaches described above will be provided in this documentation as they are implemented.

## Ben's approach:
Ben's approach to similar paper recommendation involves the construction of embeddings using transformers, namely, AI2's [SPECTER](https://github.com/allenai/specter) ("Document-level Representation Learning using Citation-informed Transformers"). In particular, SPECTER has been pre-trained on scientific papers using a similar papers task in order to produce high-performing, domain-specific embeddings. Utilizing the pre-trained [HuggingFace implementation](https://huggingface.co/allenai/specter), it is straightforward to generate embeddings for our similar papers task -- namely, the title and abstract for each paper is concatenated and treated as the textual input. A technical overview of SPECTER can be found in [this ArXiv paper](https://arxiv.org/abs/2004.07180).

From the SPECTER repo, where embeddings are created for a simple database of papers (titles + abstracts):
```
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
result = model(**inputs)
# take the first token in the batch as the embedding
embeddings = result.last_hidden_state[:, 0, :]
```

Ben's full working implementation can be found in [this Jupyter notebook](generate_SPECTER2_embeddings.ipynb). The SPECTER embeddings are used to generate the 10 most similar papers to each DHQ paper in the corpus (as of 2022). These recommendations can be found in [2022-dhq-articles-with-abstracts-and-SPECTER-recommendations.csv](2022-dhq-articles-with-abstracts-and-SPECTER-recommendations.csv). NOTE: here, we use [SPECTER 2 emgeddings](https://huggingface.co/allenai/specter2).

## Meeting notes:
- [Paper Recommendations Meetings Notes](https://drive.google.com/drive/folders/1N3-368_BLbl5exN62npnUPpcPIpS1CWW?usp=sharing)


## Additional relevant resources:
- [SPECTER](https://github.com/allenai/specter)

