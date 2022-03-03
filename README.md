## Contriever: Towards Unsupervised Dense Information Retrieval with Contrastive Learning

This repository contains pre-trained models and some evaluation code for our paper [Towards Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118).

We use a simple contrastive learning framework to pre-train models for information retrieval. Contriever, trained without supervision, is competitive with BM25 for R@100 on the BEIR benchmark. After finetuning on MSMARCO, Contriever obtains strong performance, especially for the recall at 100.

## Getting Started

Pre-trained models can be loaded through the HuggingFace transformers library:

```python
import transformers
from src.contriever import Contriever

model = Contriever.from_pretrained("facebook/contriever")
tokenizer = transformers.BertTokenizerFast.from_pretrained("facebook/contriever")
```

Embeddings for different sentences can be obtained by doing the following:

```python

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings = model(**inputs)
```

Then similarity scores between the different sentences can be obtained with a dot product between the embeddings:
```python

score01 = embeddings[0] @ embeddings[1] #1.0473
score02 = embeddings[0] @ embeddings[2] #1.0095
```

## BEIR evaluation

Scores on the BEIR benchmark can be reproduced using [eval_beir.py](eval_beir.py).

```bash

python eval_beir.py --model_name_or_path facebook/contriever-msmarco --dataset scifact
```

## Available models

|              Model              | Description |
|:-------------------------------|:--------:|
|  [facebook/contriever](https://huggingface.co/facebook/contriever) | Model pre-trained on Wikipedia and CC-net without any supervised data|
|  [facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco) | Pre-trained model fine-tuned on MS-MARCO|

## References

[1] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave [*Towards Unsupervised Dense Information Retrieval with Contrastive Learning*](https://arxiv.org/abs/2112.09118)

```bibtex
@misc{izacard2021contriever,
      title={Towards Unsupervised Dense Information Retrieval with Contrastive Learning}, 
      author={Gautier Izacard and Mathilde Caron and Lucas Hosseini and Sebastian Riedel and Piotr Bojanowski and Armand Joulin and Edouard Grave},
      year={2021},
      eprint={2112.09118},
      archivePrefix={arXiv},
}
```

## License

See the [LICENSE](LICENSE) file for more details.
