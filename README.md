# Information Retreival Using VectorSpace Model
## Introduction
This is Project 1 for the 2021 Information Retrieval course in NCCU.

The main goal is to rank the relevance between documents written in English or Traditional Chinese and given an input query.

In this project, TF-IDF and TF weighting is used to build the query vector and document vectors; Cosine Similarity and Euclidean Distance are used to calculate the relevance between query and documents.

## Environment

**Python**  3.7.3

**nltk**  3.5

**jieba**  0.39.1 (Traditional Chinese Words Segmentation Utilities)

**SciPy**  1.4.1

**numpy**  1.18.1

## Setup
To install all the required packages, run the following command.
```bash
bash install.sh
```
After running install.sh, nltk, SciPy, jieba will be automatically installed.

To work better on Traditional Chinese, we use **specialize version jieba** which good at traditional Chinese words tokenizing.

## Run the code

```bash
python VectorSpace.py
```
