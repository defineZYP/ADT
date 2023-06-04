# Adaptive Disentangled Transformer

## Overall description

This is the repository of the KDD 2023 paper "Adaptive Disentangled Trasnformer for Sequential Recommendation"

This repository reimplements the backbone model from [SASRec](https://github.com/pmixer/SASRec.pytorch)、[Bert4Rec](https://github.com/FeiSun/BERT4Rec) and [STOSA](https://github.com/zfan20/STOSA).



## Requirements

The following libraries and versions are used in the experiments, these requirements are not mandatory and are only for reference.

```
numpy = 1.22.4
Python = 3.8.10
Pytorch = 1.11.0
scipy = 1.9.3
tqdm = 4.61.2
```

 

## Usage

### Search

For each backbone, you can use this command to search the best lambdas for training the model. You can go the source file for more details.

```
python evolution.py --dataset xxx
```



### Retrain

After searching process, you will get the best candidate in `/res` and you can copy the candidate vector and your search space to `candidates_to_lambdas.py` and run it to get the best lambdas. And then copy the lambdas to function `get_lambdas` in `utils.py` and retrain the model to get the final score. 

For convenience, we have recorded part of the lambdas used in our experiments in `utils.py` and hyperparameters in `/templates` as an example.

```
python main.py --dataset xxx
```



## Data

Part of the dataset used for SASRec backbone and Bert4Rec backbone is uploaded in the repository, and ml-20m can download from this [link](https://grouplens.org/datasets/movielens/). For STOSA backbone, you can download the 5-score dataset from [Amazon Dataset](https://jmcauley.ucsd.edu/data/amazon/) or go to the [original repository](https://github.com/zfan20/STOSA) for the details.