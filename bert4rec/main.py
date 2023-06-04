import gc
import os
import torch
from options import args
from datasets.dataset import BertTrainDataset, BertEvalDataset
from trainer import BertTrainer
from utils import *

def main():
    export_root = setup_train(args)
    trainer = BertTrainer(args)
    lambda1, lambda2 = get_lambda(args.dataset, args.topk)
    best_epoch, rec_valid, rec_valid_AUC, rec_test, rec_test_AUC = trainer.train(lambda1, lambda2)
    print(f"HERE: {rec_valid[0][5]:.4f}, {rec_valid[1][5]:.4f}, {rec_valid_AUC}, {rec_test[0][5]:.4f}, {rec_test[1][5]:.4f}, {rec_test_AUC})")
    # trainer.test()

if __name__ == "__main__":
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
