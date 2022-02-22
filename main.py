"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
from encoder_train import Training
from config import cfg
from dataset import FashionDataset
import pickle
import pandas as pd
# from tqdm import tqdm
# import random
# from sklearn.model_selection import train_test_split
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=1280)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


# from config import *
# def create_dataset():
#     with open("datasets/customer_sequences.pkl", "rb") as f:
#         customer_sequences = pickle.load(f)
#     # with open("datasets/articles_processed.pkl", "rb") as f:
#     #     articles = pickle.load(f)
#     # with open("datasets/customers_processed.pkl", "rb") as f:
#     #     customers = pickle.load(f)

#     # articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
#     # customers_dict = dict(zip(customers[:,0], customers[:, 1:-1]))
#     samples = {}
#     samples['customer_id'] = []
#     samples['sequence'] = []
#     samples['target'] = []
#     samples['customer_features'] = []
#     samples['sequence_features'] = []
#     articles_df = pd.read_csv(cfg['ARTICLES_PATH'])
#     article_ids = articles_df.article_id.values
#     customer_ids = customer_sequences.keys()
#     for customer_id in tqdm(customer_ids):
#         sequences = customer_sequences[customer_id]
#         # print(sequence)
#         for sequence in sequences:
#             for i in range(len(sequence) - 2):
#                 purchased = sequence[0: i + 2]
#                 if len(purchased) > cfg['MAX_SEQUENCE_LENGTH']:
#                     purchased = purchased[-cfg['MAX_SEQUENCE_LENGTH']: ]
#                 while True:
#                     unpurchased = random.choice(article_ids)
#                     if unpurchased not in sequence:
#                         break
#                 # Positive sample
#                 samples['customer_id'].append(customer_id)
#                 samples['sequence'].append(purchased)
#                 # sequence_features = [for article in ]
#                 # samples['sequence_features'].append([articles_dict[article] for article in purchased])
#                 # samples['customer_features'].append(customers_dict[customer_id])
#                 samples['target'].append(1)
#                 # Negative sample
#                 samples['customer_id'].append(customer_id)
#                 samples['sequence'].append(purchased[:-1] + [unpurchased])
#                 # samples['sequence_features'].append([articles_dict[article] for article in purchased])
#                 # samples['customer_features'].append(customers_dict[customer_id])
#                 samples['target'].append(0)

#     indexes = [i for i in range(len(samples['target']))]
#     x_train, x_test, y_train, y_test = train_test_split(indexes, samples['target'], test_size=0.3, shuffle=True)

#     with open('datasets/all_samples.pkl', 'wb') as f:
#         pickle.dump(samples, f)
#     with open('datasets/train_indexes.pkl', 'wb') as f:
#         pickle.dump(x_train, f)
#     with open('datasets/valid_indexes.pkl', 'wb') as f:
#         pickle.dump(x_test, f)

# create_dataset()


if __name__ == '__main__':
    args = parse_args()
    cfg['DATA_DIR'] = args.data_dir
    cfg['EPOCHS'] = args.epochs
    cfg['HIDDEN_SIZE'] = args.hidden
    cfg['BATCH_SIZE'] = args.batch_size
    cfg['LR'] = args.lr

    
    train_dataset = FashionDataset(cfg, cfg['DATA_DIR'], 'train')

    trainer = Training(cfg)
    trainer.start_training()


 
  