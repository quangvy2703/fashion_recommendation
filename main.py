"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
# from encoder_train import Training
from config import cfg
# from dataset import FashionDataset
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split
from run_transformer import train_transformer
import warnings

from transformer import test_transformer
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--saved_data_dir', type=str, default='processed')
    parser.add_argument('--model_path', type=str, default='processed')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_test', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg['DATA_DIR'] = args.data_dir
    cfg['EPOCHS'] = args.epochs
    cfg['HIDDEN_SIZE'] = args.hidden
    cfg['BATCH_SIZE'] = args.batch_size
    cfg['N_LAYERS'] = args.n_layers
    cfg['LR'] = args.lr
    cfg['EPOCH_TEST'] = args.epoch_test
    cfg['MODEL_PATH'] = args.model_path
    

    # trainer = Training(cfg)
    # trainer.start_training()
    if args.testing:
        print("Starting testing phase...")
        test_transformer(args.saved_data_dir, args.epoch_test)
    else:
        print("Starting training phase...")
        train_transformer(data_dir=args.data_dir, saved_data_dir=args.saved_data_dir)

# from transformer import *

# transactions = pickle.load(open('datasets/customer_sequences.pkl', 'rb'))
# customer_ids, source, target = preprocess_corpus(transactions, 1)
# vocab = read_vocab(source, target)
# vocab.to_file(os.path.join('datasets_transformer', 'vocab.txt'))

# print('Corpus length: {}\nVocabulary size: {}'.format(
#     len(source), len(vocab.article2index)))

# examples = list(zip(source, target))[80:90]
# for source, target in examples:
#     print(f'Source: "{source}", target: "{target}"')
    
# from config import *
# def create_dataset():
#     with open("datasets/customer_sequences.pkl", "rb") as f:
#         customer_sequences = pickle.load(f)
#     with open("datasets/articles_processed.pkl", "rb") as f:
#         articles = pickle.load(f)
#     with open("datasets/customers_processed.pkl", "rb") as f:
#         customers = pickle.load(f)

#     articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
#     customers_dict = dict(zip(customers[:,0], customers[:, 1:-1]))
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
#                 # samples['sequence'].append(purchased)

#                 samples['sequence_features'].append([articles_dict[article] for article in purchased])
#                 samples['customer_features'].append(customers_dict[customer_id])
#                 samples['target'].append(1)
#                 # Negative sample
#                 samples['customer_id'].append(customer_id)
#                 # samples['sequence'].append(purchased[:-1] + [unpurchased])
#                 un = purchased[:-1] + [unpurchased]
#                 samples['sequence_features'].append([articles_dict[article] for article in un])
#                 samples['customer_features'].append(customers_dict[customer_id])
#                 samples['target'].append(0)

#     indexes = [i for i in range(len(samples['target']))]
#     x_train, x_test, y_train, y_test = train_test_split(indexes, samples['target'], test_size=0.3, shuffle=True)

#     with open('datasets/all_samples_2.pkl', 'wb') as f:
#         pickle.dump(samples, f)
#     with open('datasets/train_indexes_2.pkl', 'wb') as f:
#         pickle.dump(x_train, f)
#     with open('datasets/valid_indexes_2.pkl', 'wb') as f:
#         pickle.dump(x_test, f)

# def create_dataset_transformer():
#     with open("datasets/customer_sequences.pkl", "rb") as f:
#         customer_sequences = pickle.load(f)
#     with open("datasets/articles_processed.pkl", "rb") as f:
#         articles = pickle.load(f)
#     with open("datasets/customers_processed.pkl", "rb") as f:
#         customers = pickle.load(f)

#     articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
#     customers_dict = dict(zip(customers[:,0], customers[:, 1:-1]))

#     samples = {}
#     samples['customer_id'] = []
#     samples['target_features'] = []
#     samples['customer_features'] = []
#     samples['source_features'] = []
#     customer_ids = customer_sequences.keys()
#     for customer_id in tqdm(customer_ids, total=len(customer_ids)):
#         sequences = customer_sequences[customer_id]
#         for idx in range(len(sequences) - 1):
#             # Create a corpus
#             samples['customer_id'].append(customer_id)
#             samples['source_features'].append([articles_dict[article] for article in sequences[idx]])
#             samples['target_features'].append([articles_dict[article] for article in sequences[idx + 1]])
#             samples['customer_features'].append(customers_dict[customer_id])

#     indexes = [i for i in range(len(samples['customer_id']))]
#     x_train, x_test, y_train, y_test = train_test_split(indexes, indexes, test_size=0.3, shuffle=True)

#     with open('datasets_transformer/all_samples.pkl', 'wb') as f:
#         pickle.dump(samples, f)
#     with open('datasets_transformer/train_indexes.pkl', 'wb') as f:
#         pickle.dump(x_train, f)
#     with open('datasets_transformer/valid_indexes.pkl', 'wb') as f:
#         pickle.dump(x_test, f)
    
#     print(f"Created dataset with {len(samples['customer_id'])} corpus")
    
# def gen_test_data(transactions_df):
#     transactions_df['article_id'] = transactions_df['article_id'].astype(str)
#     transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'], format="%Y-%m-%d")
#     trans_by_cus = transactions_df.groupby(['customer_id'])
#     customer_ids = transactions_df.customer_id.unique()
#     last_date = transactions_df['t_dat'].copy().sort_values().values[-1]
#     last_train_date = last_date - np.timedelta64(cfg['PERIODS'], 'D')
#     trans_by_cus = transactions_df[transactions_df['t_dat'] >= last_train_date]
#     customer_filter_ids = trans_by_cus.customer_id.unique()
#     other_customer_ids = list(set(customer_ids) - set(customer_filter_ids))
#     trans_by_cus = trans_by_cus.groupby(['customer_id'])
#     customer_sequences = {}
#     for customer_id in tqdm(customer_filter_ids, total=len(customer_filter_ids), desc=f"{len(customer_sequences.keys())}"):
#         transaction_of_customer = trans_by_cus.get_group(customer_id).groupby(['t_dat']).agg(','.join).reset_index()
#         transactions = transaction_of_customer.article_id.values
#         customer_sequences[customer_id] = []
        
#         session = []
#         # print(transactions)
#         for transaction in transactions:
#             session = session + list(map(int, transaction.split(',')))
#         customer_sequences[customer_id] = session


#     trans_by_cus = transactions_df.groupby(['customer_id'])

#     for customer_id in tqdm(other_customer_ids, total=len(other_customer_ids)):
#         transaction_of_customer = trans_by_cus.get_group(customer_id).groupby(['t_dat']).agg(','.join).reset_index()
#         transaction = transaction_of_customer['article_id'].values

#         last_date = transaction_of_customer['t_dat'].copy().sort_values().values[-1]
#         last_train_date = last_date - np.timedelta64(cfg['PERIODS'], 'D')

#         last_session = transaction_of_customer[transaction_of_customer['t_dat'] >= last_train_date].article_id.values

#         customer_sequences[customer_id] = []
#         a_session = []
#         for session in last_session:
#             a_session = a_session + list(map(int, session.split(',')))

#         customer_sequences[customer_id] = a_session

#     with open('datasets_transformer/customer_sequences_submission.pkl', "wb") as f:
#         pickle.dump(customer_sequences, f)

# def create_dataset_submission():
#     with open("datasets_transformer/customer_sequences_submission.pkl", "rb") as f:
#         customer_sequences = pickle.load(f)
#     with open("datasets/articles_processed.pkl", "rb") as f:
#         articles = pickle.load(f)
#     with open("datasets/customers_processed.pkl", "rb") as f:
#         customers = pickle.load(f)

#     articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
#     customers_dict = dict(zip(customers[:,0], customers[:, 1:-1]))

#     samples = {}
#     samples['customer_id'] = []
#     samples['customer_features'] = []
#     samples['source_features'] = []
#     customer_ids = customer_sequences.keys()
#     for customer_id in tqdm(customer_ids, total=len(customer_ids)):
#         sequences = customer_sequences[customer_id]
  
#         # Create a corpus
        
#         samples['customer_id'].append(customer_id)
#         samples['source_features'].append([articles_dict[article] for article in sequences])
#         samples['customer_features'].append(customers_dict[customer_id])

#     with open('datasets_transformer/all_samples_submission.pkl', 'wb') as f:
#         pickle.dump(samples, f)
    
#     print(f"Created dataset with {len(samples['customer_id'])} corpus")    


# create_dataset()
# transaction_df = pd.read_csv(cfg["TRANSACTIONS_PATH"])
# gen_test_data(transaction_df)
# create_dataset_submission()




 
  