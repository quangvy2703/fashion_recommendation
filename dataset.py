
import pickle
import pandas as pd
# import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from tqdm import tqdm
import random
import os
from sklearn.model_selection import train_test_split

from sklearn_pandas import DataFrameMapper
from torch.utils.data import Dataset
import torch
from datetime import datetime

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FashionDataset(Dataset):
    def __init__(self, config, data_dir, all_samples, phase):
        self.data_dir = data_dir
        self.phase = phase
        self.config = config
        self.samples = all_samples
 
        
        with open(os.path.join(data_dir, phase + '_indexes.pkl'), 'rb') as f:
            self.indexes = pickle.load(f)
            n_remove = len(self.indexes) % self.config['BATCH_SIZE']
            self.indexes = set(self.indexes) - set(random.sample(self.indexes, k=n_remove))
            self.indexes = list(self.indexes) 

        # with open(os.path.join(self.data_dir, "articles_processed.pkl"), "rb") as f:
        #     articles = pickle.load(f)

        # with open(os.path.join(self.data_dir, "customers_processed.pkl"), "rb") as f:
        #     customers = pickle.load(f)

        # self.articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
        # self.customers_dict = dict(zip(customers[:,0], np.array(customers[:, 1:-1], dtype=np.float32)))
        # with open(os.path.join(data_dir, 'articles_processed.pkl'), 'rb') as f:
        #     self.articles = pickle.load(f) 
        # with open(os.path.join(data_dir, 'customers_processed.pkl'), 'rb') as f:
        #     self.customers = pickle.load(f) 
        


    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        index = self.indexes[idx]
        # customer_id = self.samples['customer_id'][index]
        # sequence_features = self.samples['sequence_features'][index]
        # customer_features = self.samples['customer_features'][index]
        target = int(self.samples['target'][index])

        # sequence_features = [self.articles_dict[article] for article in self.samples['sequence'][index]]
        sequence_features = self.samples['sequence_features'][index]
        _len = len(sequence_features)
        if _len < self.config['MAX_SEQUENCE_LENGTH']:
            zeros_features = self.config['TRANSACTION_ENCODER_INPUT_SIZE'] * [0]
            sequence_features = sequence_features + [zeros_features] * (self.config['MAX_SEQUENCE_LENGTH'] - _len)
        sequence_features = np.array(sequence_features, dtype = np.float)
        # customer_features = self.customers_dict[self.samples['customer_id'][index]]
        customer_features = np.array(self.samples['customer_features'][index], dtype=np.float)
        # print(customer_features)
        # sequence_features = torch.tensor(sequence_features, dtype=torch.float, device=device)
        # customer_features = torch.tensor(customer_features, dtype=torch.float, device=device)
        # target = torch.tensor(int(target), dtype=torch.long, device=device)
        return {'sequence_features': sequence_features, 'customer_features': customer_features, 'target': target, 'length': _len}



 # transactions_df['article_id'] = transactions_df['article_id'].astype(str)
# transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'], format="%Y-%m-%d")
# trans_by_cus = transactions_df.groupby(['customer_id'])
# customer_ids = set(transactions_df.customer_id.values)
    
    def gen_train_data(self, transactions_df):
        transactions_df['article_id'] = transactions_df['article_id'].astype(str)
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'], format="%Y-%m-%d")
        trans_by_cus = transactions_df.groupby(['customer_id'])
        customer_ids = set(transactions_df.customer_id.values)

        customer_sequences = {}
        for customer_id in tqdm(customer_ids):
            transaction_of_customer = trans_by_cus.get_group(customer_id).groupby(['t_dat']).agg(','.join).reset_index()
            transaction = transaction_of_customer['article_id'].values
            if len(transaction) < self.config['MIN_SEQUENCE_LENGTH']:
                continue

            excuce_date = transaction_of_customer.t_dat.values
            from_last = np.asarray(excuce_date) - np.asarray([np.datetime64('0-01-01'), *excuce_date[:-1]])
            transaction_of_customer['from_last'] = from_last.astype('timedelta64[D]').astype(np.int32)

            indexes = np.array(transaction_of_customer[transaction_of_customer['from_last'] <= self.config['PERIODS']].index.to_list())
            indexes = np.sort(np.unique(np.append(indexes, indexes-1)))

            transaction_sessions = transaction_of_customer.iloc[indexes].reset_index(drop=True)
            split_indexes = transaction_sessions.from_last[transaction_sessions.from_last > self.config['PERIODS']].index.to_list()
            transaction_sessions = np.split(transaction_sessions.article_id.values, split_indexes[1:])

            customer_sequences[customer_id] = []
            for transaction_session in transaction_sessions:
                a_session = []
                for session in transaction_session:
                    a_session = a_session + list(map(int, session.split(',')))

                customer_sequences[customer_id].append(a_session)
    #         if len(customer_sequences) > 20:
    #             break
        with open('customer_sequences.pkl', "wb") as f:
            pickle.dump(customer_sequences, f)
            
    def gen_test_data(self, transactions_df):
        transactions_df['article_id'] = transactions_df['article_id'].astype(str)
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'], format="%Y-%m-%d")
        trans_by_cus = transactions_df.groupby(['customer_id'])
        customer_ids = set(transactions_df.customer_id.values)
        last_date = transactions_df['t_dat'].copy().sort_values().values[-1]
        dat_t = transactions_df['dat_t']
        last_train_date = last_date - np.timedelta64(self.config['PERIODS'], 'D')
        
        customer_sequences = {}
        for customer_id in tqdm(customer_ids):
            transaction_of_customer = trans_by_cus.get_group(customer_id).groupby(['t_dat']).agg(','.join).reset_index()
            test_indexes = np.array(transaction_of_customer[transaction_of_customer['dat_t'] >= last_train_date].index().to_list())
            transaction = transaction_of_customer['article_id'].iloc[test_indexes].values()


            excuce_date = transaction_of_customer.t_dat.values
            from_last = np.asarray(excuce_date) - np.asarray([np.datetime64('0-01-01'), *excuce_date[:-1]])
            transaction_of_customer['from_last'] = from_last.astype('timedelta64[D]').astype(np.int32)

            indexes = np.array(transaction_of_customer[transaction_of_customer['from_last'] <= self.config['PERIODS']].index.to_list())
            indexes = np.sort(np.unique(np.append(indexes, indexes-1)))

            transaction_sessions = transaction_of_customer.iloc[indexes].reset_index(drop=True)
            split_indexes = transaction_sessions.from_last[transaction_sessions.from_last > self.config['PERIODS']].index.to_list()
            transaction_sessions = np.split(transaction_sessions.article_id.values, split_indexes[1:])
            last_transaction_session = transaction_sessions[-1]
            customer_sequences[customer_id] = []
            
            a_session = []
            for session in last_transaction_session:
                a_session = a_session + list(map(int, session.split(',')))

                customer_sequences[customer_id].append(a_session)
    #         if len(customer_sequences) > 20:
    #             break
        with open('customer_sequences.pkl', "wb") as f:
            pickle.dump(customer_sequences, f)

    def article_processing(self):
        articles_df = pd.read_csv(self.config['ARTICLES_PATH'])
        article_infor = articles_df[ARTICLE_FEATURES]
        article_mapper = DataFrameMapper([
            ('article_id', None),
            ('product_type_no', LabelBinarizer()),
            ('graphical_appearance_no', LabelBinarizer()),
            ('colour_group_code', LabelBinarizer()),
            ('perceived_colour_value_id', LabelBinarizer()),
            ('perceived_colour_master_id', LabelBinarizer()),
            ('department_no', LabelBinarizer()),
            ('index_code', LabelBinarizer()),
            ('index_group_no', LabelBinarizer()),
            ('section_no', LabelBinarizer()),
            ('garment_group_no', LabelBinarizer())
        ])

        article_features = article_mapper.fit_transform(article_infor.copy())
        # article_features = dict(zip(articles_df.article_id, article_features))
        with open('datasets/articles_processed.pkl', 'wb') as f:
            pickle.dump(article_features, f)


    
    def customer_processing(self):
        customers_df = pd.read_csv(self.config['CUSTOMERS_PATH'])
        customers_mapper = DataFrameMapper([
            ('customer_id', None),
            ('FN', LabelEncoder()),
            ('Active', LabelEncoder()),
            ('club_member_status', LabelBinarizer()),
            ('fashion_news_frequency', LabelBinarizer()),
            ('age', None),
            ('postal_code', None)
            ])

        customers_df['FN'].fillna(0, inplace=True)
        customers_df['Active'].fillna(0, inplace=True)
        customers_df['age'].fillna(self.config['AGE_FILL_VALUE'], inplace=True)
        customers_df['club_member_status'].fillna('NA', inplace=True)
        customers_df['fashion_news_frequency'].fillna('NA', inplace=True)
        customers_df['fashion_news_frequency'].replace('None', 'NONE', inplace=True)
        customers_df['postal_code'].fillna('NA', inplace=True)
        age = customers_df['age'].values.reshape(-1, 1) 
        min_max_scaler = MinMaxScaler()
        age_scaled = min_max_scaler.fit_transform(age)
        customers_df['age'] = age_scaled
        customers_features = customers_mapper.fit_transform(customers_df.copy())
        # customers_features = dict(zip(customers_df.customer_id, customers_features))
        with open('datasets/customers_processed.pkl', 'wb') as f:
            pickle.dump(customers_features, f)

    def create_dataset(self):
        with open("datasets/customer_sequences.pkl", "rb") as f:
            customer_sequences = pickle.load(f)
        # with open("datasets/articles_processed.pkl", "rb") as f:
        #     articles = pickle.load(f)
        # with open("datasets/customers_processed.pkl", "rb") as f:
        #     customers = pickle.load(f)

        # articles_dict = dict(zip(articles[:,0], articles[:, 1:]))
        # customers_dict = dict(zip(customers[:,0], customers[:, 1:-1]))

        samples = {}
        samples['customer_id'] = []
        samples['sequence'] = []
        samples['target'] = []
        samples['customer_features'] = []
        samples['sequence_features'] = []
        articles_df = pd.read_csv(self.config['ARTICLES_PATH'])
        article_ids = articles_df.article_id.values
        customer_ids = customer_sequences.keys()
        for customer_id in tqdm(customer_ids):
            sequences = customer_sequences[customer_id]
            # print(sequence)
            for sequence in sequences:
                for i in range(len(sequence) - 2):
                    purchased = sequence[0: i + 2]
                    if len(purchased) > self.config['MAX_SEQUENCE_LENGTH']:
                        purchased = purchased[-self.config['MAX_SEQUENCE_LENGTH']: ]
                    while True:
                        unpurchased = random.choice(article_ids)
                        if unpurchased not in sequence:
                            break
                    # Positive sample
                    samples['customer_id'].append(customer_id)
                    samples['sequence'].append(purchased)
                    # sequence_features = [for article in ]
                    # samples['sequence_features'].append([articles_dict[article] for article in purchased])
                    # samples['customer_features'].append(customers_dict[customer_id])
                    samples['target'].append(1)
                    # Negative sample
                    samples['customer_id'].append(customer_id)
                    samples['sequence'].append(purchased[:-1] + [unpurchased])
                    # samples['sequence_features'].append([articles_dict[article] for article in purchased])
                    # samples['customer_features'].append(customers_dict[customer_id])
                    samples['target'].append(0)

        indexes = [i for i in range(len(samples['target']))]
        x_train, x_test, y_train, y_test = train_test_split(indexes, samples['target'], test_size=0.3, shuffle=True)

        with open('datasets/all_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
        with open('datasets/train_indexes.pkl', 'wb') as f:
            pickle.dump(x_train, f)
        with open('datasets/valid_indexes.pkl', 'wb') as f:
            pickle.dump(x_test, f)
        
    
            