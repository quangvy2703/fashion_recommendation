import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from config import *
from dataset import FashionDataset
import warnings
warnings.filterwarnings('ignore')


def gen_train_data(transactions_df):
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)
    transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'], format="%Y-%m-%d")
    trans_by_cus = transactions_df.groupby(['customer_id'])
    customer_ids = set(transactions_df.customer_id.values)

    customer_sequences = {}
    for customer_id in tqdm(customer_ids):
        transaction_of_customer = trans_by_cus.get_group(customer_id).groupby(['t_dat']).agg(','.join).reset_index()
        transaction = transaction_of_customer['article_id'].values
        if len(transaction) < MIN_SEQUENCE_LENGTH:
            continue

        excuce_date = transaction_of_customer.t_dat.values
        from_last = np.asarray(excuce_date) - np.asarray([np.datetime64('0-01-01'), *excuce_date[:-1]])
        transaction_of_customer['from_last'] = from_last.astype('timedelta64[D]').astype(np.int32)

        indexes = np.array(transaction_of_customer[transaction_of_customer['from_last'] <= PERIODS].index.to_list())
        indexes = np.sort(np.unique(np.append(indexes, indexes-1)))

        transaction_sessions = transaction_of_customer.iloc[indexes].reset_index(drop=True)
        split_indexes = transaction_sessions.from_last[transaction_sessions.from_last > PERIODS].index.to_list()
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


fashion_dataset = FashionDataset(DATA_DIR, 'train_indexes.pkl')
# fashion_dataset.article_processing()
# print("Article processed.")
# fashion_dataset.customer_processing()
# print("Customer processed.")
fashion_dataset.create_dataset()
print("Processed.")
