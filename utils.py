import torch
import numpy as np
from typing import Iterable, List, Optional

# Define special symbols and indices
SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'
special_symbols = [SOS_token, EOS_token, UNK_token, PAD_token]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    # print("Create mask ", src_seq_len, tgt_seq_len)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def token_transform(seq):
    return seq.split(',')

def yield_tokens(data):
    for data_sample in data:
        yield token_transform(data_sample)


def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k

def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Relevance at k
    """

    if y_pred[k-1] in y_true:
        return 1, y_pred[k-1]
    else:
        return 0, 0

def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    y_true = np.array(y_true)
    y_true = y_true[y_true > 4]
    y_true_clone = y_true.copy()
    for i in range(0, 4):
        y_true_clone = y_true_clone[y_true_clone != i]
        
    for i in range(1, k+1):
        # print(i, y_true, y_pred)
        res, rem = rel_at_k(y_true_clone, y_pred, i)
        if res == 1:
            y_true_clone = y_true_clone[y_true_clone != rem]
        res = precision_at_k(y_true, y_pred, i) * res

        ap = ap + res
    return ap / min(k, len(y_true))

def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])