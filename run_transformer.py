from tqdm import tqdm
from transformer import *
import os
from os.path import join

def train_transformer(data_dir):
    # if os.path.exists(data_dir):
    #     X_train = torch.load(join(data_dir, "X_train.bin"))
    #     Y_train = torch.load(join(data_dir, "Y_train.bin"))
    #     X_valid = torch.load(join(data_dir, "X_valid.bin"))
    #     Y_valid = torch.load(join(data_dir, "Y_valid.bin"))
    # else:
    #     X_train, Y_train, X_valid, Y_valid = prepare_data(data_dir)
    X_train, Y_train, X_valid, Y_valid = prepare_data(data_dir)
    train_transfomer(X_train, Y_train, X_valid, Y_valid)