import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from config import *
from dataset import FashionDataset
import warnings
warnings.filterwarnings('ignore')



# fashion_dataset = FashionDataset(DATA_DIR, 'train_indexes.pkl')
# fashion_dataset.article_processing()
# print("Article processed.")
# fashion_dataset.customer_processing()
# print("Customer processed.")
# fashion_dataset.create_dataset()
# print("Processed.")
