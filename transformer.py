from ntpath import join
import pickle
import random
from tqdm import tqdm
from config import MAX_SEQUENCE_LENGTH
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from utils import *
import math
from config import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_length = MAX_SEQUENCE_LENGTH
VOCAB_SIZE = 0
min_article_count = 1

SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'

SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3


class Vocab:
    def __init__(self):
        self.index2article = {
            SOS_idx: SOS_token,
            EOS_idx: EOS_token,
            UNK_idx: UNK_token,
            PAD_idx: PAD_token
        }
        self.article2index = {v: k for k, v in self.index2article.items()}

    def index_articles(self, articles):
        for article in articles:
            self.index_article(article)

    def index_article(self, article):
        global VOCAB_SIZE
        if article not in self.article2index:
            n_articles = len(self)
            VOCAB_SIZE = n_articles
            self.article2index[article] = n_articles
            self.index2article[n_articles] = article

    def get_article_features(self, path):

        self.article_features = pickle.load(open(path, 'rb')) 
        article_features_len = len(self.article_features[0]) - 1
        self.article_features = dict(zip(self.article_features[:, 0], self.article_features[:, 1:]))
        self.article_features_size = article_features_len
        for i in [SOS_token, EOS_token, UNK_token, PAD_token]:
            self.article_features[i] = [0] * article_features_len
        article_features = []
        # print(f"Len of ia {len(self.index2article.keys())}, len of af {len(self.article_features.keys())}")
        for article_index in self.index2article.keys():
            article_id = self.index2article[article_index]
            if article_id not in [SOS_token, EOS_token, UNK_token, PAD_token]:
                article_id = int(article_id)
            article_features.append(self.article_features[article_id])
        self.article_features = article_features

    def index_article_features(self, index):
        article_id = self.index2article(index)
        return self.article_features[article_id]
    
    def get_customer_features(self, path):
        self.customer_features = pickle.load(open(path, 'rb')) 
        self.customer_features = dict(zip(self.customer_features[:, 0], self.customer_features[:, 1:-1]))

    def customer_feature(self, customer_id):
        return self.customer_features[customer_id]

    def __len__(self):
        assert len(self.index2article) == len(self.article2index)
        return len(self.index2article)

    def unidex_articles(self, indices):
        return [self.index2article[i] for i in indices]

    def to_file(self, filename):
        values = [str(w) for w, k in sorted(list(self.article2index.items())[4:])]
        with open(filename, 'w') as f:
            f.write('\n'.join(values))

    # @classmethod
    def from_file(self, filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            words = [l.strip() for l in f.readlines()]
            vocab.index_articles(words)
        return vocab



def preprocess_corpus(trans, min_article_count):
    n_articles = {}

    cus_ids = []
    source_trans = []
    target_trans = []
    customer_ids = trans.keys()
    for customer_id in tqdm(customer_ids, desc="Counting the article frequency..."):
        tran = trans[customer_id]
        tran = np.hstack(tran)
        for article in tran:
            if article in n_articles:
                n_articles[article] += 1
            else:
                n_articles[article] = 1

    
    for customer_id in tqdm(customer_ids, desc="Removing rare articles..."):
        sessions = trans[customer_id]
        tran = []
        for session in sessions:
            session = [art if n_articles[art] >= min_article_count else UNK_idx for art in session]
            if len(session) > max_length:
                session = session[-12: ]
            # trans[customer_id][idx] = session
            tran.append(session)
        cus_ids.extend([customer_id] * (len(tran) - 1))
        source_trans.extend(tran[:-1])
        target_trans.extend(tran[1:])
    total = len(n_articles.keys())
    filtered = len(set(np.hstack(np.append(source_trans, target_trans))))
    assert len(source_trans) == len(target_trans), "Source and Target must match in size!!!"
    assert len(source_trans) == len(cus_ids), "The number of customer ids is not match!!!"
    print(f"Keep {filtered} articles from {total} articles")
    print(f"Preprocessing corpus done with {len(source_trans)} corpus")
    return cus_ids, source_trans, target_trans


def read_vocab(source, target):
    vocab = Vocab()
    for tran in tqdm(source, "Reading vocab for source"):
        vocab.index_articles(tran)

    for tran in tqdm(target, "Reading vocab for target"):
        vocab.index_articles(tran)
    return vocab

 
def indexes_from_transaction(vocab, transaction):
    return [vocab.article2index[article] for article in transaction]

def tensor_from_transaction(vocab, transaction, max_seq_length):
    indexes = indexes_from_transaction(vocab, transaction)
    if len(indexes) > max_seq_length - 2:
        indexes = indexes[-(max_seq_length - 2):]
    indexes.append(EOS_idx)
    indexes.insert(0, SOS_idx)
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor

def tensor_from_pair(vocab, source, target, max_seq_length):
    source_tensor = tensor_from_transaction(vocab, source, max_seq_length).unsqueeze(1)
    target_tensor = tensor_from_transaction(vocab, target, max_seq_length).unsqueeze(1)
    return source_tensor, target_tensor



def batch_generator(batch_indices, batch_size):
    batches = math.ceil(len(batch_indices)/batch_size)
    for i in range(batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]
    

def prepare_data(data_dir="datasets_transformer", save_data_dir="saved_dir"):
    max_seq_length = MAX_SEQUENCE_LENGTH + 2
    transactions = pickle.load(open(data_dir + '/customer_sequences.pkl', 'rb'))
    customer_ids, source, target = preprocess_corpus(transactions, 1)
    vocab = read_vocab(source, target)
    vocab.to_file(os.path.join(save_data_dir, 'vocab.txt'))
    global VOCAB_SIZE
    VOCAB_SIZE = len(vocab.article2index)
    print('Corpus length: {}\nVocabulary size: {}'.format(
        len(source), len(vocab.article2index)))

    X_train, X_valid, Y_train, Y_valid = train_test_split(source, target, test_size=0.2, random_state=42)

    # customer_id_train = X_train[:, 1]
    # X_train = X_train[:, 0]
    # customer_id_valid = X_valid[:, 1]
    # X_valid = X_valid[:, 0]

    training_pairs = []
    for source_session, target_session in zip(X_train, Y_train):
        training_pairs.append(tensor_from_pair(vocab, source_session, target_session, max_seq_length))
    
    X_train, Y_train = zip(*training_pairs)
    X_train = torch.cat(X_train, dim=-1)
    Y_train = torch.cat(Y_train, dim=-1)
    torch.save(X_train, os.path.join(save_data_dir, 'X_train.bin'))
    torch.save(Y_train, os.path.join(save_data_dir, 'Y_train.bin'))
    # torch.save(customer_id_train, os.path.join(save_data_dir, 'customer_ids_train.bin'))

    valid_pairs = []
    for source_session, target_session in zip(X_valid, Y_valid):
        valid_pairs.append(tensor_from_pair(vocab, source_session, target_session, max_seq_length))      

    X_valid, Y_valid = zip(*valid_pairs)
    X_valid = torch.cat(X_valid, dim=-1)
    Y_valid = torch.cat(Y_valid, dim=-1)
    torch.save(X_valid, os.path.join(save_data_dir, 'X_valid.bin'))
    torch.save(Y_valid, os.path.join(save_data_dir, 'Y_valid.bin'))
    # torch.save(customer_id_valid, os.path.join(save_data_dir, 'customer_ids_valid.bin'))


    return X_train, Y_train, X_valid, Y_valid

def train_epoch(model, optimizer, loss_fn, batch_size, X_train, Y_train, article_features, epoch, saved_data_dir="saved_data_dir"):
    model.train()
    total_batches = int(X_train.shape[1]/batch_size) + 1
    # print(total_batches)
    indices = list(range(X_train.shape[1]))
    if epoch > 0:
        random.shuffle(indices)
    losses = 0
    step = 1
    batch_loss = 10
    t = tqdm(enumerate(batch_generator(indices, batch_size)),
                            desc=f'Training epoch {epoch+1} - step {step} - loss {batch_loss}',
                        total=total_batches)
    try:
        for step, batch in t:
            src = X_train[:, batch]
            torch.save(src, 'src.bin')
            src_features = [article_features[i] for i in src]
            # y for teacher forcing is all sequence without a last element
            # y_tf = Y_train[batch, :-1]
            # y for loss calculation is all sequence without a last element
            tgt = Y_train[:, batch]
            tgt_features =  [article_features[i] for i in tgt]
            src = torch.tensor(src, dtype=torch.long).to(DEVICE)
            tgt = torch.tensor(tgt, dtype=torch.long).to(DEVICE)
            src_features = torch.tensor(src_features, dtype=torch.double).to(DEVICE)
            tgt_features = torch.tensor(tgt_features, dtype=torch.double).to(DEVICE)
            tgt_input = tgt[:-1, :]
            tgt_features = tgt_features[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            # print("In main ", src_mask.size(), tgt_mask.size(), src_padding_mask.size(), tgt_padding_mask.size())
            logits = model(src, tgt_input, src_features, tgt_features, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            # print("Loss, ", tgt_out.size(), logits.size())
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()
            batch_loss = loss.item() / float(batch_size)
            t.set_description(f'Training epoch {epoch+1} - step {step} - loss {batch_loss}')
    except:
        torch.save(src, "src.bin")
    return losses / X_train.shape[1]

def evaluate(model, loss_fn, X_valid, Y_valid, vocab, article_features):
    model.eval()
    losses = 0
    logits = []
    for src, tgt in zip(X_valid, Y_valid):
        src_features =  [article_features[np.array(i, dtype=np.int32)] for i in src]
        # y for teacher forcing is all sequence without a last element
        # y_tf = Y_train[batch, :-1]
        # y for loss calculation is all sequence without a last element
        tgt_features =  [article_features[np.array(i, dtype=np.int32)] for i in tgt]
        src = torch.tensor(src, dtype=torch.float64).to(DEVICE)
        tgt = torch.tensor(tgt, dtype=torch.float64).to(DEVICE)

        src_features = torch.tensor(src_features, dtype=torch.float64).to(DEVICE)
        tgt_features = torch.tensor(tgt_features, dtype=torch.float64).to(DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_features = tgt_features[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logit = model(src, tgt_input, src_features, tgt_features, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits.append(vocab.unidex_words(logit[1:-1]))
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / X_valid.shape[1], logits


def train_transfomer(X_train, Y_train, X_valid, Y_valid, saved_data_dir):
    torch.manual_seed(0)

    vocab = Vocab()
    vocab = vocab.from_file(saved_data_dir + '/vocab.txt')
    print("Vocab size ", VOCAB_SIZE)
    vocab.get_article_features(cfg['DATA_DIR'] + '/articles_processed.pkl')
    article_features = np.array(vocab.article_features, dtype=np.double)

    # torch.save(article_features, "article_features.bin")
    EMB_SIZE = 512
    NHEAD = 4
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    N_EPOCHS = 100

    best_score = 0.0
    early_stop_after = 10
    early_stop_counter = 0
    best_model = None


    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, VOCAB_SIZE, vocab.article_features_size, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.double()
    transformer = transformer.to(DEVICE)
    

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    for epoch in range(N_EPOCHS):
        start_time = datetime.now()
        train_loss = train_epoch(transformer, optimizer, loss_fn, BATCH_SIZE, X_train, Y_train, article_features, epoch)
        end_time = datetime.now()
        val_loss, logits = evaluate(transformer, loss_fn, X_valid, Y_valid, vocab, article_features)
        map12 = mean_average_precision(Y_valid, logits)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, MAP@12: {map12} "f"Epoch time = {(end_time - start_time):.3f}s"))
            # Early Stop
        if map12 > best_score:
            early_stop_counter = 0
            print('The best model is found, resetting early stop counter.')
            best_score = map12
            best_model = transformer
        else:
            early_stop_counter += 1
            print('No improvements for {} epochs.'.format(early_stop_counter))
            if early_stop_counter >= early_stop_after:
                print('Early stop!')
                torch.save(best_model, os.path.join(saved_data_dir, 'best_epoch.pth'))
                break    

class CustomerEmbedding(nn.Module):
    def __init__(self, input_size, emb_size):
        super(CustomerEmbedding, self).__init__()
        self.emb_size = emb_size

        self.fc = nn.Linear(input_size, self.emb_size)
        # self.out = nn.Linear(hidden_size, hidden_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: Tensor):
        output = self.fc(input)
        # output = output * transaction_encoder_output
        # output = self.out(output)
        # output = self.softmax(output)
        return output

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # print("Token size ", tokens.size())
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) 



class ArticleEmbedding(nn.Module):
    def __init__(self, article_features_size, emb_size, dropout=0.1):
        super(ArticleEmbedding, self).__init__()
        self.emb_size = emb_size
        self.article_features_size = article_features_size
        self.article_emb = nn.Linear(article_features_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        # self.out = nn.Linear(hidden_size, hidden_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tok_emb: Tensor, article_features: Tensor):
        # print(self.article_features_size, tok_emb.size(), article_features.size())
        output = self.article_emb(article_features)
        # output = output * transaction_encoder_output
        # output = self.out(output)
        # output = self.softmax(output)
        
        return self.dropout(output + tok_emb)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                num_encoder_layers: int,
                num_decoder_layers: int,
                emb_size: int,
                nhead: int,
                n_articles: int,
                article_feature_dim: int,
                dim_feedforward=512,
                dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)
        self.generator = nn.Linear(emb_size, n_articles)
        self.token_emb = TokenEmbedding(n_articles, emb_size)
        # self.src_tok_emb = TokenEmbedding(n_articles, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(n_articles, emb_size)
        self.article_emb = ArticleEmbedding(article_feature_dim, emb_size)

    def forward(self, 
                src: Tensor,
                tgt: Tensor,
                src_feat: Tensor,
                tgt_feat: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.token_emb(src)
        tgt_emb = self.token_emb(tgt)
        # print("In S2S", src.size(), tgt.size(), src_emb.size(), tgt_emb.size())
        src_emb = self.article_emb(src_emb, src_feat)
        tgt_emb = self.article_emb(tgt_emb, tgt_feat)
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        output = self.generator(output)
        return output

    def encode(self, src: Tensor, src_mask: Tensor):
        output = self.transformer.encoder(self.article_emb(self.token_emb(src)), src_mask)
        return output

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        output = self.transformer.decoder(self.article_emb(self.token_emb(tgt)), memory, tgt_mask)
        return output

