from sklearn.utils import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FashionDataset
from encoder_model import TransactionsEncoder, CustomerEncoder, AttnDecoder

import os
import pickle
import time
import math
import random
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Training:
    def __init__(self, config):
        self.config = config
        pass

    def train(self, transaction_tensor, customer_tensor, target_tensor, \
                transaction_encoder, customer_encoder, decoder,\
                transaction_encoder_optimizer, customer_encoder_optimizer, decoder_optimizer, \
                criterion, length):
        transaction_encoder_hidden = transaction_encoder.initHidden(self.config['BATCH_SIZE']).to(device)
        transaction_encoder_optimizer.zero_grad()
        customer_encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # input_length = transaction_tensor.size(1)
        # input_length = length[0].item()
        # max_length = self.config['MAX_SEQUENCE_LENGTH']
        # transaction_encoder_outputs = torch.zeros(transaction_tensor.size(0), max_length, transaction_encoder.hidden_size, device=device)
        # print(input_length)
        # for ei in range(input_length):
        #     print(transaction_tensor.size(), transaction_encoder_hidden.size())
        #     encoder_output, transaction_encoder_hidden = transaction_encoder(transaction_tensor[:, ei, :], transaction_encoder_hidden)
        #     transaction_encoder_outputs[:, ei, :] = encoder_output[:, 0, 0]

        transaction_encoder_outputs, transaction_encoder_hidden = transaction_encoder(transaction_tensor, transaction_encoder_hidden)

        customer_encoder_output = customer_encoder(customer_tensor)

        # print(customer_encoder_output.size(), transaction_encoder_outputs.size(), transaction_encoder_hidden.size())
        decoder_output, decoder_attention = decoder(
            customer_encoder_output, transaction_encoder_outputs, transaction_encoder_hidden)

        # print(decoder_output.size(), target_tensor.size())
        # print(decoder_output, target_tensor)
        # target_tensor = target_tensor.long()
        loss = criterion(decoder_output, target_tensor)

        loss.backward()

        transaction_encoder_optimizer.step()
        customer_encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() 



    def run_epochs(self, transaction_encoder, customer_encoder, decoder, print_every=1000, learning_rate=0.01):
        # start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        transaction_encoder_optimizer = optim.SGD(transaction_encoder.parameters(), lr=learning_rate)
        customer_encoder_optimizer = optim.SGD(customer_encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        with open(os.path.join(self.config['DATA_DIR'], 'all_samples.pkl'), 'rb') as f:
            samples = pickle.load(f)
        print("Loading train data...")
        train_dataset = FashionDataset(self.config, self.config['DATA_DIR'], samples, 'train')
        print("Loading valid data...")
        valid_dataset = FashionDataset(self.config, self.config['DATA_DIR'], samples, 'valid')
        train_loader = DataLoader(train_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False, num_workers=2) 
        # valid_dataset = [1, 2, 3]
        print(f"Loaded {len(train_dataset)} train data and {len(valid_dataset)} valid data")
        # criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        total_batches = len(train_dataset) // self.config['BATCH_SIZE']
        for epoch in range(self.config['EPOCHS']):    
            # start_1 = datetime.now()
            start = datetime.now()
            for i, data in tqdm(enumerate(train_loader, 0)):
                # print("Load data time ", 
                # datetime.now() - start_1)
                # start_2 = datetime.now()
                sequence_tensor = data['sequence_features']
                customer_tensor = data['customer_features']
                target_tensor = data['target']

                sequence_tensor = sequence_tensor.float().to(device)
                customer_tensor = customer_tensor.float().to(device)
                target_tensor = target_tensor.long().to(device)
                length = data['length']
                
                loss = self.train(sequence_tensor, customer_tensor, target_tensor, 
                            transaction_encoder, customer_encoder, decoder,
                            transaction_encoder_optimizer, customer_encoder_optimizer, decoder_optimizer,
                            criterion, length)
                print_loss_total += loss
                plot_loss_total += loss
                # print("Runing time ", datetime.now() - start_2)
                if i % print_every == 0 and i > 0:
                    running_time = datetime.now() - start
                    start = datetime.now()
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print(f'[{epoch + 1}, {i + 1:5d} / {total_batches} {running_time}] loss: {print_loss_avg:.3f}')
                    
                # start_1 = datetime.now()
                # if i % plot_every == 0:
                #     plot_loss_avg = plot_loss_total / plot_every
                #     plot_losses.append(plot_loss_avg)
                #     plot_loss_total = 0
              
            torch.save(transaction_encoder.state_dict(), os.path.join(self.config['MODELS_PATH'], 'transaction_models', 'epoch_' + str(epoch) + '.pth'))
            torch.save(customer_encoder.state_dict(), os.path.join(self.config['MODELS_PATH'], 'customer_models', 'epoch_' + str(epoch) + '.pth'))
            torch.save(decoder.state_dict(), os.path.join(self.config['MODELS_PATH'], 'decoder_models', 'epoch_' + str(epoch) + '.pth'))
            
            # Evaluating
            self.run_evaluate(transaction_encoder, customer_encoder, decoder, valid_loader, epoch)

        # showPlot(plot_losses)

    def run_evaluate(self, transaction_encoder, customer_encoder, decoder, valid_loader, epoch):
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            total = 0
            correct = 0
            for i, data in tqdm(enumerate(valid_loader)):
                sequence_tensor = data['sequence_features']
                customer_tensor = data['customer_features']
                target_tensor = data['target']
                
                sequence_tensor = sequence_tensor.float().to(device)
                customer_tensor = customer_tensor.float().to(device)
                target_tensor = target_tensor.long().to(device)

                length = data['length']

                predicted, loss = self.evaluate(sequence_tensor, customer_tensor, target_tensor,
                                            transaction_encoder, customer_encoder, decoder, criterion, length)
                total_loss += loss
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                
            accuracy = 100 * correct / total
            print(f'[{epoch + 1}] Loss: {loss:.3f} Accuracy: {accuracy:.3f}')


    def evaluate(self, transaction_tensor, customer_tensor, target_tensor, \
                transaction_encoder, customer_encoder, decoder,\
                criterion, length):
        transaction_encoder_hidden = transaction_encoder.initHidden(self.config['BATCH_SIZE']).to(device)
        # input_length = transaction_tensor.size(0)
        input_length = length[0].item()
        max_length=self.config['MAX_SEQUENCE_LENGTH']
        transaction_encoder_outputs = torch.zeros(max_length, transaction_encoder.hidden_size, device=device)


        # for ei in range(input_length):
        #     encoder_output, transaction_encoder_hidden = transaction_encoder(transaction_tensor[0][ei], transaction_encoder_hidden)
        #     transaction_encoder_outputs[ei] = encoder_output[0, 0]
        transaction_encoder_outputs, transaction_encoder_hidden = transaction_encoder(transaction_tensor, transaction_encoder_hidden)

        customer_encoder_output = customer_encoder(customer_tensor)

        decoder_output, decoder_attention = decoder(
            customer_encoder_output, transaction_encoder_outputs, transaction_encoder_hidden)
        _, predicted = torch.max(decoder_output.data, 1)

        loss = criterion(decoder_output, target_tensor)
        return predicted, loss
            

    def start_training(self):
        transaction_encoder = TransactionsEncoder(self.config['TRANSACTION_ENCODER_INPUT_SIZE'], self.config['HIDDEN_SIZE']).to(device)
        customer_encoder = CustomerEncoder(self.config['CUSTOMER_ENCODER_INPUT_SIZE'], self.config['HIDDEN_SIZE']).to(device)
        attn_decoder = AttnDecoder(self.config['HIDDEN_SIZE'], self.config['N_CLASSES'], dropout_p=0.1).to(device)
        self.run_epochs(transaction_encoder, customer_encoder, attn_decoder, 100, learning_rate=self.config['LR'])

