
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import model
from utilities import *

"""
Issues: 

"""

def preprocess():
    # vocabulary == dict of (string, int)
    vocabulary, char_table = get_vocab_and_char_table("valid.txt", "train.txt")

    # reverse_vocab == dict of (int, string)
    reverse_vocab = {value:key for key, value in vocabulary.items()}

    vocab_size = len(vocabulary)
    num_char = len(char_table)
    word_embedding_dim = 300
    char_embedding_dim = 15
    word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
    char_embedding = nn.Embedding(num_char, char_embedding_dim)

    # Note: detach embedding weights from the auto_grad graph.
    # PyTorch embedding weights are learnable variables by default.
    char_embedding.weight.requires_grad = False
    word_embedding.weight.requires_grad = False
    word_emb_matrix = word_embedding.weight

    torch.save(word_emb_matrix, "cache/word_emb_matrix.pt") 
    torch.save(char_embedding, "cache/char_embedding.pt")
    torch.save(char_table, "cache/char_table.pt")
    torch.save(vocabulary, "cache/vocabulary.pt")
    torch.save(reverse_vocab, "cache/reverse_vocab.pt")


word_embedding_dim = 300
char_embedding_dim = 15

if os.path.exists("cache/word_emb_matrix.pt") is False:
    preprocess()

word_emb_matrix = torch.load("cache/word_emb_matrix.pt")
char_embedding = torch.load("cache/char_embedding.pt")
char_table = torch.load("cache/char_table.pt")
vocabulary = torch.load("cache/vocabulary.pt")
reverse_vocab =torch.load("cache/reverse_vocab.pt")
vocab_size = len(vocabulary)
num_char = len(char_table)


print("Embedding built. Start building network.")


USE_GPU = True
cnn_batch_size = 700

lstm_seq_len = 35  # BPTT for 35 time steps
lstm_batch_size = 20
# cnn_batch_size == lstm_seq_len * lstm_batch_size

net = model.charLM(char_embedding_dim, 
                   word_embedding_dim, 
                   lstm_seq_len,
                   lstm_batch_size,
                   vocab_size,
                   use_gpu=USE_GPU)

for param in net.parameters():
    nn.init.uniform(param.data, -0.05, 0.05)


print("Network built.")


def train():
    
    torch.manual_seed(1024)

    # list of strings
    input_words = read_data("./train.txt")
    valid_set = read_data("./valid.txt")

    global word_emb_matrix
    global net

    if os.path.exists("cache/train_X.pt"):
        X = torch.load("cache/train_X.pt")
        valid_X = torch.load("cache/valid_X.pt")
    else:
        X = seq2vec(input_words, char_embedding, char_embedding_dim, char_table)
        X = X.unsqueeze(0)
        X = torch.transpose(X, 0, 1)

        valid_X = seq2vec(valid_set, char_embedding, char_embedding_dim, char_table).unsqueeze(0)
        valid_X = torch.transpose(valid_X, 0, 1)
        
        torch.save(X, "cache/train_X.pt")
        torch.save(valid_X, "cache/valid_X.pt")
    



    if USE_GPU is True and torch.cuda.is_available():
        X = X.cuda()
        valid_X = valid_X.cuda()
        net = net.cuda()
        word_emb_matrix = word_emb_matrix.cuda()
        torch.cuda.manual_seed(1024)


    num_epoch = 1  # 25 epochs in the paper
    num_iter_per_epoch = X.size()[0] // cnn_batch_size
    
    print("Start training.")
    
    valid_generator = batch_generator(valid_X, cnn_batch_size)
    leaning_rate = 0.001

    old_PPL = 0

    for epoch in range(num_epoch):
    
        input_generator = batch_generator(X, cnn_batch_size)

        batch_valid = valid_generator.__next__()
        output_valid = net(batch_valid, word_emb_matrix)
        output_valid = torch.transpose(output_valid, 0, 1)

        loss_valid = get_loss(output_valid, valid_set, vocabulary, cnn_batch_size, epoch)
        PPL = torch.exp(loss_valid / lstm_seq_len)
        print("[epoch {}] PPL={}".format(epoch, PPL.data))

        if old_PPL == 0:
            old_PPL = PPL
        else:
            print("PPL decrease={}".format(old_PPL - PPL))
            if old_PPL - PPL <= 1.0:
                leaning_rate /= 2
                print("halved learning rate")


        optimizer  = optim.SGD(net.parameters(), 
                               lr = leaning_rate, 
                               momentum=0.85)

        
        for t in range(num_iter_per_epoch):
            batch_input = input_generator.__next__()
            
            # detach hidden state of LSTM from last batch
            net.repackage_hidden()
            
            output = net(batch_input, word_emb_matrix)
            # [num_word, vocab_size]
            output = torch.transpose(output, 0, 1)
            
            #distribution = get_distribution(output, word_emb_matrix)
            loss = get_loss(output, input_words, vocabulary, cnn_batch_size, t)

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
            optimizer.step()
            
            
            if t % 300 == 0:
                
                output_valid = net(batch_valid, word_emb_matrix)             
                output_valid = torch.transpose(output_valid, 0, 1)
                loss_valid = get_loss(output_valid, valid_set, vocabulary, cnn_batch_size, epoch)
                PPL = torch.exp(loss_valid / lstm_seq_len)

                print("[epoch {} step {}] \n\ttrain loss={}".format(epoch+1, t+1, loss.data))
                print("\tvalid loss={}".format(loss_valid.data))
                print("\tPPL={}".format(PPL.data))



    print("Training finished.")



def test():
    text_words = read_data("./valid.txt")
    
    if os.path.exists("cache/test_X.pt"):
        X = torch.load("cache/test_X.pt")
    else:
        X = seq2vec(text_words, char_embedding, char_embedding_dim, char_table)
        X = X.unsqueeze(0)
        X = torch.transpose(X, 0, 1)    
        torch.save(X, "cache/test_X.pt")
    

    global net
    global word_emb_matrix

    if USE_GPU is True and torch.cuda.is_available():
        X = X.cuda()
        net = net.cuda()
        word_emb_matrix = word_emb_matrix.cuda()
        torch.cuda.manual_seed(1024)


    num_iter = X.size()[0] // cnn_batch_size
    generator = batch_generator(X, cnn_batch_size)

    predict_words = []
    predict_ix = []

    for t in range(num_iter):
        batch_input = generator.__next__()

        output = net(batch_input, word_emb_matrix)
        output = torch.transpose(output, 0, 1)
        # [vocab_size, num_words]
        
        # LongTensor of [num_words]
        _, targets = torch.max(output, 0)
        
        predict_ix += list(targets)

    predict_ix = torch.cat(predict_ix, 0)
    length = int(predict_ix.size()[0])
    
    
    ix_list = [vocabulary[text_words[ix]] for ix in range(1, length+1)]

    truth_ix = torch.LongTensor(ix_list)
    truth_ix = Variable(truth_ix, requires_grad=False)
    if USE_GPU is True and torch.cuda.is_available():
        truth_ix = truth_ix.cuda()

    tmp = predict_ix == truth_ix
    accuracy = torch.sum(tmp.int()) / length
    
    print("Accuracy={}%".format(100 * accuracy.data))



try:
    train()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    torch.save(net.state_dict(), "cache/model.pt")

test()



