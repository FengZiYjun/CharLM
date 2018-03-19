"""
 Functions used in training
"""


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



def batch_generator(x, batch_size):
    # x: [num_words, in_channel, height, width]
    # partitions x into batches
    num_step = x.size()[0] // batch_size
    for t in range(num_step):
        yield x[t*batch_size:(t+1)*batch_size]


def get_distribution(output, word_emb_matrix):
    word_distributions = []
    for x in range(output.size()[0]):  # for each batch
        for y in range(output.size()[1]):  # for each sequence
            dist = torch.sum(output[x][y].view(1, 300) * word_emb_matrix, 1)
            dist = F.softmax(dist, dim=0)
            word_distributions.append(dist.unsqueeze(0))

    # (vocab_size, num_words)
    return torch.transpose(torch.cat(word_distributions, 0), 0, 1)



def get_loss(output, input_words, vocabulary, cnn_batch_size, batch_no, lstm_seq_len):
    """ Sum up log prob for a sequence. And average all sequences. """
    # output == Tensor of [vocab_size, batch_size], the predicted distribution
    # input_words == list of strings, the raw text
    # vocabulary == dict of (string: int)
    # cnn_batch_size == int, the size of each batch durng iterations
    # batch_no == int, denotes the i-th batch
    # Return: loss == FloatTensor of size [1]
    prediction = []
    batch_begin_ix = batch_no * cnn_batch_size
    for ix in range(cnn_batch_size):
        next_ix = batch_begin_ix + ix + 1
        if next_ix == len(input_words): # the last word
            next_ix -= 1
        next_word_ix = vocabulary[input_words[next_ix]]
        prediction.append(output[next_word_ix][ix])
    all_loss = torch.log(torch.cat(prediction, 0)).view(-1, lstm_seq_len)
    loss = torch.mean(-torch.sum(all_loss, 1))
    return loss


def char_embedding_lookup(word, char_embedding, char_table):
    # use the simplest lookup method. change later.
    vec = [char_table.index(s) for s in word]
     # [len(vec), char_embedding_dim]
    encoded_word = char_embedding(Variable(torch.LongTensor(vec), requires_grad=False)).data
    return torch.transpose(encoded_word, 0, 1)


def seq2vec(input_words, char_embedding, char_embedding_dim, char_table):
    """ convert the input strings into character embeddings """
    # input_words == list of string
    # char_embedding == torch.nn.Embedding
    # char_embedding_dim == int
    # char_table == list of unique chars
    # Returns: tensor of shape [len(input_words), char_embedding_dim, max_word_len+2]
    max_word_len = max([len(word) for word in input_words])
    tensor_list = []
    
    #start_column = Variable(torch.ones(char_embedding_dim, 1))
    #end_column = Variable(torch.ones(char_embedding_dim, 1))
    start_column = torch.ones(char_embedding_dim, 1)
    end_column = torch.ones(char_embedding_dim, 1)

    
    for word in input_words:
        # convert string to word embedding
        word_encoding = char_embedding_lookup(word, char_embedding, char_table)
        # add start and end columns
        word_encoding = torch.cat([start_column, word_encoding, end_column], 1)
        # zero-pad right columns
        word_encoding = F.pad(word_encoding, (0, max_word_len-word_encoding.size()[1]+2)).data
        # create dimension
        word_encoding = word_encoding.unsqueeze(0)

        tensor_list.append(word_encoding)

    return torch.cat(tensor_list, 0)


def read_data(file_name):
    # Return: list of strings
    with open(file_name, 'r') as f:
        corpus = f.read().lower()
    import re
    corpus = re.sub(r"<unk>", "unk", corpus)
    return corpus.split()


def create_char_table(vocabulary):
    # vocabulary == list of strings, all unique words
    # Return: list of unique chars
    return list(set([char for word in vocabulary for char in word]))


def get_vocab_and_char_table(*file_name):
    text = []
    for file in file_name:
        text += read_data(file)
    vocabulary = {word:ix for ix, word in enumerate(set(text))}
    char_table = create_char_table(vocabulary)
    return vocabulary, char_table

