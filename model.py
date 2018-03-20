
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
2. two layers of highways (remain 1)
3. Use sequential to define combined layers (not urgent)
"""


class charLM(nn.Module):
    """CNN + highway network + LSTM
    # Input: 
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        lstm_seq_len: num of words in a LSTM sequence / num of time steps that LSTM goes through
        lstm_batch_size: num of sequences in a "LSTM" batch
        vocab_size: num of unique words
        num_char: total number of characters in all data sets
        use_gpu: True or False
    """
    def __init__(self, char_emb_dim, word_emb_dim, 
                lstm_seq_len, lstm_batch_size, 
                vocab_size, num_char,
                use_gpu):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.lstm_seq_len = lstm_seq_len
        self.lstm_batch_size = lstm_batch_size
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu

        # char embedding layer
        self.char_embed = nn.Embedding(num_char, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        
        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,           # in_channel
                    out_channel, # out_channel
                    kernel_size=(self.char_emb_dim, filter_width), # (height, width)
                    bias=True
                    )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.fc1 = nn.Linear(self.highway_input_dim, self.highway_input_dim, bias=True)
        self.fc2 = nn.Linear(self.highway_input_dim, self.highway_input_dim, bias=True)

        # LSTM
        self.lstm_num_layers = 2
        # num of hidden units == word_embedding_dim
        self.hidden = (Variable(torch.zeros(self.lstm_num_layers, 
                                            self.lstm_batch_size, 
                                            self.word_emb_dim)),
                       Variable(torch.zeros(self.lstm_num_layers, 
                                            self.lstm_batch_size, 
                                            self.word_emb_dim))
                       )

        self.lstm = nn.LSTM(input_size=self.highway_input_dim, 
                            hidden_size=self.word_emb_dim, 
                            num_layers=self.lstm_num_layers,
                            bias=True,
                            dropout=0.5,
                            batch_first=True)

        # output layer
        #self.emb_bias = Variable(torch.zeros(self.vocab_size))
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

        
        if self.use_gpu is True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
            self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
            self.lstm = self.lstm.cuda()
            self.dropout = self.dropout.cuda()
            self.char_embed = self.char_embed.cuda()
            self.linear = self.linear.cuda()
            self.batch_norm = self.batch_norm.cuda()
        

    def forward(self, x):
        # Input: Variable of Tensor with shape [num_seq, seq_len, max_word_len+2]
        # Return: Variable of Tensor with shape [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]
        
        x = self.char_embed(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]
        
        x = x.view(x.size()[0], 1, x.size()[1], -1)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]
        
        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway_layer(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.contiguous().view(self.lstm_seq_len, lstm_batch_size, -1)
        # [num_seq, seq_len, total_num_filters]
        
        x, self.hidden = self.lstm(x, self.hidden)
        # [num_seq, seq_len, hidden_size]
        
        x = self.dropout(x)
        # [num_seq, seq_len, hidden_size]
        
        x = x.contiguous().view(lstm_batch_size*lstm_seq_len -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]
        return x


    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)            
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)
        
        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)

    def highway_layer(self, y):
        transform_gate = F.sigmoid(self.fc1(y))
        carry_gate = 1 - transform_gate
        return transform_gate * F.relu(self.fc2(y)) + carry_gate * y

    def repackage_hidden(self):
        self.hidden = (Variable(self.hidden[0].data, requires_grad=True), 
                        Variable(self.hidden[1].data, requires_grad=True))
