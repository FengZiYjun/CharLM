
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class charLM(nn.Module):
    def __init__(self, char_emb_dim, word_emb_dim, 
                lstm_seq_len, lstm_batch_size, vocab_size,
                use_gpu):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.lstm_seq_len = lstm_seq_len
        self.lstm_batch_size = lstm_batch_size
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,           # in_channel
                    out_channel, # out_channel
                    kernel_size=(self.char_emb_dim, filter_width), # (height, width)
                    bias=True
                    )
            )

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
                            dropout=0.5)

        # output layer
        self.emb_bias = Variable(torch.zeros(self.vocab_size))
        self.dropout = nn.Dropout(p=0.5)

        if self.use_gpu is True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
            self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
            self.lstm = self.lstm.cuda()
            self.emb_bias = self.emb_bias.cuda()
            self.dropout = self.dropout.cuda()


    def forward(self, x, word_emb):
        x = self.conv_layers(x)
        x = self.highway_layers(x)

        cnn_batch_size = x.size()[0]
        lstm_batch_size = cnn_batch_size // self.lstm_seq_len
        output, self.hidden = self.lstm(x.view(self.lstm_seq_len, lstm_batch_size, -1), self.hidden)
        
        output = self.dropout(torch.transpose(output, 0, 1))
        # (batch, seq_len, hidden_size)
        batch = [] 

        for x in range(output.size()[0]):
            y = torch.mm(output[x], torch.transpose(word_emb, 0, 1)) + self.emb_bias
            batch.append(y)
        # (batch*seq_len, vocab_size)
        return F.softmax(torch.cat(batch, 0), dim=1) 


    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (cnn_batch_size, out_channel, 1, width)            
            chosen = torch.max(feature_map, 3)[0]
            # (cnn_batch_size, out_channel, 1)            
            chosen = chosen.view(chosen.size()[0], -1)
            # (cnn_batch_size, out_channel)
            chosen_list.append(chosen)
            
        return torch.cat(chosen_list, 1)

    def highway_layers(self, y):
        transform_gate = F.sigmoid(self.fc1(y))
        carry_gate = 1 - transform_gate
        return transform_gate * F.relu(self.fc2(y)) + carry_gate * y

    def repackage_hidden(self):
        self.hidden = (Variable(self.hidden[0].data, requires_grad=True), 
                        Variable(self.hidden[1].data, requires_grad=True))
