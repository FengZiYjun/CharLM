
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from model import charLM
from utilities import *
from collections import namedtuple


def preprocess(word_embed_dim):
    
    word_dict, char_dict = create_word_char_dict("valid.txt", "train.txt", "test.txt")
    num_words = len(word_dict)
    num_char  = len(char_dict)
    char_dict["BOW"] = num_char+1
    char_dict["EOW"] = num_char+2
    char_dict["PAD"] = 0
    
    #  dict of (int, string)
    reverse_word_dict = {value:key for key, value in word_dict.items()}
    max_word_len = max([len(word) for word in word_dict])

    #word_embedding = nn.Embedding(vocab_size, word_embed_dim)
    #char_embedding = nn.Embedding(num_char, char_embedding_dim)

    # Note: detach embedding weights from the auto_grad graph.
    # PyTorch embedding weights are learnable variables by default.
    #char_embedding.weight.requires_grad = False
    #word_embedding.weight.requires_grad = False
    #word_emb_matrix = word_embedding.weight

    objects = {
        "word_dict": word_dict,
        "char_dict": char_dict,
        "reverse_word_dict": reverse_word_dict,
        #"word_embed_matrix": word_embed_matrix,
        "max_word_len": max_word_len
    }
    
    torch.save(objects, "cache/prep.pt")
    print("Preprocess done.")


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def train(net, data, opt):
    
    torch.manual_seed(1024)

    train_input = torch.from_numpy(data.train_input)
    train_label = torch.from_numpy(data.train_label)
    valid_input = torch.from_numpy(data.valid_input)
    valid_label = torch.from_numpy(data.valid_label)

    # [num_seq, seq_len, max_word_len+2]
    num_seq = train_input.size()[0] // opt.lstm_seq_len
    train_input = train_input[:num_seq*opt.lstm_seq_len, :]
    train_input = train_input.view(-1, opt.lstm_seq_len, opt.max_word_len+2)

    num_seq = valid_input.size()[0] // opt.lstm_seq_len
    valid_input = valid_input[:num_seq*opt.lstm_seq_len, :]
    valid_input = valid_input.view(-1, opt.lstm_seq_len, opt.max_word_len+2)

    num_epoch = opt.epochs
    num_iter_per_epoch = train_input.size()[0] // opt.lstm_batch_size
    
    leaning_rate = opt.init_lr
    old_PPL = 100000

    # Log-SoftMax
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):

        ##############  Validation  ####################
        loss_batch = []
        PPL_batch = []
        iterations = valid_input.size()[0] // opt.lstm_batch_size
        
        valid_generator = batch_generator(valid_input, opt.lstm_batch_size)
        vlabel_generator = batch_generator(valid_label, opt.lstm_batch_size*opt.lstm_seq_len)

        for t in range(iterations):
            batch_input = valid_generator.__next__()
            batch_label = vlabel_generator.__next__()

            valid_output = net(to_var(batch_input))
            length = valid_output.size()[0]

            # [num_sample-1, len(word_dict)] vs [num_sample-1]
            valid_loss = criterion(valid_output, to_var(batch_label))

            PPL = torch.exp(valid_loss.data / opt.lstm_seq_len)

            loss_batch.append(float(valid_loss))
            PPL_batch.append(float(PPL))

        PPL = np.mean(PPL_batch)
        print("[epoch {}] valid PPL={}".format(epoch, PPL))
        print("valid loss={}".format(np.mean(loss_batch)))
        print("PPL decrease={}".format(float(old_PPL - PPL)))

        if float(old_PPL - PPL) <= 1.0:
            leaning_rate /= 2
            print("halved lr:{}".format(leaning_rate))

        old_PPL = PPL

        ##################################################

        optimizer  = optim.SGD(net.parameters(), 
                               lr = leaning_rate, 
                               momentum=0.85)

        # split the first dim
        input_generator = batch_generator(train_input, opt.lstm_batch_size)
        label_generator = batch_generator(train_label, opt.lstm_batch_size*opt.lstm_seq_len)

        for t in range(num_iter_per_epoch):
            batch_input = input_generator.__next__()
            batch_label = label_generator.__next__()

            # detach hidden state of LSTM from last batch
            net.repackage_hidden()
            
            output = net(to_var(batch_input))
            # [num_word, vocab_size]
            
            loss = criterion(output, to_var(batch_label))

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
            optimizer.step()
            
            
            if t % 300 == 0:
                print("[epoch {} step {}] train loss={}".format(epoch+1, t+1, float(loss.data)))



    print("Training finished.")



def test(net, data, opt):
    
    test_input = torch.from_numpy(data.test_input)
    test_label = torch.from_numpy(data.test_label)

    num_seq = test_input.size()[0] // opt.lstm_seq_len
    test_input = test_input[:num_seq*opt.lstm_seq_len, :]
    # [num_seq, seq_len, max_word_len+2]
    test_input = test_input.view(-1, opt.lstm_seq_len, opt.max_word_len+2)

    criterion = nn.CrossEntropyLoss()

    #output_list = []
    loss_list = []
    num_hits = 0
    total = 0
    iterations = test_input.size()[0] // opt.lstm_batch_size
    for t in range(iterations):
        test_output = net(to_var(test_input[t*opt.lstm_batch_size:(t+1)*opt.lstm_batch_size]))
        
        total += test_output.size()[0]
        batch_label = test_label[t*opt.lstm_batch_size*opt.lstm_seq_len+1:(t+1)*opt.lstm_batch_size*opt.lstm_seq_len+1]

        test_loss = criterion(test_output, to_var(batch_label)).data
        loss_list.append(test_loss)
        test_predict = torch.max(test_output, dim=1)[1]
        num_hits += torch.sum((batch_label.cuda() == test_predict.data).int())

    test_loss = torch.mean(torch.cat(loss_list), 0)
    accuracy =  float(num_hits) / total
    PPL = torch.exp(test_loss / opt.lstm_seq_len)

    
    print("Final Loss={0:.4f}".format(float(test_loss)))
    print("Accuracy={0:.4f}%".format(100 * float(accuracy)))
    print("Final PPL={0:.4f}".format(float(PPL)))


################################################################

word_embed_dim = 300
char_embedding_dim = 15

if os.path.exists("cache/prep.pt") is False:
    preprocess(word_embed_dim)

objetcs = torch.load("cache/prep.pt")

word_dict = objetcs["word_dict"]
char_dict = objetcs["char_dict"]
reverse_word_dict = objetcs["reverse_word_dict"]
#word_embed_matrix = objetcs["word_embed_matrix"]
max_word_len = objetcs["max_word_len"]
num_words = len(word_dict)

print("word/char dictionary built. Start making inputs.")


if os.path.exists("cache/data_sets.pt") is False:
    train_text = read_data("./train.txt")
    valid_text = read_data("./valid.txt")
    test_text  = read_data("./test.txt")

    train_set = np.array(text2vec(train_text, char_dict, max_word_len))
    valid_set = np.array(text2vec(valid_text, char_dict, max_word_len))
    test_set  = np.array(text2vec(test_text,  char_dict, max_word_len))

    # Labels are next-word index in word_dict with the same length as inputs
    train_label = np.array([word_dict[w] for w in train_text[1:]] + [word_dict[train_text[-1]]])
    valid_label = np.array([word_dict[w] for w in valid_text[1:]] + [word_dict[valid_text[-1]]])
    test_label  = np.array([word_dict[w] for w in test_text[1:]] + [word_dict[test_text[-1]]])

    category = {"tdata":train_set, "vdata":valid_set, "test": test_set, 
                "trlabel":train_label, "vlabel":valid_label, "tlabel":test_label}
    torch.save(category, "cache/data_sets.pt") 
else:
    data_sets = torch.load("cache/data_sets.pt")
    train_set = data_sets["tdata"]
    valid_set = data_sets["vdata"]
    test_set  = data_sets["test"]
    train_label = data_sets["trlabel"]
    valid_label = data_sets["vlabel"]
    test_label = data_sets["tlabel"]


DataTuple = namedtuple("DataTuple", 
            "train_input train_label valid_input valid_label test_input test_label")
data = DataTuple(train_input=train_set,
                 train_label=train_label,
                 valid_input=valid_set,
                 valid_label=valid_label,
                 test_input=test_set,
                 test_label=test_label)

print("Loaded data sets. Start building network.")



USE_GPU = True
cnn_batch_size = 700

lstm_seq_len = 35  # BPTT for 35 time steps
lstm_batch_size = 20
# cnn_batch_size == lstm_seq_len * lstm_batch_size

net = charLM(char_embedding_dim, 
            word_embed_dim, 
            lstm_seq_len,
            lstm_batch_size,
            num_words,
            len(char_dict),
            max_word_len,
            use_gpu=USE_GPU)

for param in net.parameters():
    nn.init.uniform(param.data, -0.05, 0.05)


Options = namedtuple("Options", ["num_epoch", 
        "cnn_batch_size", "init_lr", "lstm_seq_len",
        "max_word_len", "lstm_batch_size", "epochs"])
opt = Options(num_epoch=25,
              cnn_batch_size=lstm_seq_len*lstm_batch_size,
              init_lr=1.0,
              lstm_seq_len=lstm_seq_len,
              max_word_len=max_word_len,
              lstm_batch_size=lstm_batch_size,
              epochs=2)


print("Network built. Start training.")



'''
try:
    train(net, data, opt)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


torch.save(net.state_dict(), "cache/model.pt")
print("Model saved.")
'''

net.load_state_dict(torch.load("cache/model.pt"))
test(net, data, opt)



