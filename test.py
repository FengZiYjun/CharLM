

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


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
    test_generator = batch_generator(test_input, opt.lstm_batch_size)
    label_generator = batch_generator(test_label, opt.lstm_batch_size*opt.lstm_seq_len)

    hidden = (to_var(torch.zeros(2, opt.lstm_batch_size, opt.word_embed_dim)), 
              to_var(torch.zeros(2, opt.lstm_batch_size, opt.word_embed_dim)))
    
    for t in range(iterations):
    	batch_input = test_generator.__next__ ()
    	batch_label = label_generator.__next__()

    	hidden = [state.detach() for state in hidden]
        test_output, hidden = net(to_var(batch_input), hidden)
        
        total += test_output.size()[0]

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


#############################################################


word_embed_dim = 300
char_embedding_dim = 15

if os.path.exists("cache/prep.pt") is False:
    preprocess(word_embed_dim)

objetcs = torch.load("cache/prep.pt")

word_dict = objetcs["word_dict"]
#char_dict = objetcs["char_dict"]
#reverse_word_dict = objetcs["reverse_word_dict"]
#word_embed_matrix = objetcs["word_embed_matrix"]
max_word_len = objetcs["max_word_len"]
num_words = len(word_dict)

print("word/char dictionary built. Start making inputs.")


if os.path.exists("cache/data_sets.pt") is False:
    
    test_text  = read_data("./test.txt")

    test_set  = np.array(text2vec(test_text,  char_dict, max_word_len))

    # Labels are next-word index in word_dict with the same length as inputs
    test_label  = np.array([word_dict[w] for w in test_text[1:]] + [word_dict[test_text[-1]]])

    category = {"test": test_set, "tlabel":test_label}
    torch.save(category, "cache/data_sets.pt") 
else:
    data_sets = torch.load("cache/data_sets.pt")
    test_set  = data_sets["test"]
    test_label = data_sets["tlabel"]


DataTuple = namedtuple("DataTuple", "test_input test_label")
data = DataTuple( test_input=test_set,
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

net.load_state_dict(torch.load("cache/model.pt"))

#for param in net.parameters():
#    nn.init.uniform(param.data, -0.05, 0.05)


Options = namedtuple("Options", ["num_epoch", 
        "cnn_batch_size", "init_lr", "lstm_seq_len",
        "max_word_len", "lstm_batch_size", "epochs",
        "word_embed_dim"])
opt = Options(num_epoch=25,
              cnn_batch_size=lstm_seq_len*lstm_batch_size,
              init_lr=1.0,
              lstm_seq_len=lstm_seq_len,
              max_word_len=max_word_len,
              lstm_batch_size=lstm_batch_size,
              epochs=2,
              word_embed_dim=word_embed_dim)


print("Network built. Start testing.")



test(net, data, opt)
