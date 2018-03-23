
# PyTorch-Character-Aware-Neural-Language-Model

This is the PyTorch implementation of character-aware neural language model proposed in this [paper](https://arxiv.org/abs/1508.06615) by Yoon Kim. 

## Requirements
The code is run and tested with **Python 3.5.2** and **PyTorch 0.3.1**.

## HyperParameters
| HyperParam | value |
| ------ | :-------|
| LSTM batch size | 20 |
| LSTM sequence length | 35 |
| LSTM hidden units | 300 |
| epochs | 35 |
| initial learning rate | 1.0 |
| character embedding dimension | 15 |

## Demo
Train the model with split train/valid/test data.

`python train.py`

The trained model will saved in `cache/net.pkl`.
Test the model.

`python test.py`

This yields PPl=131.4480, cross entropy loss=4.8.

## Acknowledgement 
This implementation borrowed ideas from
https://github.com/jarfo/kchar

https://github.com/cronos123/Character-Aware-Neural-Language-Models


