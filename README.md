
# PyTorch-Character-Aware-Neural-Language-Model

This is the PyTorch implementation of character-aware neural language model proposed in the paper ![Character-aware Neural Language Model()](https://arxiv.org/abs/1508.06615).

## Requiredments
The code is run and tested with Python 3.5 and PyTorch 0.3.1.

## parameter settings
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
Test the model.
`python test.py`

PPl=131.4480

## Acknowledgement 



