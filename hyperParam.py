#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:56:36 2018

@author: Rohan
"""


 
# List of Hyperparameters and Why Hyperparameters!!
epochs = 100
#One cycle is Forward propogation and Back propogation: one whole iteration of training.
batch_size = 32
#or 128 can also be used.
rnn_size = 1024
num_layers = 3
#No. of layers in  encoder RNN and deoder RNN
encoding_embedding_size = 1024
#No. of columns in embedding matrix
decoding_embedding_size = 1024
#No. of columns in decoding matrix
learning_rate = 0.001
#It shoudn't be too high or low if its too high then your model will learn too fast and VC it will take ages!
learning_rate_decay = 0.9
#By what % the learning_rate is reduced while learning over iteraton.
#Most of the time 0.9 or 1 ie generally 90%!
min_learning_rate = 0.0001
keep_probability = 0.5
'''
p : neuron is present with certain propbability during training.
keep_prob =  (1 - drop_out_rate) 
Thus, we basically deactivate neurons with prob (1 - p) Note: Only for training !!!
drop_rate is 50% Recommended Value
Hence, keep_prob =  1 - drop_rate = 0.5
'''