#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:42:24 2018

@author: Rohan
"""
import tensorflow as tf
from dataClearUtils import clean_text
from chatbotTraining import questionswords2int, answersints2word, test_predictions, keep_prob, inputs
import numpy as np
from hyperParam import batch_size


"""********** TEST CHAT THE SEQ2SEQ MODEL RNN ***************
"""
 
 
# load the trained model with and run the chatting session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# convert the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    	#we have eliminated the non frequent words so if those words come into the question then we will use <OUT> Token
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# setting up the chat runtime
while(True):
    question = input("You: ")
    if question == 'buy bro':
        break
    question = convert_string2int(question, questionswords2int)
    
    #apply padding: to make sure this q has same length as the length of the q used while training ie 25 we have trained it for question whith length 25
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    
    #NN only accepts input batches ... ie input in batch format
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('pandaBot: ' + answer)




"""
troubleshoot


ValueError: Variable bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/weights already exists, disallowed. 
Did you mean to set reuse=True in VarScope? Originally defined at:
    
=>    
tf.reset_default_graph()


"""

