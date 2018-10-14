#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 23:30:29 2018

@author: Rohan
"""

import tensorflow as tf


"""********** CONSTRUCT SEQ2SEQ ***************
"""



'''
********** Pit stop 1 ***************

## In tensorflow all variables are used in tensors. Tensors are more advanced than numpy arrays!!

## ** WHY? **
### Provide fastest computation in deep learing
### More advance data structre
## All the variables used in tensors must be defined as tensorflow placeholder.


## Thus, we need to create some placeholders for input and target 

'''


def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    # keep_prob parameter used to control deop out rate: drop out is rate of the neuron you choose to override during one iteration in training ..
    # Usually it is 20 %... you deactiavte 20 % of neuron during training
    
    return inputs, targets, lr, keep_prob


"""
********** Pit stop 2 ***************

So before we start creating the encoding layer and the decoding layer we have to prepare a set of targets.
And that's two things we will do in this tutorial.
Create the batches and then adding the S.O.S token.

"""


def preprocess_targets(targets, word2int, batch_size):
    
    """
    Well for this batch and every other batch the left side of the concatenation will be a vector of 10 elements only containing the S.O.S tokens.
    And more precisely the unique identifiers including the SOS as tokens and the right side of the concatenation will be all the 10 answers in this batch of answers.
    But except the last column because we don't want to get the last token we want needed for the decoder.
    We don't want EOS for decoding.
    
    
    """
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
 
# creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    
    '''
    rnn_inputs = model_inputs function ka values
    rnn_size = number of input tensors inthe layer and not the hidden layer ka number
    num_layers = layer hidden
    keep_prob = we are applying dropout regulization to LSTM .. so we have stacked LSTM to improve accuracy
    sequence_legnth = the list of length of the each q in the batch
    '''
    
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    '''
    to create LSTM cell class. contrib module rnn submodule and then from der
    '''
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    '''
    deactivating the certain percentage of neurons during the training
    usually it is 20 percent ... meaninf their weights are not updated during training
    input1 = rnn network on which you want to apply dropout
    second argument is the parameter of the class
    '''
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    '''
    Create encoder cell:
    Biderectional method return two parameters second is encoder state..and first is encoder output.so we write _, to indicate we just want second parameter by the function 
    '''
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    '''
    So, this dynamic version of directional RNN basically takes your input that we're about to answer right

    now and will build independent forward and backward RNN's but be careful when you build this 

    kind of bidirectional dynamic RNN we have to make sure that the input size of the forward cell and the backwards cell must match.
    '''
    return encoder_state
 
# decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    
    """
    encoder_state = input to proceed to decoding from above function
    decoder_cell = cell in the RNN of the decoder
    decoder_embedded_input = inputs on which we are applying embedding... go to tensorflow website user guide
                             embedding is an mapping from word to vectors ..
                             format decoder is accepting as its input
    decoding_scope =  website ... variable scope class in API section ... an advanced ds that will wrap your tensorflow variables
    output_function = function to return the decoded output
    """
    # 1st the attention state ... initilise them with zero 3D matix
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # attention_keys ... keys to be compared with the target states
    # values used to create context vector. context is returned by the encoder to be used by the decoder as first element for decoding
    # score fucntion: compute similarity between keys and target states
    # construct is used to build the attention state
    
    '''
    training decoder function which will decode the traing set
    '''
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    #second argument is decoder_final_state 
    #third argument is decoder_final_context_state
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    
    """
    We keep separately 10 percent of the data for validation of the model
    We used attention_decoder_fn_train in b4 function 
    But, here we will use : attention_decoder_fn_inference fucntion : to dedude logically 
    Once the chatbot is trained it has logic inside it so we after the training it will asnwer on the new data which we kept 10 per separate
    """
    #sos_id = token id
    #eos_id =     "
    #maximum_length = length of the longestans you can find in the batch 
    #num_words = take the length of dic answerswords2int
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # attention_keys ... keys to be compared with the target states
    # values used to create context vector. context is returned by the encoder to be used by the decoder as first element for decoding
    # score fucntion: compute similarity between keys and target states
    # construct is used to build the attention state
    
    '''
    training decoder function which will decode the test set
    '''
    
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    
    #we need to keep the attention part of nn while testing the test data. to get relevant prediction
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    #drop out not used for test part only for training 
    return test_predictions
 
#  Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #to reduce overfiding and improve the accuracy
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        #to stack several LSTM layer and not just one
        #decoder_cell: stacked LSTM layers with Drop out applied 
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# And final seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    
    '''
        inputs: questions from cornel movie dataset!
        targets: answers from  cornel movie dataset!
        num_words: total number of words in all the ans and q so defined separately.
        encoder_embedding_size: no of the dimension of the embe matrix for decoder.
    '''
    
    
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
 
 