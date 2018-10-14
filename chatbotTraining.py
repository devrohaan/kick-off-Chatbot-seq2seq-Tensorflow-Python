# -*- coding: utf-8 -*-


# Importing training and support libraries.

import numpy as np
import tensorflow as tf
import time
import sys
sys.path.append('./')

# Importing seq2seqModel utilities.

from seq2seqModelUtils import model_inputs, preprocess_targets, encoder_rnn, decode_training_set, decode_test_set, decoder_rnn, seq2seq_model
from hyperParam import epochs, batch_size, rnn_size, num_layers, encoding_embedding_size, decoding_embedding_size, learning_rate, learning_rate_decay, min_learning_rate, keep_probability
from dataClearUtils import clean_text

"""********** DATA EXTRACTION AND PREPROCESSING START   ***************
"""
 
# Importing the training dataset

lines = open('/PATH/trainingdata/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('/PATH/trainingdata/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


'''
********** Pit stop 1 ***************

## To train a chatbot/Conversational Agent always create the dataset in two formats.
## First : Input dataset
## Second  : Output dataset

## ** WHY? **
## The reason behind this model needs to learn that for every input i it must be compared with the output j.
## So to learn the conversation how it is carried out by humans.

## ** HOW? **
## And the best way to do this is to use dictonary.

'''


# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

'''
Sample Dictionary: id2line
    
L1000: Oh, Christ.  Don't tell me you've changed your mind.  I already sent 'em a check.
L10000: Oh... chamber runs.  Uh huh, that's good.  Well, hey... you guys know any songs?

'''


# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

'''
Sample Dictionary: conversations_ids

Index             Conversations

0        ['L194', 'L195', 'L196', 'L197']
1        ['L198', 'L199']
2        ['L200', 'L201', 'L202', 'L203']
'''



'''
********** Pit stop 2 ***************

## Getting the question and answers separetely.
## First list on question 
## Second list of answeres. but both should of the same size
## Question at index i must contain the answer in index i of the answers list
## So in conversation_ids first id is q and then the next id is answer to that question 
# Getting separately the questions and the answers

'''

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

 

'''
********** Pit stop 3 ***************

## Data Preprocessing
## First : Clean the data
## Second  : Filter out the questions and answers that are too short or too long.
## Third: Create bag of words => Assiging unique integer to each words from two dicts.
          Remove non frequent words => Tokenisation and removing the words below threshold.
          **In general remove 5% of least appeared words.
'''



# First: Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
 
# First: Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


# Second: Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1

##Third: Creating bag of words
    
# Counting each word with its total occurence.

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Remove non frequent words.
# create two dictionaries that map the questions words and the answers words to a unique integer
threshold_questions = 15
threshold_answers = 15

questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number ## assign unique integer
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number ## assign unique integer
        word_number += 1





## add unique tokens EOS SOS and special symbols to bag of words question2int and answer2int

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
 
'''
********** Pit stop 4 ***************

## We need to create inverse dic of answerswords2int. 
## we will need the inverse mapping from answerswords2int's integer to answerswords2int's word for seq2seq model.
## how to inverse the dictionary
## w_i corressponds to word integer value in answerswords2int
## w corressponds to word ie key in answerswords2int
'''
 
# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

## for decoding part we need to add EOS that we have created to every answer.
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'


'''
********** Pit stop 5 ***************

## Now we need to transalte all clean_question and clean_answers to integers.
## Means we are converting the words of each question from clean_questions into a collection of unique integer as per the bag of words, we have created ie questionword2int.

## We are doing this so that we shall able to sort all the qestion and answers as per their length (ie total number of words in a question or answer).
## ** WHY? **
## to optimize the training performace !!!

## Now what has happend we have filtered out some of the words based on some threshold.
## So, in our clean_q list while converting the list to integer we must substitite those words as ...
## OUT to indicate that word was filtered out so we will use the unique integer assigned to OUT in qword2int dictionary.

'''


questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

'''
********** Pit stop 6 ***************

## Let;s sort the question and answer as per their lengths.
## We dont want to include conversations that are very long for Ex. very big texts
## We focus of questions and answers whiich are of one length 1, 2 ,3 , ... So, We are choosing stopping_length as 25.
## We are teaching our bot short sentences.
## This will speed up the training and it will reduce the amount of padding during the training.
'''


sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])

 


"""    ********** TRAINING THE SEQ2SEQ MODEL ***************
"""

# defining tensor flow session on which tensorflow training will work
# to do so always reset the graph first
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# load the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# set the sequence length : we will be using the questions and answers for training that are having only 25 words in it.
# REF: code just above starting the building model we have chosen the length to be 25

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# set the input tensors shape
input_shape = tf.shape(inputs)
 
# getting the training and test predictions
# inputs are not in shape so we can use reshape method from numpy or this
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)


'''
********** Pit stop 7 ***************

## Setting up the loss Error, the optimizer , gradient clipping!!
## GC : to avoid vanishing grading issue we use gradient clipping.
## loss error : weighted cross entropy loss error when dealing with sequences.
## optimizer:  Adam optimizer.
## We will apply gradient clipping to avoid vanishing or exploding gradient issues.
'''

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)



'''
********** Pit stop 8 ***************

## Padding the sequences with <PAD> token :Eeach sentence of the batch must have same length!!

## ** Example **

## quetion: [ 'who', 'are', 'you' ]
## answer: [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
	
## After padding:

## quetion: [ 'who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>, <PAD>]
## answer: [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD>,]
	
## lenghts of question and answers sequence must be same!	
'''


def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Split the data into batches of questions and answers 
# total no of questions / batch_size = number of batches
# Splitting the data into batches of questions and answers.
    
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        

# Keeping 10 to 20 % of the training data to validate.
# splitting training and validation/test sets.

training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Actual Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
#for each epoch we need to iterate over all the batches
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Training Done!!!!")
 
