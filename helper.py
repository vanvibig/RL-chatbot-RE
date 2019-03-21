
import tensorflow as tf
import numpy as np
import re


def model_inputs(embed_dim, reinforcement=False):
    word_vectors = tf.placeholder(
        tf.float32, [None, None, embed_dim], name="word_vectors")
    reward = tf.placeholder(tf.float32, shape=(), name="rewards")
    caption = tf.placeholder(tf.int32, [None, None], name="caption")
    caption_mask = tf.placeholder(
        tf.float32, [None, None], name="caption_mask")
    if reinforcement:
        # with reinforcement learning, there is an extra placeholder for reward
        return word_vectors, caption, caption_mask, reward
    else:
        # normal training returns only the word_vectors, captions and caption_masks placeholders
        return word_vectors, caption, caption_mask


def encoding_layer(word_vectors, lstm_size, num_layers, keep_prob, vocab_size):
    cells = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(
            lstm_size), keep_prob) for _ in range(num_layers)
    ])

    outputs, state = tf.nn.dynamic_rnn(cells, word_vectors, dtype=tf.float32)

    return outputs, state


def decode_train(enc_state, dec_cell, dec_input,
                 target_sequence_length, output_sequence_length,
                 outputs_layer, keep_prop):
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        dec_cell, output_keep_prob=keep_prop)  # apply dropout to the LSTM cell
    helper = tf.contrib.seq2seq.TrainingHelper(
        dec_input, target_sequence_length)  # training helper for decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell, helper, enc_state, outputs_layer)
    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True, maximum_iterations=output_sequence_length)

    return outputs


def decode_generate(encoder_state, dec_cell, dec_embeddings,
                    target_sequence_length, output_sequence_length,
                    vocab_size, output_layer, batch_size, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(
        dec_cell, output_keep_prob=keep_prob)

    # decoder helper for inference
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        dec_embeddings, tf.fill([batch_size], 1), 2)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell, helper, encoder_state, output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        impute_finished=True,
        maximum_iterations=output_sequence_length)

    return outputs

def decoding_layer(dec_input, enc_state, 
                   target_sequence_length, output_sequence_length,
                   lstm_size, num_layers, n_words,
                   batch_size, keep_prob, embeding_size, Train = True):
    target_vocab_size = n_words
    with tf.device("/cpu:0"):
        dec_embeddings = tf.Variable(
                tf.random_uniform([target_vocab_size, embeding_size], -0.1, 0.1),
                name = 'Wemb')
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(num_layers)])
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
    if Train:
        with tf.variable_scope("decode"):
            train_output = decode_train(enc_state, cells, dec_embed_input,
                                        target_sequence_length,
                                        output_sequence_length,
                                        output_layer,
                                        keep_prob)
    with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        infer_output = decode_generate(enc_state, cells, dec_embeddings,
                                       target_sequence_length,
                                       output_sequence_length,
                                       target_vocab_size,
                                       output_layer,
                                       batch_size,
                                       keep_prob)
    if Train:
        return train_output, infer_output
    return infer_output

def bos_inclusion(caption,batch_size):
    sliced_target = tf.strided_slice(caption, [0,0], [batch_size, -1], [1,1])
    concat = tf.concat([tf.fill([batch_size, 1],1), sliced_target], 1)
    return concat

def pad_sequences(questions, sequence_length = 22):
    lengths = [len(x) for x in questions]
    num_samples = len(questions)
    x = np.zeros((num_samples, sequence_length)).astype(int)
    for idx, sequence in enumerate(questions):
        if not len(sequence):
            continue
        truncated = sequence[-sequence_length:]
        truncated = np.asarray(truncated, dtype=int)
        x[idx, :len(truncated)] = truncated
    return x

def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    data = ' '.join(words)
    return data

def make_batch_input(batch_input, input_sequence_length, embed_dims, word2vec):
    for i in range(len(batch_input)):
        batch_input[i] = [word2vec[w] if w in word2vec else np.zeros(embed_dims) for w in batch_input[i]]
        if len(batch_input[i]) > input_sequence_length:
            batch_input[i] = batch_input[i][ :input_sequence_length]
        else:
            for _ in range(input_sequence_length - len(batch_input[i])):
                batch_input[i].append(np.zeros(embed_dims))
    return np.array(batch_input)

def replace(target, symbols): # remove special symbols
    for symbol in symbols:
        target =  list(map(lambda x: x.replace(symbol,''), target))
        
    return target

def make_batch_target(batch_target, word_to_index, target_sequence_length):
    target = batch_target
    target = list(map(lambda x: '<bos> ' + x, target))
    symbols = ['.', ',', '"', '\n', '?', '!', '\\', '/']
    target = replace(target,symbols)
    
    for idx, each_cap in enumerate(target):
        word = each_cap.lower().split(' ')
        if len(word) < target_sequence_length:
            target[idx] = target[idx] + ' <eos>' # append the end symbol
        else:
            new_word = ''
            for i in range(target_sequence_length-1):
                new_word = new_word + word[i] + ' '
            target[idx] = new_word + '<eos>'
    target_index = [
            [word_to_index[word] if word in word_to_index else word_to_index['<unk>'] 
            for word in sequence.lower().split(' ')] for sequence in target]
    # print(target_index[0])
    caption_matrix = pad_sequences(target_index, target_sequence_length)
    caption_matrix = np.hstack([caption_matrix, np.zeros([len(caption_matrix), 1])]).astype(int)
    
    caption_masks = np.zeros((caption_matrix.shape[0], caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum(), caption_matrix)))
    # print(nonzeros)
    # print(caption_matrix[1])
    for ind, row in enumerate(caption_masks): # set the masks as an array of ones where actual words exist and zeros otherwise
        row[:nonzeros[ind]] = 1
        # print(row)
        # print(caption_masks[0])
        # print(caption_matrix[0])
        
    return caption_matrix, caption_masks
    
def generic_batch(generic_response,  batch_size, 
                  word_to_index, target_sequence_length):
    size = len(generic_response)
    if size > batch_size:
        generic_response = generic_response[:batch_size]
    else:
        for j in range(batch_size-size):
            # kv note
            generic_response.append('')
    return make_batch_target(generic_response, word_to_index, target_sequence_length)

def index2sentence(generated_word_index, prob_logit, ixtoword): 
    """ if the predicted word is 'unknown,<unk>--index == 3, replace with the second most probable word
    Also if the predicted word is <pad> representing a pad or <bos> replace with the next most probable word
    """
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3 or generated_word_index[i] <= 1:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1

            generated_word_index[i] = curindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])

    # generate sentence
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1    #The sentence ends where the punctuation <eos> is found The rest of the sentence is truncated
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)

    """ Modify the output sentence to take off '<eos>, <bos>, '--' 
    Start every sentence with a capital letter, replace i, i'm i'd with I, I'm. I'd respectively and end with a full stop '.'
    """
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace('<eos>', '')
    generated_sentence = generated_sentence.replace('--', '')
    generated_sentence = generated_sentence.split('  ')
    for i in range(len(generated_sentence)):
        generated_sentence[i] = generated_sentence[i].strip() 
        if len(generated_sentence[i]) > 1:
            generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
        else:
            generated_sentence[i] = generated_sentence[i].upper()
    generated_sentence = ' '.join(generated_sentence)
    generated_sentence = generated_sentence.replace(' i ', ' I ')
    generated_sentence = generated_sentence.replace("i'm", "I'm")
    generated_sentence = generated_sentence.replace("i'd", "I'd")

    return generated_sentence



