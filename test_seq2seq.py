import data_parser
from gensim.models import KeyedVectors
from seq_model import Chatbot
import tensorflow as tf
import numpy as np
import helper as h

reinforcement_model_path = "model/reinforcement/seq2seq"
forward_model_path = "model/forward/seq2seq"
reverse_model_path = "model/reverse/seq2seq"

path_to_questions = 'sample_input.txt'
responses_path = 'sample_output.txt'

word_count_threshold = 1
dim_wordvec = 300
# dim_hidden = 1000
dim_hidden = 512
input_sequence_length = 22
output_sequence_length = 22
learning_rate = 0.0001

batch_size = 2
# batch_size = 1

def test(model_path=forward_model_path):
    testing_data = open(path_to_questions, 'r', encoding='utf-8', errors='ignore')
    word_vectors = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

    _, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    # model = Chatbot(dim_wordvec, len(index_to_word), dim_hidden,
    #                 batch_size,
    #                 input_sequence_length, target_sequence_length,
    #                 Training=False)

    model = Chatbot(dim_wordvec, len(index_to_word), dim_hidden,
                    batch_size,
                    input_sequence_length,
                    output_sequence_length,
                    learning_rate,
                    Training=False)

    optimizer, place_holders, predictions, logits, losses = model.build_model()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    saver.restore(sess, model_path)

    with open(responses_path, 'w', encoding='utf-8', errors='ignore') as out:
        for idx, question in enumerate(testing_data):
            print('question => ', question)
            out.write('question: {}\n'.format(question))

            question = [h.refine(w) for w in question.lower().split()]
            question = [word_vectors[w] if w in word_vectors else np.zeros(dim_wordvec) for w in question]

            question.insert(0, np.random.normal(size=(dim_wordvec,)))

            if len(question) > input_sequence_length:
                question =  question[: input_sequence_length]
            else:
                for _ in range(input_sequence_length - len(question)):
                    question.append(np.zeros(dim_wordvec))
            
            question = np.array([question])

            feed_dict = {
                place_holders['word_vectors']: np.concatenate([question]*2,0)
            }

            word_indices, prob_logit = sess.run(
                [predictions, logits],
                feed_dict=feed_dict
            )

            # print(word_indices[0].shape)
            generated_sentence = h.index2sentence(
                word_indices[0], 
                prob_logit[0],
                index_to_word)

            print('generated_sentence => ', generated_sentence)
            out.write('answer: {}\n\n'.format(generated_sentence))

forward_model_path = "model/forward/seq2seq"
if __name__ == "__main__":
    test("model/forward/seq2seq")