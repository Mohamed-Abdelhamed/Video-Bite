from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)
DrivePath = '/content/gdrive/MyDrive/vidcap/'

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from os import listdir
from keras.preprocessing import sequence

data_path = DrivePath + "Dataset/"

class vidcapModel:
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') # (token_unique, 1000)
        
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False) # c_state, m_state are concatenated along the column axis 
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W') # (4096, 1000)
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W') # (1000, n_words)
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_generator(self):
        # batch_size = 1
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image]) # (80, 4096)
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


class vidcapController:
    dim_image = 4096
    dim_hidden = 256

    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80

    n_epochs = 1000
    batch_size = 50
    learning_rate = 0.0001

    ixtoword = pd.Series(np.load(DrivePath + 'ixtoword_special.npy').tolist())

    model = vidcapModel(
                dim_image=dim_image,
                n_words=len(ixtoword),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step)
        
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession()
    #print("start to restore")
    saver = tf.train.Saver()
    saver.restore(sess, DrivePath + "models/model-230")
    #print("restore success")

    test_folder_path = data_path + "testing_data/feat/"
    test_path = listdir(test_folder_path)
    test_features = [ (file[:-4],np.load(data_path + "testing_data/feat/" + file)) for file in test_path]
        
    def GenerateCaption(self):
        for idx, video_feat in self.test_features:
            #print(idx)
            video_feat = video_feat.reshape(1,80,4096)
            #print(video_feat.shape)
            if video_feat.shape[1] == self.n_frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

            generated_word_index = self.sess.run(self.caption_tf, feed_dict={self.video_tf:video_feat, self.video_mask_tf:video_mask})
            #print(generated_word_index)
            generated_words = self.ixtoword[generated_word_index]
            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            generated_sentence = generated_sentence.replace('<pad> ', '')
            generated_sentence = generated_sentence.replace(' <pad>', '')
            #print(generated_sentence,'\n')
        return(generated_sentence)

#vidcontroller = vidcapController()
#vidcontroller.GenerateCaption()