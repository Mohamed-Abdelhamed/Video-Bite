import tensorflow as tf
import pandas as pd
import numpy as np

general_path = "/media/mark/G/Downloads/Graduation project/FRAMES CODES/S2VT/"
data_path = general_path + "data/"
model_path = "/media/mark/G/Video-Bite/models/videoToText/model-230"
videos_path = data_path + "testing_data/feat/"

class VideoToText:
    def __init__(self,videos):
        self.videos = videos
    
    def extractText(self):
        videosTexts = []
        dim_image = 4096
        dim_hidden= 256

        n_video_lstm_step = 80
        n_caption_lstm_step = 20
        n_frame_step = 80

        ixtoword = pd.Series(np.load(data_path + 'ixtoword_special.npy').tolist())

        model = Video_Caption_Generator(
                    dim_image=dim_image,
                    n_words=len(ixtoword),
                    dim_hidden=dim_hidden,
                    n_lstm_steps=n_frame_step,
                    n_video_lstm_step=n_video_lstm_step,
                    n_caption_lstm_step=n_caption_lstm_step,
                )



        video_tf, video_mask_tf, caption_tf = model.build_generator()
        sess = tf.InteractiveSession()
        print("start to restore")
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("restore success")

        for video in self.videos:
            video_feat = np.load(videos_path + video + ".npy")
            video_feat = video_feat.reshape(1,80,4096)
            if video_feat.shape[1] == n_frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            generated_words = ixtoword[generated_word_index]
            generated_sentence =  ' '.join(generated_words).replace('<bos> ', '').replace(' <eos>', '').replace('<pad> ', '').replace(' <pad>', '')
            videosTexts.append(generated_sentence)
            
        return videosTexts




class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])
        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

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
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)


        return video, video_mask, generated_words