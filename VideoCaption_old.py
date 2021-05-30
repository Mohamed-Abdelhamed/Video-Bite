import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np

data_path = "models/Video-caption/data/"
model_path =  "models/Video-caption/models/model-230"
videos_path =  "models/Video-caption/video_feats/"

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
        ixtoword = pd.Series(np.load(data_path + 'ixtoword_special.npy',allow_pickle=1).tolist())
        print('aaaaaaaaaaaaaaaa')
        print(ixtoword)
        model = Video_Caption_Generator(
                    dim_image=dim_image,
                    n_words=len(ixtoword),
                    dim_hidden=dim_hidden,
                    n_lstm_steps=n_frame_step,
                    n_video_lstm_step=n_video_lstm_step,
                    n_caption_lstm_step=n_caption_lstm_step,
                )



        video_tf, video_mask_tf, caption_tf = model.build_generator()
        config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
        sess = tf.InteractiveSession(config=config)
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










import sys
sys.path.append('/home/mark/caffe/python')
import caffe

import cv2
import os
import skimage

num_frames = 80
vgg_model = 'models/Video-caption/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = 'models/Video-caption/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
mean = 'models/Video-caption/vgg/ilsvrc_2012_mean.npy'

class featureExtraction():
    def __init__(self, video):
        self.video = video

    def extractFeatures(self):
        cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

        video_fullpath = videos_path + self.video
        
        cap  = cv2.VideoCapture( video_fullpath )
        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            frame_list.append(frame)
            frame_count += 1

        frame_list = np.array(frame_list)

        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        cropped_frame_list = []
        for frame in frame_list:
            f = self.preprocess_frame(frame)
            cropped_frame_list.append(f)

        cropped_frame_list = np.array(cropped_frame_list)

        feats = cnn.get_features(cropped_frame_list)
        save_full_path = os.path.join(videos_npy_path, self.video + '.npy')
        np.save(save_full_path, feats)

    def preprocess_frame(self, image, target_height=224, target_width=224):
        if len(image.shape) == 2:
            image = np.tile(image[:,:,None], 3)
        elif len(image.shape) == 4:
            image = image[:,:,:,0]

        image = skimage.img_as_float(image).astype(np.float32)
        height, width, _ = image.shape
        if width == height:
            resized_image = cv2.resize(image, (target_height,target_width))

        elif height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

        return cv2.resize(resized_image, (target_height, target_width))




class CNN(object):
    def __init__(self, deploy=vgg_deploy, model=vgg_model, mean=mean, batch_size=10, width=227, height=227):
        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        return net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes)

        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):
            image_batch = image_list[start:end]
            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)
            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)
            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]
            all_feats[start:end] = feats

        return all_feats
