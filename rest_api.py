import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from builder import SummaryBuilder,Director
from Fusion import Fusion
import numpy as np
import threading
import requests as req
#from image_model import ImageModel


app = Flask(__name__)
api = Api(app)

class ProcessVideoController(Resource):

    def get(self,video_name,video_id):
            thread = threading.Thread(target=self.processVideo, kwargs={'video_name': video_name , 'video_id':video_id})
            thread.start()
            return jsonify({
                "status": "200",
                })
    def processVideo(self,video_name,video_id):
            print('insideeeeeeeeeeeeeeeeeeee processs video thread..............................')
            video_name = video_name.replace('-','/')
            print(video_name)
            video = '/home/fadybassel/videobite/public' + video_name
            director= Director(SummaryBuilder(video))
            parts= director.getSummaryParts()
            F = Fusion()
            emotions = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
            predicted_emotion = emotions[np.argmax(parts.getEmotion())]
            fusion = F.fusion(parts.getAudio(),parts.getCaption()[0],predicted_emotion)
            dictToSend = {'video_id':video_id,'summary':fusion}
            res = req.post('http://192.168.0.15:8001/api/VideoSummaryUpdate', json=dictToSend)
            print ('response from server:')
            print(res)
            print("------------------------------------------------------")
            print(fusion)


api.add_resource(ProcessVideoController, '/processvideo/<string:video_name>/<string:video_id>')


if __name__=='__main__':
    app.run(debug=True, host='192.168.0.15', port=8000)