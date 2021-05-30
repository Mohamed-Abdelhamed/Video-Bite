from constants import DEFAULT_PATH
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
            print('inside processs video thread..............................')
            print(video_name)
            print(video_id)
            video_name = video_name.replace('-','/')
            video = DEFAULT_PATH + video_name
            director= Director(SummaryBuilder(video))
            parts= director.getSummaryParts()
            F = Fusion()
            emotions = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
            predicted_emotion = emotions[np.argmax(parts.getEmotion())]
            # predicted_emotion= ''
            fusion = F.fusion(parts.getAudio(),parts.getCaption(),predicted_emotion)
            dictToSend = {'video_id':video_id,'summary':fusion}
            print(dictToSend)
            #res = req.post('http://192.168.1.9:8001/api/VideoSummaryUpdate', json=dictToSend)
            #print ('response from server:')
            #print(res)
            #print("------------------------------------------------------")
            #print(fusion)

# ProcessVideoController().processVideo('/skating.mp4', 100)
api.add_resource(ProcessVideoController, '/processvideo/<string:video_name>/<string:video_id>')


if __name__=='__main__':
    app.run(debug=True, host='192.168.50.234', port=8000)
    # app.run(debug=True, host='192.168.1.9', port=8000)
