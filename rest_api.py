from constants import DEFAULT_PATH
import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from builder import SummaryBuilder,Director
from Fusion import Fusion
import numpy as np
import threading
import requests as req
from keywords import extractKeywords
from timestamps import generateTimestamps
#from image_model import ImageModel


app = Flask(__name__)
api = Api(app)

class ProcessVideoController(Resource):

    def get(self,video_path,video_id,file_name):
            thread = threading.Thread(target=self.processVideo, kwargs={'video_path': video_path , 'video_id':video_id , 'file_name':file_name})
            thread.start()
            return jsonify({
                "status": "200",
                })
    def processVideo(self,video_path,video_id,file_name):
            print('inside processs video thread..............................')
            print(video_id)
            video = DEFAULT_PATH+video_path.replace('-','/')
            print(video)
            director= Director(SummaryBuilder(video,file_name))
            parts= director.getSummaryParts()
            F = Fusion()
            emotions = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
            predicted_emotion = emotions[np.argmax(parts.getEmotion())]
            # predicted_emotion= ''
            fusion = F.fusion(parts.getAudio(),parts.getCaption(),predicted_emotion)
            timestamps = generateTimestamps(parts.getCaption())
            keyowrds = extractKeywords(fusion)
            dictToSend = {'video_id':video_id,'summary':fusion,'timestamps':timestamps,'keywords':keyowrds}
            print(dictToSend)
            res = req.post('http://192.168.0.16:8001/api/VideoSummaryUpdate', json=dictToSend)
            print ('response from server:')
            print(res)
            print("------------------------------------------------------")
            print(fusion)

# ProcessVideoController().processVideo('/skating.mp4', 100)
api.add_resource(ProcessVideoController, '/processvideo/<string:video_path>/<string:video_id>/<string:file_name>')


if __name__=='__main__':
    app.run(debug=True, host='192.168.0.16', port=8000)
    # app.run(debug=True, host='192.168.1.9', port=8000)
