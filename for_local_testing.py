from constants import DEFAULT_PATH
from builder import SummaryBuilder,Director
from Fusion import Fusion
from keywords import extractKeywords
from timestamps import generateTimestamps
import numpy as np

def processViideo(video_name,video_id):
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
        # timestamps = generateTimestamps(parts.getCaption())
        # keyowrds = extractKeywords(parts.fusion)
        # dictToSend = {'video_id':video_id,'summary':fusion,'timestamps':timestamps,'keywords':keyowrds}
        # print(dictToSend)
        print("------------------------------------------------------")
        print(fusion)



processViideo("/o2EUvWyuDAU",13)