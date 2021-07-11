from Emotion import DetectEmotionController
from Audio import SpeechToText
from video.VideoCaption import VideoToText

class summaryParts:
   def __init__(self):
      self.__audio = []
      self.__caption = []
      self.__emotion = []

   def setAudio(self, audio):
      self.__audio = audio

   def setCaption(self, caption):
      self.__caption=caption

   def setEmotion(self, emotion):
      self.__emotion = emotion

   def getAudio(self):
      return self.__audio

   def getCaption(self):
      return self.__caption

   def getEmotion(self):
      return self.__emotion



class Builder:
      def buildAudio(self): pass
      def buildCaption(self): pass
      def buildEmotion(self): pass
      def getSummaryParts(self):pass     


class SummaryBuilder(Builder):
   
   def __init__(self,video_link,video_name):
      self.parts = summaryParts()
      self.video = video_link
      self.video_name = video_name
   def buildAudio(self):
      audio = SpeechToText(self.video) #audio extraction class
      self.parts.setAudio(audio.getaudio()) 
      
   
   def buildCaption(self):
    
      videoText = VideoToText(self.video)
      self.parts.setCaption(videoText.extractText(self.video_name))
   
   
   def buildEmotion(self):
      print('videoooooooooooooooooooooooooooooooo')
      print(self.video)
      emotion = DetectEmotionController([self.video]) #emotion extraction class
      self.parts.setEmotion(emotion.extract_emotions(2)[0])
      

   def getSummaryParts(self):
       return self.parts
       



class Director:
   
   def __init__(self,builder):
      self.__builder = builder
   
   def getSummaryParts(self):
        
        
        self.__builder.buildAudio()
        self.__builder.buildCaption()
        self.__builder.buildEmotion()
        
        return self.__builder.getSummaryParts()
