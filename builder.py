from Emotion import DetectEmotionController
from Audio import SpeechToText
from video.VideoCaption import VideoToText
# from VideoCaption import VideoToText , featureExtraction
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
   
   def __init__(self,video):
      self.parts = summaryParts()
      self.video = video
   
   def buildAudio(self):
      # print(self.video)
      # exit()
      audio = SpeechToText(self.video) #audio extraction class
      self.parts.setAudio(audio.getaudio()) 
      #self.parts.setAudio(self.video+" ana msh ananeya ananeya ana 3yzak leya lw7dy") 
   
   def buildCaption(self):
      #videofeat = featureExtraction(self.video)
      #videoText = VideoToText(videofeat.extractFeatures()) #video to text extraction class      
      videoText = VideoToText(self.video.rsplit('.', -1)[0]) # take video name without extenions (ex .mp4)
      # videoText = VideoToText(['04Gt01vatkk_248_265.avi'])
      self.parts.setCaption(videoText.extractText())
      # self.parts.setCaption("This is sentence")
   
   def buildEmotion(self):
      print('videoooooooooooooooooooooooooooooooo')
      print(self.video)
      emotion = DetectEmotionController([self.video]) #emotion extraction class
      self.parts.setEmotion(emotion.extract_emotions(2)[0])
      #self.parts.setEmotion(self.video +" sad")

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
