from Emotion import Emotion
from Fusion import Fusion
from SpeechToText import SpeechToText
from SplitVideo import SplitVideo
from VideoToText import VideoToText

def main():
    print()
    sv = SplitVideo('tedTalk video') 
    smallVideos, audio = sv.myfunc()
    
    print('_______________________________')
    print()

    st = SpeechToText('tedTalk video')
    audioText =  st.myfunc()

    print(audioText)
    print('_______________________________')
    print()
    
    emotion = Emotion(smallVideos)
    arrayEmotions =  emotion.extractEmotions()
    
    print('_______________________________')
    print()
    
    videoToText=VideoToText(smallVideos)
    arrayFrameText= videoToText.extractText()
    
    print('_______________________________')
    print()
    
    fusion = Fusion( audioText,  videoToText, arrayEmotions)

    summary = fusion.myfunc()
 
    print(summary)

if __name__ == "__main__":
       main() 