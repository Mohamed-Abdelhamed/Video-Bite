from Emotion import Emotion
from Fusion import Fusion
from SpeechToText import SpeechToText
from SplitVideo import SplitVideo
from VideoToText import VideoToText
import asyncio
async def main():
    loop = asyncio.get_event_loop()
    print()
    sv = SplitVideo('tedTalk video')
    smallVideos, audio = sv.myfunc()
    
    print('_______________________________')
    print()

    st = SpeechToText('tedTalk video') 
    emotion = Emotion(smallVideos) 
    videoToText=VideoToText(smallVideos)

    
    
    result1 = asyncio.ensure_future(emotion.extractEmotions())
    result2 = asyncio.ensure_future(st.myfunc())
    result3 = asyncio.ensure_future(videoToText.extractText())

    one , two , three = await asyncio.gather(result1, result2,result3)

    print()
    print("__________________________________")
    print()
    print(one)
    print(two)
    print(three)
    print()
    print("__________________________________")
    print()


    fusion = Fusion( one,  two, three)

    summary = fusion.myfunc()
 
    print(summary)

if __name__ == "__main__":
       asyncio.run(main()) 