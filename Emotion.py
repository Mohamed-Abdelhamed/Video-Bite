import asyncio
class Emotion:
    def __init__(self,smallVideos):
        self.smallVideos = smallVideos

    def extractEmotions(self):
        #asyncio.sleep(5)
        print("inside extract emotions i return list containing the dominant emotion in each small video")
        return ['sad','happy','neutral']

