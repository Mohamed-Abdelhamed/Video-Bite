class SplitVideo:
    def __init__(self,video):
        self.video = video

    def myfunc(self):
        print("I returned list of small videos of video ("+self.video+") and video audio...")
        return [1,2,3] , "audio"   