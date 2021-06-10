# UPLOAD VIDEO
import requests
from statistics import mode

class SpeechToText:
  def __init__(self,audio):
    # self.audio = '/home/markrefaat/videobite/public/skating.mp4'
    self.audio = audio
    self.videoId= ""
    # self.authKey = "773123bf14534e1c823fa8a63eaa5c74"
    self.authKey = "c9bb18dea137460ba2f72d40b814bded"

  def read_file(self,filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

  def send_to_api(self):
    filename = self.audio
    headers = {'authorization': self.authKey}
    response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=self.read_file(filename))

    upload_url = response.json()['upload_url']
    print(upload_url)


    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
      "audio_url": upload_url
    }

    headers = {
        "authorization": self.authKey,
        "content-type": "application/json"
    }

    response = requests.post(endpoint, json=json, headers=headers)
    videoId = response.json()['id']
    print(response.json())
    print(videoId)
    self.videoId=videoId
  
  def getaudio(self,videoId="jhw1rxbr2-f28f-4bd5-a7f8-cd2327f4ac0f"): #q2x7mahw8-a256-4b6d-a9b9-166c828a0253
    if videoId == "":
      videoId=self.videoId
      
    print("heelooo")
    endpoint = "https://api.assemblyai.com/v2/transcript/" + videoId

    headers = {"authorization": self.authKey,}

    response = requests.get(endpoint, headers=headers)
    print(response.json()['words'])
    return response.json()['words']
   


