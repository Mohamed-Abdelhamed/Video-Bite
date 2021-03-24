# UPLOAD VIDEO

import sys
import time
import requests
import json
import csv
from statistics import mode
from collections import Counter

class SpeechToText:
  def __init__(self,audio):
    self.audio = audio
    self.videoId=""
    self.authKey = "773123bf14534e1c823fa8a63eaa5c74"

  def read_file(self,filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

  def send_to_api(self):
    filename ="/content/gdrive/MyDrive/Video Bite - Grad/Projects/Audio to Text/example_videos/movie5.mp4"
    headers = {'authorization': "773123bf14534e1c823fa8a63eaa5c74"}
    response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=self.read_file(filename))

    upload_url = response.json()['upload_url']
    print(upload_url)


    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
      "audio_url": upload_url
    }

    headers = {
        "authorization": "773123bf14534e1c823fa8a63eaa5c74",
        "content-type": "application/json"
    }

    response = requests.post(endpoint, json=json, headers=headers)
    videoId = response.json()['id']
    print(response.json())
    print(videoId)
    self.videoId=videoId
  
  def getaudio(self,videoId=""):
    if videoId != "":
      videoId=self.videoId
      
    print("heelooo")
    endpoint = "https://api.assemblyai.com/v2/transcript/" + videoId

    headers = {"authorization": "773123bf14534e1c823fa8a63eaa5c74",}

    response = requests.get(endpoint, headers=headers)

    print(response.json()['text'])
