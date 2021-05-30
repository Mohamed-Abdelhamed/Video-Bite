import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Fusion:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        
    def fusion(self,audio,caption,emotion):
        inputtext = ''
        i = 0
        # print(float(caption[3]['start']))
        # print((float(audio[100]['start'])/1000))

        # for a in audio:
            
        #     while i < len(caption) and float(caption[i]['start']) <= (float(a['start'])/1000):
        #         inputtext += caption[i]['sentence'] + '. '
        #         i+=1
        #     inputtext += a['text'] + ' '

        # for c in caption:
        #     i = 0
        #     inputtext += c['sentence'] + '. '
        #     while i < len(audio) and float(c['end']) >= (float(audio[i]['start'])/1000):
        #         inputtext += audio[i]['text'] + ' '
        #         i+=1
        #     input_ids = self.tokenizer(inputtext, return_tensors="pt").input_ids
        #     output_ids = self.model.generate(input_ids)[0]
        #     print(self.tokenizer.decode(output_ids, skip_special_tokens=True))
            
            

        # for a in audio:                    
        #     inputtext += a['text'] + ' '
        # for c in caption:
        #     inputtext += c['sentence'] + '. '
        # print(inputtext)

        # inputtext = audio + ' '
        # for c in caption:
        #     inputtext += c['sentence'] + '. '

        inputtext = "A boy is skateboarding on a ramp. A person is seen riding a skateboard down a hill and performing tricks on a ramp. A man is seen sitting on a beach with a camera and leads into a man riding on a surfboard on a beach. A man is seen walking down a street with a camera. A man is seen walking down a ramp. A man is seen walking down a ramp while a man is shown on a skateboard. A man is seen walking down a street. The video ends with the closing credits. A man is skateboarding on a ramp. You think it's gonna be easy if you think you're just gonna get that business started without me? Trials a trim. Forget about it. Don't even try to be successful. It's a wrap. It's not going to be easy. But I want you to feel that pain going through your body. And as pain lead your body just what's gonna take in place since then. And so you got to change your mindset. All right? We got to stop looking at pain as if it's something negative. All right? It to me. If it was easy, everybody was doing no pain, no gain. But I guarantee you, if you can outlast pay. If you can get through that pay. If you can get through that discomfort. All right? If you can outlast that discomfort, I guarantee you, baby, on the other side of it is successful. All rolls to success. You got to go through pain. They are success. When you travel down success, you got to go through the road of pain. Baby, I told you before, if it was easy, everybody would do it. And you keep talking about the mistakes. You keep talking about the past. You keep talking about your trials. You keep talking about your situation. And I want you to know that everybody has ever been great. Everybody has had an optimal overcome. They've had a barrier that you had declined. There is no individual who's every success. And he didn't have to go to an obstacle or a barrier to get there. Listen to me very close. Sometimes it's gonna be hard. Sometimes you're gonna live all the value and nowhere you see success. Nowhere that you see anything that removing looks like success. But you got to embrace the face. You got to believe that all this is not happening right now. If you keep pressing, if you keep pressing just what one day is going to be your day? It only takes one extra push up. It only takes one extra mile. It only takes one extra a grain. It only takes one extra. It only takes one extra stuff there to get you the way you're trying to get you. And the goal is you got to go a little further than the man who's trying to get what you're trying to get. You can't go around it. You can't go under it. You can't go over it. The only way you can get to it is through it. You got to be able to see it there believe that he ever just around when there's never going around you. When you got pain in your life, when you're tired, you feel like give it up. And you feel like when you look around, you don't see anything that looks anything like that. You got a break to face. But one day can't be your if you quit. If you quit, no day will ever be your day. Every single day wake up. You got to give it all you got you got a 106% Don't Fool yourself into seeif you feel making excuse no excuses you cannot take success. You got to work for it You got to breathe it You got to sleep in You got to eat it You got to put for 120%."
        input_ids = self.tokenizer(inputtext, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids)[0]
        output_ids = self.model.generate(input_ids, num_beams=4, early_stopping=True)
        print([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids])

        # print('The dominant emotion through the video is '+emotion+': \n'+self.tokenizer.decode(output_ids, skip_special_tokens=True))
        # return 'The dominant emotion through the video is '+emotion+': \n'+self.tokenizer.decode(output_ids, skip_special_tokens=True)


# import json
# import requests

# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# headers = {"Authorization": "Bearer api_DYNYayQSPptmntCSFkzUcLPYDntEuSApAu"}

# def query(payload):
# 	data = json.dumps(payload)
# 	response = requests.request("POST", API_URL, headers=headers, data=data)
# 	return json.loads(response.content.decode("utf-8"))
# data = query(
#     {
#         "inputs": "A boy is skateboarding on a ramp. A person is seen riding a skateboard down a hill and performing tricks on a ramp. A man is seen sitting on a beach with a camera and leads into a man riding on a surfboard on a beach. A man is seen walking down a street with a camera. A man is seen walking down a ramp. A man is seen walking down a ramp while a man is shown on a skateboard. A man is seen walking down a street. The video ends with the closing credits. A man is skateboarding on a ramp. You think it's gonna be easy if you think you're just gonna get that business started without me? Trials a trim. Forget about it. Don't even try to be successful. It's a wrap. It's not going to be easy. But I want you to feel that pain going through your body. And as pain lead your body just what's gonna take in place since then. And so you got to change your mindset. All right? We got to stop looking at pain as if it's something negative. All right? It to me. If it was easy, everybody was doing no pain, no gain. But I guarantee you, if you can outlast pay. If you can get through that pay. If you can get through that discomfort. All right? If you can outlast that discomfort, I guarantee you, baby, on the other side of it is successful. All rolls to success. You got to go through pain. They are success. When you travel down success, you got to go through the road of pain. Baby, I told you before, if it was easy, everybody would do it. And you keep talking about the mistakes. You keep talking about the past. You keep talking about your trials. You keep talking about your situation. And I want you to know that everybody has ever been great. Everybody has had an optimal overcome. They've had a barrier that you had declined. There is no individual who's every success. And he didn't have to go to an obstacle or a barrier to get there. Listen to me very close. Sometimes it's gonna be hard. Sometimes you're gonna live all the value and nowhere you see success. Nowhere that you see anything that removing looks like success. But you got to embrace the face. You got to believe that all this is not happening right now. If you keep pressing, if you keep pressing just what one day is going to be your day? It only takes one extra push up. It only takes one extra mile. It only takes one extra a grain. It only takes one extra. It only takes one extra stuff there to get you the way you're trying to get you. And the goal is you got to go a little further than the man who's trying to get what you're trying to get. You can't go around it. You can't go under it. You can't go over it. The only way you can get to it is through it. You got to be able to see it there believe that he ever just around when there's never going around you. When you got pain in your life, when you're tired, you feel like give it up. And you feel like when you look around, you don't see anything that looks anything like that. You got a break to face. But one day can't be your if you quit. If you quit, no day will ever be your day. Every single day wake up. You got to give it all you got you got a 106% Don't Fool yourself into seeif you feel making excuse no excuses you cannot take success. You got to work for it You got to breathe it You got to sleep in You got to eat it You got to put for 120%.",
#         'parameters': {
#             'min_length':56,
#             'max_length':142,
#         },
#     }
# )

# print(data)




t = Fusion()
t.fusion('','','')