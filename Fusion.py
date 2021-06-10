import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , pipeline


class Fusion:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    def fusion(self,audio,caption,emotion):
        inputtext = ""
        i = 0
        # print(float(caption[3]['start']))
        # print((float(audio[100]['start'])/1000))

        # 1
        for a in audio:
            
            while i < len(caption) and float(caption[i]['start']) <= (float(a['start'])/1000):
                inputtext += caption[i]['sentence'] + '. '
                i+=1
            inputtext += a['text'] + ' '

        # print(inputtext)
        # return 1           
        # # 2
        # for c in caption:
        #     i = 0
        #     inputtext += c['sentence'] + '. '
        #     while i < len(audio) and float(c['end']) >= (float(audio[i]['start'])/1000):
        #         inputtext += audio[i]['text'] + ' '
        #         i+=1
        #     input_ids = self.tokenizer(inputtext, return_tensors="pt").input_ids
        #     output_ids = self.model.generate(input_ids)[0]
        #     print(self.tokenizer.decode(output_ids, skip_special_tokens=True))
            
            
        # 3
        # for a in audio:                    
        #     inputtext += a['text'] + ' '
        # for c in caption:
        #     inputtext += c['sentence'] + '. '
        # print(inputtext)

        # 4    
        # inputtext = audio + ' '
        # for c in caption:
        #     inputtext += c['sentence'] + '. '
        
        input_ids = self.tokenizer(inputtext, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids)[0]
        output_ids = self.model.generate(input_ids, num_beams=4, early_stopping=True)
        return "the dominant emotion in the video is "+ emotion + "\n" +[self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids][0]
        # return "hi"
        # print('The dominant emotion through the video is '+emotion+': \n'+self.tokenizer.decode(output_ids, skip_special_tokens=True))
        # return 'The dominant emotion through the video is '+emotion+': \n'+self.tokenizer.decode(output_ids, skip_special_tokens=True)