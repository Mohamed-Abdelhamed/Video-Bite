import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Fusion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
        
    def fusion(self,audio,caption,emotion):
        inputtext = caption + '. ' + audio
        input_ids = self.tokenizer(inputtext, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids)[0]
            
        return 'The dominant emotion through the video is '+emotion+': \n'+self.tokenizer.decode(output_ids, skip_special_tokens=True)






