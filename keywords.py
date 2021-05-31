from monkeylearn import MonkeyLearn
from pprint import pprint

def extractKeywords(text): 
   #API key goes here  
    ml = MonkeyLearn('')
    model_id = 'ex_YCya9nrn'
    result = ml.extractors.extract(model_id, [text])
    return result.body