from monkeylearn import MonkeyLearn
from pprint import pprint

def extractKeywords(text="gwepog egmwepo gweomgwepog gpowemfwep megowemgw vew pmgew vwepogmew"): 
   #API key goes here  
    ml = MonkeyLearn('68844fb2f7507de0b9ee4f52db2b8ac8335908b0')
    model_id = 'ex_YCya9nrn'
    result = ml.extractors.extract(model_id, [text])
    return result.body
