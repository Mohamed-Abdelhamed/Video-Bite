import json
import sys
import time
import requests
import csv
from statistics import mode
from collections import Counter

# endpoint = "https://api.assemblyai.com/v2/transcript/srhponb3x-dbfb-4fc4-9937-ed3a82edf257"

# headers = {
#     "authorization": "",
# }

# response = requests.get(endpoint, headers=headers)

# # print(response.json())

# time_split = 20
# json_data = response.json()
# all_sentences = []
# sentence = ''
# time = 0
# for i, word in enumerate(json_data['words']):
#     time = time + ((word['end'] - word['start']) / 1000)
#     if time <= time_split:
#         sentence += word['text'] + ' '
#     else:
#         all_sentences.append(sentence)
#         sentence = ''
#         time = 0
#         sentence += word['text'] + ' '
# # print(all_sentences)

time_split = 20
with open('data.json') as json_file:
    json_data = json.load(json_file)
    all_sentences = []
    last_times = []
    sentence = ''
    time = 0
    for i, word in enumerate(json_data['words']):
        time = time + ((word['end'] - word['start']) / 1000)
        if time <= time_split:
            sentence += word['text'] + ' '
        else:
            all_sentences.append(sentence)
            sentence = ''
            time = 0
            sentence += word['text'] + ' '
            last_times.append(word['end'] / 1000)
    all_sentences.append(sentence)

    # print(last_times)
    # print(all_sentences)

with open('result-time4.csv') as emotions:
    csv_reader = csv.reader(emotions, delimiter=',')
    next(csv_reader)
    line_count = 0
    paragraph_no = 0
    list_emotions_of_paragraph = []
    for row in csv_reader:
        if(paragraph_no < len(all_sentences) -1):
            if float(row[11]) <= last_times[paragraph_no]:
                list_emotions_of_paragraph.append(row[12])
            else:
                if list_emotions_of_paragraph:
                    try:
                        dominat_emotion_in_paragraph = mode(list_emotions_of_paragraph)
                        all_sentences[paragraph_no] = "She " + dominat_emotion_in_paragraph + " said " + all_sentences[paragraph_no]
                    except:
                        print('')
                list_emotions_of_paragraph = []
                paragraph_no += 1
    # print(all_sentences)

    paragraph = "".join(all_sentences)
    print("Full paragraph with emotions")
    print(paragraph)
    print()



from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 10


parser = PlaintextParser.from_string(paragraph, Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

print("Summarized Text using textrank")
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
print()




# import re
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words
# from sumy.summarizers.lsa import LsaSummarizer

# def summarize(language="english"):
#     """ Generate segmented summary

#     Args:
#         srt_file(str) : The name of the SRT FILE
#         n_sentences(int): No of sentences
#         language(str) : Language of subtitles (default to English)

#     Returns:
#         list: segment of subtitles

#     """
#     parser = PlaintextParser.from_string(
#         json_data['text'], Tokenizer(language))
#     stemmer = Stemmer(language)
#     summarizer = LsaSummarizer(stemmer)
#     summarizer.stop_words = get_stop_words(language)
#     segment = []
#     for sentence in summarizer(parser.document, 14):
#         index = int(re.findall("\(([0-9]+)\)", str(sentence))[0])
#         item = all_sentences[index]
#         segment.append(srt_segment_to_range(item))
#     print(segment)
#     return segment

# summarize()

