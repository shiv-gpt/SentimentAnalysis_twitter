import json
import os
import csv
import nltk
# from text_cleaner import remove, keep
# from text_cleaner.processor.common import ASCII
# from text_cleaner.processor.chinese import CHINESE, CHINESE_SYMBOLS_AND_PUNCTUATION
# from text_cleaner.processor.misc import RESTRICT_URL
import regex
import re

data_dir = '/home/shivang/Desktop/Pet_Projects/'

def loadData():
    counter = 0
    tweets = []
    words = set(nltk.corpus.words.words())
    with open(data_dir + 'Output.json', 'r') as f:
        for line in f:
            # line = f.readline() # read only the first tweet/line
            tweet = json.loads(line) # load it as Python dict
            # print(json.dumps(tweet, indent=4)) # pretty-print
            counter = counter + 1
            # text = " ".join(w for w in nltk.wordpunct_tokenize(tweet["text"]) if w.lower() in words or not w.isalpha())
            # text = regex.sub(r'[^\p{Latin}]', '', 'hakuna 123456 &%^$#&#&$#&')
            text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', tweet["text"])
            # pattern = re.compile(r'".*?"', re.DOTALL)
            # pattern.sub(lambda x: x.group().replace('\n', ''), text)
            text = text.strip()
            text = text.replace('\r', ' ').replace('\n', ' ')
            print(text)
            # print("\n")
            # text = " ".join(text.split())
            if text is not "":
                print(str(counter))
                tweets.append((tweet["id"], text))


    with open(data_dir + 'tweets.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        wr.writerow(["id", "tweet"])
        wr.writerows(tweets)


    # with open(data_dir + 'rawTweets.csv', "r") as input, open(data_dir + 'tweets.csv', "w") as output:
    #     w = csv.writer(output)
    #     for record in csv.reader(input):
    #         w.writerow(tuple(s.remove("\n") for s in record))
    # print(str(counter))


if __name__ == "__main__":
    loadData()
