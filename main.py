import os
import random
from pathlib import Path
import pandas as pd
import nltk
import numpy as np
import gensim
from collections import Counter
import re as regex
from xgboost import XGBClassifier as XGBoostClassifier

# Citation : http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost

seed = 666
random.seed(seed)
data_dir = '/home/shivang/Desktop/Pet_Projects/'

# Data PreProcessing
class dataCleanup():
    data = []
    processedData = []
    wordList = []

    dataModel = None
    dataLabels = None
    isTesting = None

    categories = ['positive', 'negative', 'neutral']

    def init(self, csv_file, isTestingSet=False):
        self.isTesting = isTestingSet
        if not isTestingSet:
            self.data = pd.read_csv(csv_file, header=0, names=['id', 'category', 'tweet'])
            self.data = self.data[self.data['category'].isin(self.categories)]
        else:
            self.data = pd.read_csv(csv_file, header=0, names=['id', 'tweet'], dtype={"id":"int64","tweet":"str"},nrows=4000)
            self.data = self.data.loc[(1 ^ pd.isnull(self.data["tweet"]))&(1 ^ pd.isnull(self.data["id"])), :]

        self.processedData = self.data
        self.wordList = []

    def addColumn(self, columnName, columnContent):
        self.processedData.loc[:, columnName] = pd.Series(columnContent, index=self.processedData.index)

    def buildFeatures(self):
        def countByExp(exp, wordArr):
            return len(list(filter(exp, wordArr)))

        def countOccurences(character, wordArr):
            counter = 0
            for j, word in enumerate(wordArr):
                for char in word:
                    if char == character:
                        counter = counter + 1
            return counter

        def countByRegex(regex, plainText):
            return len(regex.findall(plainText))

        self.addColumn("splitted_text", map(lambda txt: txt.split(" "), self.processedData["tweet"]))

        #find the number of upper case Words
        uppercase = list(map(lambda txt: countByExp(lambda word: word == word.upper(), txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_uppercase", uppercase)

        #find number of !
        exclamations = list(map(lambda txt: countOccurences("!", txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_exclamation", exclamations)

        #find number of ?
        questions = list(map(lambda txt: countOccurences("?", txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_question", questions)

        #find number of ...
        ellipsis = list(map(lambda txt: countByRegex(regex.compile(r"\.\s?\.\s?\."), txt), self.processedData["tweet"]))
        self.addColumn("number_of_ellipsis", ellipsis)

        #find number of #
        hashtags = list(map(lambda txt: countOccurences("#", txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_hashtags", hashtags)

        #find number of @
        mentions = list(map(lambda txt: countOccurences("@", txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_mentions", mentions)

        #find number of quotes ""
        quotes = list(map(lambda plain_text: int(countOccurences("'", [plain_text.strip("'").strip('"')]) / 2 + countOccurences('"', [plain_text.strip("'").strip('"')]) / 2), self.processedData["tweet"]))
        # quotes = list(map(lambda plain_text: int(countOccurences("'", [plain_text.strip("'").strip('"')]) / 2 + countOccurences('"', [plain_text.strip("'").strip('"')]) / 2), self.processedData["tweet"]))
        self.addColumn("number_of_quotes", quotes)

        #find number of Urls
        urls = list(map(lambda txt: countByRegex(regex.compile(r"http.?://[^\s]+[\s]?"), txt), self.processedData["tweet"]))
        self.addColumn("number_of_urls", urls)

        emoticons = {}
        content = Path(data_dir + 'emoticons.txt').read_text()
        sentiment = True
        for l in content.split("\n"):
            if "positive" in l.lower():
                sentiment = True
                continue
            elif "negative" in l.lower():
                sentiment = False
                continue
            emoticons[l] = sentiment

        def isPositive(emoticon):
            if emoticon in emoticons:
                return emoticons[emoticon]
            return False

        def isEmoticon(char):
            return char in emoticons

        #find number of positive emoticons
        positiveEmoticons = list(map(lambda txt: countByExp(lambda word: isEmoticon(word) and isPositive(word), txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_positiveEmoticons", positiveEmoticons)

        negativeEmoticons = list(map(lambda txt: countByExp(lambda word: isEmoticon(word) and not isPositive(word), txt), self.processedData["splitted_text"]))
        self.addColumn("number_of_negativeEmoticons", negativeEmoticons)



    @staticmethod
    def removeByRegex(t, regExp):
        t.loc[:, "tweet"].replace(regExp, "", inplace=True)
        return t

    def removeUrls(self, t):
        return dataCleanup.removeByRegex(t, regex.compile(r"http.?://[^\s]+[\s]?"))

    def removeNotAvailable(self, t):
        return t[t['tweet']!='Not Available']

    def removeUserNames(self, t):
        return dataCleanup.removeByRegex(t, regex.compile(r"@[^\s]+[\s]?"))

    def removeSpecialCharacters(self, t):
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$", "@", "%", "^", "*", "(", ")", "{", "}", "[", "]", "|", "/", "\\", ">", "<", "-", "!", "?", ".", "'", "--", "---", "#"]):
            t.loc[:, "tweet"].replace(remove, "", inplace=True)
        return t

    def removeNumbers(self, t):
        return dataCleanup.removeByRegex(t, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

    def cleanData(self):
        t = self.processedData
        t = self.removeUrls(t)
        t = self.removeUserNames(t)
        t = self.removeNotAvailable(t)
        t = self.removeSpecialCharacters(t)
        t = self.removeNumbers(t)
        self.processedData = t

# Tokenize and Stemming of tweets
class stemAndTokenizeData():
    stemmer = None
    tokenizer = None

    def stem(self, t,  stemmer=nltk.PorterStemmer()):
        self.stemmer = stemmer
        def stemAndJoin(row):
            row["tweet"] = list(map(lambda str: stemmer.stem(str.lower()), row["tweet"]))
            return row
        t = t.apply(stemAndJoin, axis=1)
        return t

    def tokenize(self, t, tokenizer=nltk.word_tokenize):
        self.tokenizer = tokenizer
        def tokenizeRow(row):
            row["tweet"] = tokenizer(row["tweet"])
            row["tokenized_tweet"] = [] + row["tweet"]
            return row
        t = t.apply(tokenizeRow, axis=1)
        return t

class buildWordList():
    requiredList = ["not", "n't", "isn't", "wasn't", "weren't", "wouldn't", "won't"]
    wordList = []

    def buildWordsDict(self, t):
        wordDict = Counter()
        for i in t.index:
            wordDict.update(t.loc[i, "tweet"])
        return wordDict

    def filterStopWords(self, t, wordDict):
        stopwords = nltk.corpus.stopwords.words("english")
        requiredList = ["not", "n't", "isn't", "wasn't", "weren't", "wouldn't", "won't"]
        for i, stopword in enumerate(stopwords):
            if stopword not in requiredList:
                del wordDict[stopword]

    def buildWordListFunction(self, t, minOccurance=3, maxOccurance=500, stopwords=nltk.corpus.stopwords.words("english"), requiredList=None):
        if requiredList is None:
            requiredList = self.requiredList

        if os.path.isfile(data_dir + 'wordlist.csv'):
            wordDf = pd.read_csv(data_dir + 'wordlist.csv')
            wordDf = wordDf[wordDf["occurences"] > minOccurance]
            # print(wordDf.head(5))
            self.wordList = list(wordDf.loc[:, "word"])
            return

        wordDict = self.buildWordsDict(t)
        self.filterStopWords(t, wordDict)

        #Create a new DataFrame wordDf to store words along with their occurences in a csv file
        wordDf = pd.DataFrame(data={"word":[w for w,o in wordDict.most_common() if minOccurance < o < maxOccurance], "occurences":[o for w,o in wordDict.most_common() if minOccurance < o < maxOccurance]}, columns=["word", "occurences"])
        wordDf.to_csv(data_dir + 'wordlist.csv', index_label='id')
        self.wordList = [w for w,o in wordDict.most_common() if minOccurance < o < maxOccurance]


#Bag of Words Transformation
class bagOfWords():

    dataModel = None
    dataLabels = None

    def buildDataModel(self, t, wordList, w2V, stopwords=nltk.corpus.stopwords.words("english"), isTestingSet=False):
        requiredList = ["not", "n't", "isn't", "wasn't", "weren't", "wouldn't", "won't"]
        stopwords = list(filter(lambda sw: sw not in requiredList, stopwords))
        featureColumns = [column for column in t.columns if column.startswith("number_of")]
        similarityColumns = ["bad_similarity", "good_similarity", "information_similarity"]
        labelColumn = []
        if isTestingSet is False:
            labelColumn = ["label"]

        #create a map of all the words in the wordlist
        bowColumns = labelColumn + ["original_id"] + featureColumns + similarityColumns + list(map(lambda i: "word2vec_{0}".format(i), range(0, w2V.dimensions))) + list(map(lambda w: w + "_bow", wordList))
        labels = []
        rows = []

        for i in t.index:
            currentRow = []
            if isTestingSet is False:
                currentLabel = t.loc[i, "category"]
                labels.append(currentLabel)
                currentRow.append(currentLabel)

            currentRow.append(t.loc[i, "id"])

            for j, c in enumerate(featureColumns):
                currentRow.append(t.loc[i, c])

            tokens = t.loc[i, "tokenized_tweet"]
            for mw in map(lambda w : w.split("_")[0], similarityColumns):
                currentSimilarities = [abs(sim) for sim in map(lambda word : w2V.getSimilarity(mw, word.lower()), tokens) if sim is not None]
                if len(currentSimilarities) <= 1:
                    currentRow.append(0 if len(currentSimilarities)==0 else currentSimilarities[0])
                    continue
                maxSim = max(currentSimilarities)
                minSim = min(currentSimilarities)
                currentSimilarities = [((sim - minSim)/(maxSim - minSim)) for  sim in currentSimilarities]
                currentRow.append(np.array(currentSimilarities).mean())

            tokens = t.loc[i, "tokenized_tweet"]
            currentW2V = []
            for j,word in enumerate(tokens):
                vector = w2V.getVector(word.lower())
                if vector is not None:
                    currentW2V.append(vector)

            print(len(currentW2V))
            if len(currentW2V) == 0:
                averageW2V = []
            else:
                averageW2V = list(np.array(currentW2V).mean(axis=0))
            currentRow = currentRow + averageW2V


            tokens = set(t.loc[i, "tweet"])
            for j, word in enumerate(wordList):
                currentRow.append(1 if word in tokens else 0)

            rows.append(currentRow)

        self.dataModel = pd.DataFrame(rows, columns=bowColumns)
        self.dataLabels = pd.Series(labels)


class useWord2Vec():
    word2Vec = None
    dimensions = 0
    def load(self, pathToWord2Vec):
        self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format(pathToWord2Vec, binary=False)
        self.word2Vec.init_sims(replace=True)
        self.dimensions = self.word2Vec.vector_size

    def getVector(self, word):
        if word not in self.word2Vec.vocab:
            return None
        return self.word2Vec.syn0norm[self.word2Vec.vocab[word].index]

    def getSimilarity(self, word1, word2):
        if word1 not in self.word2Vec.vocab or word2 not in self.word2Vec.vocab:
            return None
        return self.word2Vec.similarity(word1, word2)


if __name__ == "__main__":
    cd = dataCleanup()
    cd.init(data_dir + 'trainData.csv')
    cd.buildFeatures()
    cd.cleanData()
    t = cd.processedData
    st = stemAndTokenizeData()
    t = st.tokenize(t)
    t = st.stem(t)

    uW2V = useWord2Vec()
    uW2V.load(data_dir + 'glove.twitter.27B.200d.txt')

    bw = buildWordList()
    bw.buildWordListFunction(t)
    bow = bagOfWords()
    bow.buildDataModel(t, bw.wordList, uW2V)
    dataModel = bow.dataModel
    dataModel.drop("original_id", axis=1, inplace=True)

    print("Training Model built!");

    cdTest = dataCleanup()
    cdTest.init(data_dir + 'tweets.csv', isTestingSet=True)
    cdTest.buildFeatures()
    cdTest.cleanData()
    tTest = cdTest.processedData
    stTest = stemAndTokenizeData()
    tTest = stTest.tokenize(tTest)
    tTest = stTest.stem(tTest)
    bwTest = buildWordList()
    bwTest.buildWordListFunction(tTest)
    bowTest = bagOfWords()
    bowTest.buildDataModel(tTest, bwTest.wordList, uW2V, isTestingSet=True)
    dataModelTest = bowTest.dataModel

    print("Testing Model built!");
    xgboost = XGBoostClassifier(seed=seed,n_estimators=403,max_depth=10,objective="binary:logistic",learning_rate=0.15)
    xgboost.fit(dataModel.iloc[:,1:],dataModel.iloc[:,0])

    print("Training Finished!");

    predictions = xgboost.predict(dataModelTest.iloc[:,1:])
    results = pd.DataFrame([],columns=["Id","Category"])
    results["Id"] = dataModelTest["original_id"].astype("int64")
    results["Category"] = predictions
    results.to_csv("results.csv",index=False)
    print("Results have been saved to file!!");
