import os.path
import sys
import math
import nltk
import spacy
import en_core_web_sm
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
from operator import itemgetter
from collections import defaultdict

# ----------------------------------------
#  Data input 
# ----------------------------------------

class sentiment:
    def readFileToCorpus(self, f):
        """ Reads in the text file f which contains one sentence per line.
        """
        setSentence = set()
        if os.path.isfile(f):
            file = open(f, "r", encoding='utf-8', errors='ignore')  # open the input file in read-only mode
            corpus = []  # this will become a list of sentences
            print("Reading file", f, "...")
            for line in file:
                # eliminate duplicate and to lower case
                if line not in setSentence:
                    setSentence.add(line)
                    corpus.append(line.lower()) # append this list as an element to the list of sentences
            return corpus
        else:
            print("Error: corpus file", f, "does not exist")
            sys.exit()  # exit the script


    def __init__(self):
        self.st = LancasterStemmer()
        self.nlp = spacy.load("en_core_web_sm")
        self.posWords = set()
        self.negWords = set()
        self.posWordsOriginal = []
        self.negWordsOriginal = []


    # find negation in the sentence based on dependency
    def findDependency(self, sen):
        findNegative = False
        negatedWord = ""
        doc = self.nlp(sen)
        for token in doc:
            #print(token.dep_ + "(" + token.head.text + ", " + token.text + ")")
            if findNegative == True:
                if token.head.text == negatedWord:
                    return token.text
            if token.dep_ == 'neg':
                findNegative = True
                negatedWord = token.head.text
        return ""


    def findPOSWord(self, line, attr, negatedWord):
        text = word_tokenize(line)
        taggedToken = nltk.pos_tag(text)
        if attr == 0:
            for item in taggedToken:
                if item[1][0] == 'V' or item[1][0] == 'J':
                    if negatedWord == "" or item[0] != negatedWord:
                        # add this word to set after stemming             
                        self.negWords.add(self.st.stem(item[0]))
                        self.negWordsOriginal.append(self.st.stem(item[0]))
                        # add the synonyms to set
                        for syn in wordnet.synsets(item[0]): 
                            for l in syn.lemmas():
                                self.negWords.add(l.name())
                    else:
                        self.negWords.add("NOT_" + self.st.stem(item[0]))
                        self.negWordsOriginal.append("NOT_" + self.st.stem(item[0]))
        else:
            for item in taggedToken:
                if item[1][0] == 'V' or item[1][0] == 'J':
                    if negatedWord == "" or item[0] != negatedWord:
                        self.posWords.add(self.st.stem(item[0]))
                        self.posWordsOriginal.append(self.st.stem(item[0]))
                        # add the synonyms to set
                        for syn in wordnet.synsets(item[0]): 
                            for l in syn.lemmas():
                                self.posWords.add(l.name()) 
                    else:
                        self.posWords.add("NOT_" + self.st.stem(item[0]))
                        self.posWordsOriginal.append("NOT_" + self.st.stem(item[0]))


    def train(self, corpus, attr):
        i = 0.0
        for line in corpus:
            i += 1.0
            if i % 500 == 0:
                print("another 500 sentences trained now")

            negatedWord = self.findDependency(line)
            self.findPOSWord(line, attr, negatedWord)

    def pathSimilarity(self, word):
        wordset = wordnet.synsets(str(word))
        if len(wordset) == 0:
            return 0.0, 0.0

        largestSimilarityNeg = 0.0
        for i in range(min(50, len(self.negWordsOriginal))):
            comparedWord = wordnet.synsets(str(self.negWordsOriginal[i]))
            if len(comparedWord) == 0:
                continue
            dist = wordnet.path_similarity(wordset[0], comparedWord[0])
            if dist is not None and dist > largestSimilarityNeg:
                largestSimilarityNeg = dist

        largestSimilarityFromPos = 0.0
        for i in range(min(50, len(self.posWordsOriginal))):
            comparedWord = wordnet.synsets(str(self.posWordsOriginal[i]))
            if len(comparedWord) == 0:
                continue
            dist = wordnet.path_similarity(wordset[0], comparedWord[0])
            if dist is not None and dist > largestSimilarityFromPos:
                largestSimilarityFromPos = dist

        return largestSimilarityNeg, largestSimilarityFromPos


    def test(self, corpus):
        posCount = 0.0
        negCount = 0.0
        numFoundInNeg = 0.0
        numFoundInPos = 0.0

        for line in corpus:
            negatedWord = self.findDependency(line)
            text = word_tokenize(line)
            taggedToken = nltk.pos_tag(text)
            setTest = set()
            for item in taggedToken:
                if item[1][0] == 'V' or item[1][0] == 'J':
                    if negatedWord == item[0]:            
                        setTest.add("NOT_" + item[0])
                    else:
                        setTest.add(item[0])

            # if we can find the words in the existing set
            score = 0.0
            for word in setTest:
                if word in self.negWords:
                    score -= 1.0
                    numFoundInNeg += 1.0
                if word in self.posWords:
                    score += 1.0
                    numFoundInPos += 1.0
                # do distance analysis
                elif word not in self.posWords and word not in self.negWords:
                    negSim, posSim = self.pathSimilarity(word)
                    score -= negSim
                    score += posSim


            if score >= 0:
                posCount += 1.0
            else:
                negCount += 1.0

        return posCount, negCount, numFoundInNeg, numFoundInPos

                    

if __name__ == "__main__":
    sentimentReader = sentiment()
    negLowerCaseCorpus = sentimentReader.readFileToCorpus('rt-polarity.neg')
    posLowerCaseCorpus = sentimentReader.readFileToCorpus('rt-polarity.pos')
    negTrain, negTest = train_test_split(negLowerCaseCorpus, test_size=0.3)
    posTrain, posTest = train_test_split(posLowerCaseCorpus, test_size=0.3)

    # need to eliminate from both pos word set and neg word set
    sentimentReader.train(negTrain, 0)
    posNum, negNum, negFound, posFound = sentimentReader.test(negTest)
    #print(negFound)
    #print(len(sentimentReader.negWords))
    print("accuracy in negCorpus prediction: " + str(negNum / len(negTest)))

    sentimentReader.train(posTrain, 1)
    posNum, negNum, negFound, posFound = sentimentReader.test(posTest)
    #print(negFound)
    #print(len(sentimentReader.posWords))
    print("accuracy in posCorpus prediction: " + str(posNum / len(posTest)))

