########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict

# Unknown word token
UNK = 'UNK'

class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]
# A class for evaluating POS-tagged data

class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
        self.TP = defaultdict(float)
        self.FP = defaultdict(float)
        self.FN = defaultdict(float)

        self.totalCorrectWord = 0.0
        self.totalWord = 0.0
        self.totalSentence = 0.0
        self.correctSentence = 0.0
        self.tag_counts = defaultdict(float)

        if os.path.isfile(goldFile):
            file1 = open(goldFile, "r") # open the input file in read-only mode
            for line in file1:
                raw = line.split()
                sentence = []
                for token in raw:
                    self.tag_counts[TaggedWord(token).tag] += 1.0
        self.tagList = list(self.tag_counts.keys())

        self.matrix = [[0 for x in range(len(self.tag_counts))] \
            for y in range(len(self.tag_counts))]
        self.train(goldFile, testFile)
        
        #print(len(self.tag_counts))


    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    def train(self, goldFile, testFile):
        goldsens = self.readLabeledData(goldFile)
        testsens = self.readLabeledData(testFile)
        
        for i in range(len(goldsens)):
            self.totalSentence += 1.0
            flag = True
            for j in range(len(goldsens[i])):
                goldWord = goldsens[i][j]
                testWord = testsens[i][j]
                self.totalWord += 1.0
                self.matrix[self.tagList.index(goldWord.tag)][self.tagList.index(testWord.tag)] += 1.0
                if(goldWord.tag == testWord.tag):         
                    self.totalCorrectWord += 1.0
                    self.TP[goldWord.tag] += 1.0
                else:
                    flag = False
                    self.FP[testWord.tag] += 1.0
                    self.FN[goldWord.tag] += 1.0
            if flag:
                self.correctSentence += 1.0

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        print("Return the percentage of correctly-labeled tokens")

        return self.totalCorrectWord / self.totalWord

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        print("Return the percentage of sentences where every word is correctly labeled")
        return self.correctSentence / self.totalSentence

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")
        f=open(outFile, 'w+')
        #self.viterbi(data[0])
        header = "\t"
        for i in range(len(self.tagList)):
            header += self.tagList[i] + '\t'
        print(header.rstrip(), end="\n", file = f)

        for i in range(len(self.tagList)):
            senString = ""
            senString += self.tagList[i] + '\t'
            for j in range(len(self.tagList)):
                senString += str(self.matrix[i][j]) + '\t'
            #print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        print("Return the tagger's precision when predicting tag t_i")
        return self.TP[tagTi] / (self.TP[tagTi] + self.FP[tagTi])

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        print("Return the tagger's recall for correctly predicting gold tag t_j")
        return self.TP[tagTj] / (self.TP[tagTj] + self.FN[tagTj])


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getRecall('NNP'))
        print("Precision for tag NNP: ", eval.getPrecision('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")
