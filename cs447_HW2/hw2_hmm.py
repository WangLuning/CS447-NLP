########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

class matrix_item:
    def __init__(self):
        self.pointer = -1
        self.number = 0.0

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
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

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###
        self.test_word_counts = defaultdict(float)
        self.old_word_counts = defaultdict(float)
        self.tag_counts = defaultdict(float)
        self.tag_bicounts = defaultdict(float)
        self.word_counts = defaultdict(float)
        self.word_tag_counts = defaultdict(float)
        self.TagVocabBeforeIt = defaultdict(float)
        self.WordVocabBeforeIt = defaultdict(float)
        self.total_tag = 0.0
        
    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        #print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")

        # deal with rare data, changing words occurrence < 5 to UNK
        
        for sen in data:
            for item in sen:
                self.old_word_counts[item.word] += 1.0

        for sen in data:
            for i in range(len(sen)):
                words = sen[i].word
                if self.old_word_counts[words] < 5:
                    sen[i].word = UNK
        

        for sen in data:
            previtem = TaggedWord("@_@");
            for item in sen:
                self.total_tag += 1.0
                self.word_counts[item.word] += 1.0
                self.word_tag_counts[item.word + " " + item.tag] += 1.0
                self.tag_counts[item.tag] += 1.0
                if previtem.tag == '@':
                    previtem = item
                    continue
                self.tag_bicounts[item.tag + " " + previtem.tag] += 1.0
                previtem = item

        for key in list(self.tag_counts.keys()):
            for key2 in self.word_tag_counts.keys():
                if key2.find(" " + key) != -1:
                    self.WordVocabBeforeIt[key] += 1.0
            for key2 in self.tag_bicounts.keys():
                if key2.find(" " + key) != -1:
                    self.TagVocabBeforeIt[key] += 1.0
    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        #self.viterbi(data[0])

        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            #print(senString)
            print(senString.rstrip(), end="\n", file=f)
        

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        #print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)

        m = len(self.tag_counts)
        n = len(words)
        matrix = [[matrix_item() for x in range(m)] for y in range(n)]

        tags = list(self.tag_counts.keys())
        vocabs = list(self.word_counts.keys())
        #print (tags[-1])

        for j in range(m):
            curword = UNK
            if words[0] in vocabs:
                curword = words[0]

            matrix[0][j].number = (self.tag_counts[tags[j]] + 1.0) \
                / (self.total_tag + len(self.tag_counts)) \
                * (self.word_tag_counts[curword + " " + tags[j]]) \
                / (self.tag_counts[tags[j]])

        for i in range(1, n):
            # this is defininte tag ",", no need to calculate, we can assign all other probabilities to 0
            if words[i] == ',':
                for j in range(m):
                    if tags[j] != ',':
                        matrix[i][j].number = 0.0
                    else:
                        matrix[i][j].number = 1.0
                        maxPossible = 0.0
                        maxPossiblePointer = 0
                        for k in range(m):
                            tmp = matrix[i - 1][k].number \
                                * (self.tag_bicounts[tags[j] + " " + tags[k]]) \
                                / (self.tag_counts[tags[k]])
                        
                            if tmp > maxPossible:
                                maxPossible = tmp
                                maxPossiblePointer = k

                        matrix[i][j].pointer = maxPossiblePointer
                continue

            curword = UNK
            if words[i] in vocabs:
                curword = words[i]

            # we don't know what the tag is, need to calculate           
            for j in range(m):
                maxPossible = 0.0
                maxPossiblePointer = 0
                for k in range(m):
                    tmp = matrix[i - 1][k].number \
                        * (self.tag_bicounts[tags[j] + " " + tags[k]] + 1.0) \
                        / (self.tag_counts[tags[k]] + self.TagVocabBeforeIt[tags[k]])
                        
                    if tmp > maxPossible:
                        maxPossible = tmp
                        maxPossiblePointer = k

                matrix[i][j].number = maxPossible * (self.word_tag_counts[curword + " " + tags[j]]) \
                                    / (self.tag_counts[tags[j]])
                matrix[i][j].pointer = maxPossiblePointer

        # find the tag of the last word
        maxlastword = 0
        lastwordtag = 0
        for j in range(m):
            if matrix[n - 1][j].number > maxlastword:
                maxlastword = matrix[n - 1][j].number
                lastwordtag = j

        tagger_sen = []
        tagger_sen.insert(0, tags[lastwordtag])
        tagger_index = lastwordtag

        for i in range(n - 1, 0, -1):
            tagger_index = matrix[i][tagger_index].pointer
            tagger_sen.insert(0, tags[tagger_index])

        return tagger_sen # this returns a dummy list of "NULL", equal in length to words

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
