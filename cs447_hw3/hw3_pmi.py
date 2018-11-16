########################################
## CS447 Natural Language Processing  ##
##           Homework 3               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Use pointwise mutual information to compare words in the movie corpora
##
import os.path
import sys
import math
from operator import itemgetter
from collections import defaultdict
import heapq

# ----------------------------------------
#  Data input 
# ----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r")  # open the input file in read-only mode
        i = 0  # this is just a counter to keep track of the sentence numbers
        corpus = []  # this will become a list of sentences
        print("Reading file", f, "...")
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence) # append this list as an element to the list of sentences
            # if i % 1000 == 0:
            #    sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print("Error: corpus file", f, "does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit()  # exit the script

# --------------------------------------------------------------
# PMI data structure
# --------------------------------------------------------------
class PMI:
    # Given a corpus of sentences, store observations so that PMI can be calculated efficiently
    def __init__(self, corpus):
        #print("\nYour task is to add the data structures and implement the methods necessary\
         #to efficiently get the pairwise PMI of words from a corpus")
        self.sentenceCount = len(corpus)
        self.numWords = defaultdict()
        self.numBiWords = defaultdict()
        for sen in corpus:
            wordSet = set(sen)
            for word in wordSet:
                if word in self.numWords:
                    self.numWords[word] += 1
                else:
                    self.numWords[word] = 1

            wordList = list(wordSet)
            for i in range(len(wordList)):
                for j in range(i + 1, len(wordList)):
                    curKey = self.pair(wordList[i], wordList[j])
                    if curKey in self.numBiWords:
                        self.numBiWords[curKey] += 1
                    else:
                        self.numBiWords[curKey] = 1

    # Return the pointwise mutual information (based on sentence (co-)occurrence frequency) for w1 and w2
    def getPMI(self, w1, w2):
        #print("\nSubtask 1: calculate the PMI for a pair of words")
        p_w1 = self.numWords[w1] / self.sentenceCount
        p_w2 = self.numWords[w2] / self.sentenceCount
        if self.pair(w1, w2) in self.numBiWords:
            p_w1_w2 = self.numBiWords[self.pair(w1, w2)] / self.sentenceCount
        else:
            p_w1_w2 = 0.0

        if p_w1_w2 == 0.0:
            return float('-inf')
        return math.log(p_w1_w2, 2) - math.log(p_w1, 2) - math.log(p_w2, 2)

    # Given a frequency cutoff k, return the list of observed words that appear in at least k sentences
    def getVocabulary(self, k):
        #print("\nSubtask 2: return the list of words where a word is in the list iff it occurs in at least k sentences")
        #return ["the", "a", "to", "of", "in"]
        commonWord = []
        for word, count in self.numWords.items():
            if count >= k:
                commonWord.append(word)
        return commonWord

    # Given a list of words and a number N, return a list of N pairs of words that have the highest PMI
    # (without repeated pairs, and without duplicate pairs (wi, wj) and (wj, wi)).
    # Each entry in the list should be a triple (pmiValue, w1, w2), where pmiValue is the
    # PMI of the pair of words (w1, w2)
    def getPairsWithMaximumPMI(self, words, N):
        #print("\nSubtask 3: given a list of words and a number N, find N pairs of words with the greatest PMI")
        resTriple = []
        countHeap = 0

        # this is the min heap
        for i in range(0, len(words)):
            for j in range(i + 1, len(words)):
                curTriple = []
                curTriple.append(self.getPMI(words[i], words[j]))
                curTriple.append(min(words[i], words[j]))
                curTriple.append(max(words[i], words[j]))

                if countHeap < N:
                    heapq.heappush(resTriple, curTriple)
                else:
                    if resTriple[0][0] < curTriple[0]:
                        heapq.heappop(resTriple)
                        heapq.heappush(resTriple, curTriple)

                countHeap += 1

        # do the max heap
        heapq._heapify_max(resTriple)
        return resTriple

    #-------------------------------------------
    # Provided PMI methods
    #-------------------------------------------
    # Writes the first numPairs entries in the list of wordPairs to a file, along with each pair's PMI
    def writePairsToFile(self, numPairs, wordPairs, filename): 
        f=open(filename, 'w+')
        count = 0
        for (pmiValue, wi, wj) in wordPairs:
            if count > numPairs:
                break
            count += 1
            print("%f %s %s" % (pmiValue, wi, wj), end="\n", file=f)

    # Helper method: given two words w1 and w2, returns the pair of words in sorted order
    # That is: pair(w1, w2) == pair(w2, w1)
    def pair(self, w1, w2):
        return (min(w1, w2), max(w1, w2))

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('movies.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("luke", "vader")
    print("  PMI of \"luke\" and \"vader\": ", lv_pmi)
    numPairs = 10
    k = 200
    # for k in 2, 5, 10, 50, 100, 200:
    commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
    wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
    pmi.writePairsToFile(numPairs, wordPairsWithGreatestPMI, "pairs_minFreq="+str(k)+".txt")
