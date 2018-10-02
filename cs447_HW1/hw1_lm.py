########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import math
import random
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i == 300:
                break
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef

def getVocal(corpus):
    vocab = set()
    for sent in corpus:
        for word in sent:
            vocab.add(word)
    return vocab

# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.bicounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        self.start_counting = self.start_count(corpus)
    #enddef

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor

        prevword = start
        for sen in corpus:
            for word in sen:
                if word == start:
                    prevword = start
                    continue
                self.bicounts[word +  " " + prevword] += 1.0
                prevword = word

    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    def start_count(self, corpus):
        count = 0.0
        for sen in corpus:
            for word in sen:
                if word == start:
                    count += 1.0
        return count

    def prob(self, word):
        return self.counts[word] / self.total

    # Returns the probability of word in the distribution
    def probBi(self, word, prevword):
        return self.bicounts[word + " " + prevword]/self.counts[prevword]

    #enddef

    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

    def drawBi(self, prevword):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.probBi(word, prevword)
            if rand <= 0.0:
                return word
        return "Not uniform distribution exception"
        print("error")
        #endif
    #endfor

    def drawBi_start(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.bicounts[word + " " + start] / self.start_counting
            if rand <= 0.0:
                return word

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        perplexity = 0.0

        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                perplexity += math.log(self.prob(word))
                prevword = word

        perplexity = math.exp(-1 / self.total * perplexity)
        print("Implement the getCorpusPerplexity method: " + str(perplexity))
        return perplexity

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        generatedCorpus = []

        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            generatedCorpus.append(sen)
            stringGenerated = str(prob) + " " + " ".join(sen)
            print(stringGenerated, end="\n", file=filePointer)
        return generatedCorpus
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def generateSentence(self):
        sent = []
        cur = start
        sent.append(cur)
        cur = self.draw()

        while cur != end:
            sent.append(cur)
            cur = self.draw()
        
        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        #print("Implement the getSentenceProbability method in each subclass")
        probab = 0.0

        for word in sen:
            if word == start:
                continue
            if self.counts[word] == 0:
                return 0
            probab = probab + math.log(self.counts[word]) - math.log(self.total)
        return math.exp(probab)
    #enddef

    #endddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def prob(self, word):
        return (self.counts[word] + 1) / (self.total + len(self.counts))

    def generateSentence(self):
        sent = []
        cur = start

        while cur != end:
            sent.append(cur)
            cur = self.draw()
        
        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        #print("Implement the getSentenceProbability method in each subclass")
        probab = 0.0

        for word in sen:
            if word == start:
                continue
            probab = probab + math.log(self.counts[word] + 1) - math.log(self.total + len(self.counts))
        return math.exp(probab)
    #enddef

    #endddef
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def generateSentence(self):
        sent = []
        sent.append(start)
        cur = self.drawBi_start()

        while cur != end:
            sent.append(cur)
            cur = self.drawBi(cur)
        
        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        #print("Implement the getSentenceProbability method in each subclass")
        probab = 0.0
        prevword = "#"
        for word in sen:
            if prevword == start:
                probab += math.log(self.bicounts[word + " " + prevword]) \
                    - math.log(self.start_counting)
            elif word != start:
                probab += math.log(self.bicounts[word + " " + prevword]) \
                    - math.log(self.counts[prevword])

            prevword = word

        return math.exp(probab)
    #enddef

    def getCorpusPerplexity(self, corpus):
        perplexity = 0.0
        
        prevword = "#"
        for sen in corpus:
            for word in sen:
                if word == start:
                    prevword = word
                    continue
                if self.bicounts[word + " " + prevword] == 0:
                    return float("Inf")
                if prevword == start:
                    perplexity += math.log(self.bicounts[word + " " + prevword]) - math.log(self.start_counting)
                    continue
                perplexity += math.log(self.probBi(word, prevword))
                prevword = word

        perplexity = math.exp(-1 / self.total * perplexity)
        print("Implement the getCorpusPerplexity method: " + str(perplexity))
        return perplexity
    #endddef
#endclass

# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothedBigramModelAD(LanguageModel):
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.bicounts = defaultdict(float)
        self.total = 0.0
        self.S_ww_count = defaultdict(float)
        self.train(corpus)
        self.discounting = self.discounting_factor()
        self.start_counting = self.start_count(corpus)
        self.discount_index = 0.0
        self.first = list(self.counts.keys())[0]
        for k in self.counts.keys():
            self.discount_index += self.probBi_AD(k, self.first)
        print(self.discount_index)
    #enddef

    def discounting_factor(self):
        n1 = 0.0
        n2 = 0.0
        for word in self.bicounts.keys():
            if self.bicounts[word] == 1.0:
                n1 += 1.0
            elif self.bicounts[word] == 2.0:
                n2 += 1.0

        D = n1 / (n1 + 2.0 * n2)
        return D

    def S_ww(self, prevword):
        count = 0.0
        for k in self.bicounts.keys():
            if k.find(" " + prevword) != -1:
                count += 1.0
        return count

    def prob(self, word):
        return (self.counts[word] + 1) / (self.total + len(self.counts))

    def probBi_AD(self, word, prevword):
        if prevword is None:
            print("prev none error")
        if self.counts[prevword] == 0:
            print("non existing prevword: " + prevword)

        return max(0, self.bicounts[word + " " + prevword] - self.discounting)/self.counts[prevword] \
            + self.discounting * self.S_ww(prevword) / self.counts[prevword] \
            * self.prob(word)

    def drawBi(self, prevword):
        rand = random.random()
        for word in self.counts.keys():
            if prevword == start:
                continue
            rand -= self.probBi_AD(word, prevword) / self.discount_index
            if rand <= 0.0:
                return word
        return end
    #endfor

    def generateSentence(self):
        sent = []
        sent.append(start)
        cur = self.drawBi_start()

        while cur != end:
            sent.append(cur)
            cur = self.drawBi(cur)
        
        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        #print("Implement the getSentenceProbability method in each subclass")
        probab = 0.0
        prevword = start
        for word in sen:
            if word == start:
                prev = word
                continue
            if prevword == start:
                probab += math.log(max(0, self.bicounts[word + " " + prevword] - self.discounting)/self.start_counting + self.discounting/self.start_counting * self.S_ww(prevword) * self.prob(word))
                continue
            probab += math.log(self.probBi_AD(word, prevword))
            prevword = word

        return math.exp(probab)
    #enddef

    def getCorpusPerplexity(self, corpus):
        perplexity = 0.0
        
        prevword = start
        for sen in corpus:
            for word in sen:
                if word == start:
                    prevword = word
                    continue
                if prevword == start:
                    perplexity += math.log(max(0, self.bicounts[word + " " + prevword] - self.discounting)/self.start_counting + self.discounting/self.start_counting*self.S_ww(prevword)*self.prob(word))
                    continue
                perplexity += math.log(self.probBi_AD(word, prevword))
                prevword = word

        perplexity = math.exp(-1 / self.total * perplexity)
        print("Implement the getCorpusPerplexity method: " + str(perplexity))
        return perplexity
#endclass

# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(SmoothedBigramModelAD):
    def pc(self, word):
        num_count = 0.0
        denum_count = len(self.bicounts) * 1.0

        for k in self.bicounts.keys():
            if k.find(word + " ") != -1:
                num_count += 1.0
        return num_count / denum_count

    def probBi_KN(self, word, prevword):
        #return self.probBi(word, prevword)
        if self.counts[prevword] == 0:
            return 1.0
        return self.discounting/self.counts[prevword]*self.S_ww(prevword)*self.pc(word)

    def drawBi(self, prevword):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.probBi_KN(word, prevword) / self.discount_index
            if rand <= 0.0:
                return word
    #endfor

    def getSentenceProbability(self, sen):
        #print("Implement the getSentenceProbability method in each subclass")
        probab = 0.0
        prevword = start
        for word in sen:
            if word == start:
                prev = word
                continue
            if prevword == start:
                probab += math.log(max(0, self.bicounts[word + " " + prevword] - self.discounting)/self.start_counting + self.discounting/self.start_counting * self.S_ww(prevword) * self.pc(word))
                continue
            probab += math.log(self.probBi_KN(word, prevword))
            prevword = word

        return math.exp(probab)
    #enddef

    def getCorpusPerplexity(self, corpus):
        perplexity = 0.0
        
        prevword = start
        for sen in corpus:
            for word in sen:
                if word == start:
                    prevword = word
                    continue
                if prevword == start:
                    perplexity += math.log(max(0, self.bicounts[word + " " + prevword] - self.discounting)/self.start_counting + self.discounting/self.start_counting*self.S_ww(prevword)*self.pc(word))
                    continue
                perplexity += math.log(self.probBi_KN(word, prevword))
                prevword = word

        perplexity = math.exp(-1 / self.total * perplexity)
        print("Implement the getCorpusPerplexity method: " + str(perplexity))
        return perplexity
#endclass


#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    vocab = set()
    # Please write the code to create the vocab over here before the function preprocessTest
    vocab = getVocal(trainCorpus)
    #print("length of vocab" + str(len(vocab)))
    #print(vocab)

    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # Run sample unigram dist code
    #print("Unsmoothed UnigramModel output:")
    UnigramModel = UnigramModel(trainCorpus)
    uniGenCorpus = UnigramModel.generateSentencesToFile(20, "unigram_output.txt")
    UnigramModel.getCorpusPerplexity(posTestCorpus)
    UnigramModel.getCorpusPerplexity(negTestCorpus)
    print("unigram finished")

    # Run smoothed unigram model
    SmoothedUnigramModel = SmoothedUnigramModel(trainCorpus)
    smUniGenCorpus = SmoothedUnigramModel.generateSentencesToFile(20, "smooth_unigram_output.txt")
    SmoothedUnigramModel.getCorpusPerplexity(posTestCorpus)
    SmoothedUnigramModel.getCorpusPerplexity(negTestCorpus)
    print("sm unigram finished")

    # Run unsmoothed bigram model
    BigramModel = BigramModel(trainCorpus)
    biGenCorpus = BigramModel.generateSentencesToFile(20, "bigram_output.txt")
    BigramModel.getCorpusPerplexity(posTestCorpus)
    BigramModel.getCorpusPerplexity(negTestCorpus)
    print("bigram finished")

    # Run AD smoothed bigram model
    SmoothedBigramModelAD = SmoothedBigramModelAD(trainCorpus)
    biADGenCorpus = SmoothedBigramModelAD.generateSentencesToFile(20, "smooth_bigram_ad_output.txt")
    SmoothedBigramModelAD.getCorpusPerplexity(posTestCorpus)
    SmoothedBigramModelAD.getCorpusPerplexity(negTestCorpus)
    print("sm ad bigram finished")

    # Run KN smoothed bigram model
    SmoothedBigramModelKN = SmoothedBigramModelKN(trainCorpus)
    biKNGenCorpus = SmoothedBigramModelKN.generateSentencesToFile(20, "smooth_bigram_kn_output.txt")
    SmoothedBigramModelKN.getCorpusPerplexity(posTestCorpus)
    SmoothedBigramModelKN.getCorpusPerplexity(negTestCorpus)
    print("sm kn bigram finished")

