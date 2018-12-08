from collections import defaultdict
import math
from math import log

# Constant for NULL word at position zero in target sentence
NULL = "NULL"

# Your task is to finish implementing IBM Model 1 in this class
class IBMModel1:

    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.

        self.transLengthProb = defaultdict(lambda : defaultdict(float))
        self.transWordProb = defaultdict(lambda : defaultdict(float))
        self.counts = defaultdict(lambda : defaultdict(float))
        # Read the corpus
        self.initialize(trainingCorpusFile);

        # Initialize any additional data structures here (e.g. for probability model)

    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
    def initialize(self, fileName):
        f = open(fileName)
        i = 0
        j = 0;
        tTokenized = ();
        fTokenized = ();
        for s in f:
            if i == 0:
                tTokenized = s.split()
                # Add null word in position zero
                tTokenized.insert(0, NULL)
                self.tCorpus.append(tTokenized)
            elif i == 1:
                fTokenized = s.split()
                self.fCorpus.append(fTokenized)
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] +1
            else:
                i = -1
                j += 1
                #if j % 1000 == 0:
                    #print('another 1000 sentences processed')
            i +=1
        f.close()
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon=0.01):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities()        # <you need to implement initializeTranslationProbabilities()>
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>

        oldL = 0.0
        for i in range(numIterations):
            print ("Starting training iteration "+str(i))
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>

            # from here we use convergence to stop the loop if the accuracy is high enough
            useConvergenceModel = False
            if useConvergenceModel:
                newL = 0.0
                for idx in range(len(self.tCorpus)):
                    m = len(self.fCorpus[idx])
                    n = len(self.tCorpus[idx])
                    tSen = self.tCorpus[idx]
                    fSen = self.fCorpus[idx]

                    newL += log(self.getTranslationLengthProbability(m, n))
                    newL -= m * log(n + 1)

                    for f in fSen:
                        tmp = 0.0
                        for e in tSen:
                            tmp += self.transWordProb[f][e]
                        newL += log(tmp)

                newL /= len(self.tCorpus)
                print("step towards convergence: " + str(oldL - newL))
                if i == 0:
                    oldL = newL
                    continue

                if abs(oldL - newL) < convergenceEpsilon:
                    break
                else:
                    oldL = newL

    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self):
        # Implement this method
        for i in range(len(self.tCorpus)):
            n = len(self.tCorpus[i])
            m = len(self.fCorpus[i])
            if m not in self.transLengthProb.keys():
                self.transLengthProb[m] = {}
            if n not in self.transLengthProb[m].keys():
                self.transLengthProb[m][n] = 1
            else:
                self.transLengthProb[m][n] += 1.0

        for foreignKey in self.transLengthProb.keys():
            totalCount = float(sum(self.transLengthProb[foreignKey].values()))
            for key in self.transLengthProb[foreignKey]:
                self.transLengthProb[foreignKey][key] /= totalCount

    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self):
        # Implement this method
        for t in self.trans.keys():
            total = float(sum(self.trans[t].values()))
            for f in self.trans[t].keys():
                self.transWordProb[f][t] = self.trans[t][f] / total
        #print(self.transWordProb['No']['Don\''])

    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        # Implement this method
        senCounts = []
        self.counts = {}
        # for each sentence
        for index, e in enumerate(self.tCorpus):
            senCounts.append(defaultdict(lambda: defaultdict(int)))          
            # for each word in fSen
            for f_j in self.fCorpus[index]:
                totalCount = 0.0
                # for each word in tSen
                for e_i in e:
                    totalCount += self.getWordTranslationProbability(f_j, e_i)
                for e_i in e:
                    eachWordProb = self.getWordTranslationProbability(f_j, e_i)
                    senCounts[index][e_i][f_j] = eachWordProb / totalCount
        
        for index in range(len(senCounts)):
            for e in senCounts[index]:
                for f in senCounts[index][e]:
                    if e not in self.counts.keys():
                        self.counts[e] = {}
                    if f not in self.counts[e].keys():
                        self.counts[e][f] = 0.0
                    self.counts[e][f] += senCounts[index][e][f]

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        # Implement this method
        
        for e in self.counts:
            totalNum = float(sum(self.counts[e].values()))
            #if e == 'NULL':
                #print(totalNum)
            for f in self.counts[e]:
                self.transWordProb[f][e] = self.counts[e][f] / totalNum
        #print(self.counts['Don\'']['No'])
        #print(self.transWordProb['No']['Don\''])

    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        # double check if we have null in the beginning of target sentence
        if tSen[0] != NULL:
            tSen.insert(0, NULL)
        
        correctAlignment = []
        for j in range(len(fSen)):
            maxProb, alignIdx = 0.0, 0

            for i in range(len(tSen)):
                if self.transWordProb[fSen[j]][tSen[i]] > maxProb:
                    maxProb = self.transWordProb[fSen[j]][tSen[i]]
                    alignIdx = i

            correctAlignment.append(alignIdx)                    
        
        return correctAlignment # Your code above should return the correct alignment instead

    # Return q(tLength | fLength), the probability of producing an English sentence of length tLength given a non-English sentence of length fLength
    # (Can either return log probability or regular probability)
    def getTranslationLengthProbability(self, fLength, tLength):
        # Implement this method
        if fLength in self.transLengthProb.keys():
            if tLength in self.transLengthProb[fLength].keys():
                return self.transLengthProb[fLength][tLength]
        return 0.0

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        if f_j in self.transWordProb.keys():
            if e_i in self.transWordProb[f_j].keys():
                return self.transWordProb[f_j][e_i]
        return 0.0

    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        # Implement this method (make your output legible and informative)
        for key in self.transLengthProb:
            for key2 in self.transLengthProb[key]:
                lengthFile.write("foreign length is " + str(key2) + " english length is " + str(key) + ":" + str(self.transLengthProb[key][key2]) + '\n')

        for key in self.transWordProb:
            for key2 in self.transWordProb[key]:
                translateProbFile.write("foreign word is " + str(key) + " english word is " + str(key2) + ":" + str(self.transWordProb[key][key2]) + '\n')

        lengthFile.close();
        translateProbFile.close()

# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # Initialize model
    model = IBMModel1('eng-spa.txt')
    # Train model
    model.trainUsingEM(10);
    #model.printModel('after_training')
    # Use model to get an alignment
    fSen = 'No pierdas el tiempo por el camino .'.split()
    tSen = 'Don\' t dawdle on the way'.split()
    alignment = model.align(fSen, tSen);
    print (prettyAlignment(fSen, tSen, alignment))
