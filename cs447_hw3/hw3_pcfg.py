import sys
import os
import math

# The start symbol for the grammar
TOP = "TOP"

'''
A grammatical Rule has a probability and a parent category, and is
extended by UnaryRule and BinaryRule
'''


class Rule:

    def __init__(self, probability, parent):
        self.prob = probability
        self.parent = parent

    # Factory method for making unary or binary rules (returns None otherwise)
    @staticmethod
    def createRule(probability, parent, childList):
        if len(childList) == 1:
            return UnaryRule(probability, parent, childList[0])
        elif len(childList) == 2:
            return BinaryRule(probability, parent, childList[0], childList[1])
        return None

    # Returns a tuple containing the rule's children
    def children(self):
        return ()

'''
A UnaryRule has a probability, a parent category, and a child category/word
'''


class UnaryRule(Rule):

    def __init__(self, probability, parent, child):
        Rule.__init__(self, probability, parent)
        self.child = child

    # Returns a singleton (tuple) containing the rule's child
    def children(self):
        return (self.child,)  # note the comma; (self.child) is not a tuple

'''
A BinaryRule has a probability, a parent category, and two children
'''


class BinaryRule(Rule):

    def __init__(self, probability, parent, leftChild, rightChild):
        Rule.__init__(self, probability, parent)
        self.leftChild = leftChild
        self.rightChild = rightChild

    # Returns a pair (tuple) containing the rule's children
    def children(self):
        return (self.leftChild, self.rightChild)

'''
An Item stores the label and Viterbi probability for a node in a parse tree
'''


class Item:

    def __init__(self, label, prob, numParses):
        self.label = label
        self.prob = prob
        self.numParses = numParses

    # Returns the node's label
    def toString(self):
        return self.label

'''
A LeafItem is an Item that represents a leaf (word) in the parse tree (ie, it
doesn't have children, and it has a Viterbi probability of 1.0)
'''


class LeafItem(Item):

    def __init__(self, word):
        # using log probabilities, this is the default value (0.0 = log(1.0))
        Item.__init__(self, word, 0.0, 1)

'''
An InternalNode stores an internal node in a parse tree (ie, it also
stores pointers to the node's child[ren])
'''


class InternalItem(Item):

    def __init__(self, category, prob, children=(), leaf = False):
        Item.__init__(self, category, prob, 0)
        self.children = children
        # Your task is to update the number of parses for this InternalItem
        # to reflect how many possible parses are rooted at this label
        # for the string spanned by this item in a chart
        self.numParses = -1  # dummy numParses value; this should not be -1!
        self.isLeaf = False
        if len(self.children) > 2:
            print("Warning: adding a node with more than two children (CKY may not work correctly)")

    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        if self.isLeaf == True:
            return "( " + self.label + " " + \
                str(self.children[0]) + " )"

        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"

'''
A Cell stores all of the parse tree nodes that share a common span

Your task is to implement the stubs provided in this class
'''


class Cell:

    def __init__(self):
        self.items = {}

    def addItem(self, item):
        # Add an Item to this cell
        pass

    def getItem(self, label):
        # Return the cell Item with the given label
        pass

    def getItems(self):
        # Return the items in this cell
        pass

'''
A Chart stores a Cell for every possible (contiguous) span of a sentence

Your task is to implement the stubs provided in this class
'''


class Chart:

    def __init__(self, sentence):
        # Initialize the chart, given a sentence
        pass

    def getRoot(self):
        # Return the item from the top cell in the chart with
        # the label TOP
        pass

    def getCell(self, i, j):
        # Return the chart cell at position i, j
        pass

'''
A PCFG stores grammatical rules (with probabilities), and can be used to
produce a Viterbi parse for a sentence if one exists
'''


class PCFG:

    def __init__(self, grammarFile, debug=False):
        # in ckyRules, keys are the rule's RHS (the rule's children, stored in
        # a tuple), and values are the parent categories
        self.ckyRules = {}
        self.debug = debug                  # boolean flag for debugging
        # reads the probabilistic rules for this grammar
        self.readGrammar(grammarFile)
        # checks that the grammar at least matches the start symbol defined at
        # the beginning of this file (TOP)
        self.topCheck()

    '''
    Reads the rules for this grammar from an input file
    '''

    def readGrammar(self, grammarFile):
        if os.path.isfile(grammarFile):
            file = open(grammarFile, "r")
            for line in file:
                raw = line.split()
                # reminder, we're using log probabilities
                prob = math.log(float(raw[0]))
                parent = raw[1]
                children = raw[
                    3:]   # Note: here, children is a list; below, rule.children() is a tuple
                rule = Rule.createRule(prob, parent, children)
                if rule.children() not in self.ckyRules:
                    self.ckyRules[rule.children()] = set([])
                self.ckyRules[rule.children()].add(rule)

    '''
    Checks that the grammar at least matches the start symbol (TOP)
    '''

    def topCheck(self):
        for rhs in self.ckyRules:
            for rule in self.ckyRules[rhs]:
                if rule.parent == TOP:
                    return  # TOP generates at least one other symbol
        if self.debug:
            print("Warning: TOP symbol does not generate any children (grammar will always fail)")

    '''
    Your task is to implement this method according to the specification. You may define helper methods as needed.

    Input:        sentence, a list of word strings
    Returns:      The root of the Viterbi parse tree, i.e. an InternalItem with label "TOP" whose probability is the Viterbi probability.
                   By recursing on the children of this node, we should be able to get the complete Viterbi tree.
                   If no such tree exists, return None\
    '''


    def CKY(self, sentence):
        # change all the words into determiners first
        # label, prob, children, numParses
        determineWords = [InternalItem("", 0.0) for x in range(len(sentence))]

        #print(self.ckyRules.keys())
        for i in range(len(sentence)):
            curWord = []
            curWord.append(sentence[i])
            maxProb = -100.0
            for rule in self.ckyRules[tuple(curWord)]:
                if rule.prob > maxProb:
                    maxProb = rule.prob
                    determineWords[i].label = rule.parent
                    determineWords[i].prob = rule.prob
                    determineWords[i].children = rule.children()
                    determineWords[i].numParses = 1.0 #slen(self.ckyRules[tuple(curWord)])


        matrix = [[InternalItem("", -100.0) for x in range(len(sentence))] for y in range(len(sentence))]
        for i in range(len(sentence)):
            matrix[i][i] = determineWords[i]
            matrix[i][i].isLeaf = True

        for col in range(1, len(sentence)):
            endRow = len(sentence) - col # this endRow is not reachable
            for row in range(0, endRow):
                maxProb = -100.0
                curRow = 0 + row
                curCol = col + row
                matrix[curRow][curCol].numParses = 0.0
                # how many different cutting method
                for cut in range(curRow, curCol):
                    label1 = matrix[curRow][cut].label
                    label2 = matrix[cut + 1][curCol].label
                    curKey = []
                    curKey.append(label1)
                    curKey.append(label2)
                    # this label combination does not exist
                    if tuple(curKey) not in self.ckyRules:
                        continue
                    # what is the prob of the current cut itself
                    matrix[curRow][curCol].numParses += matrix[curRow][cut].numParses * matrix[cut + 1][curCol].numParses
                    for item in self.ckyRules[tuple(curKey)]:
                        thisCutProb = item.prob

                    if maxProb < matrix[curRow][cut].prob + matrix[cut + 1][curCol].prob + thisCutProb:

                        maxProb = matrix[curRow][cut].prob + matrix[cut + 1][curCol].prob + thisCutProb
                        matrix[curRow][curCol].prob = maxProb

                        matrix[curRow][curCol].label = list(self.ckyRules[tuple(curKey)])[0].parent

                        childrenOfCur = []
                        childrenOfCur.append(matrix[curRow][cut])
                        childrenOfCur.append(matrix[cut + 1][curCol])
                        matrix[curRow][curCol].children = tuple(childrenOfCur)

        finalTuple = []
        finalTuple.append(matrix[0][len(sentence) - 1])
        finalItem = InternalItem(TOP, matrix[0][len(sentence) - 1].prob, finalTuple)
        finalItem.numParses = matrix[0][len(sentence) - 1].numParses
        return finalItem


if __name__ == "__main__":
    pcfg = PCFG('toygrammar.pcfg')
    sen = "the man eats the sushi".split()

    tree = pcfg.CKY(sen)
    if tree is not None:
        print(tree.toString())
        print("Probability: " + str((tree.prob)))
        print("Num parses: " + str(tree.numParses))
    else:
        print("Parse failure!")
