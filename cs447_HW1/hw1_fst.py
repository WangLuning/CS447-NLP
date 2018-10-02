from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
E = set("e")
U = set("u")
I = set("i")
DOUBLE = set("nptr")

##########
# every state is like a DFS process, if you do not end it, it will go down
##########

# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("q1") # a non-accepting state
    f.addState("q_ing") # a non-accepting state
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)
    f.addState("with_e")
    f.addState("with_i")
    f.addState("with_ie")
    f.addState("doubling_last")
    f.addState("q_not_doubled")
    f.addState("with_auo")
    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself: 
    f.addSetTransition("q0", AZ, "q1")
    # AZ-E =  the set AZ without the elements in the set E
    #f.addSetTransition("q1", AZ, "q1")

    # get rid of this transition! (it overgenerates):
    f.addSetTransition("q1", AZ - VOWS, "q1")


    ## rule 1: omitting the last e
    f.addTransition("q1", "e", "", "with_e")

    f.addTransition("with_e", "", "ing", "q_EOW")

    # rule 1 nested rule 3: doubling the last char
    for i in range(0, 26):
        # DFS: one to doubling the char, one not to double
        if chr(i + 97) in DOUBLE:
            f.addTransition("with_e", chr(i + 97), "e" + chr(i + 97) + chr(i + 97), "doubling_last")
            f.addTransition("with_e", chr(i + 97), "e" + chr(i + 97), "q_not_doubled")
        else:
            f.addTransition("with_e", chr(i + 97), "e" + chr(i + 97), "q1")
        
    # rule 3: how to deal with doubling last cons or not
    f.addTransition("doubling_last", "", "ing", "q_EOW")
    f.addSetTransition("q_not_doubled", AZ - VOWS, "q1")
    f.addTransition("q_not_doubled", "e", "", "with_e")
    f.addTransition("q_not_doubled", "i", "", "with_i")
    f.addSetTransition("q_not_doubled", VOWS - I - E, "with_auo")

    ## rule 2: changing ie into y
    f.addTransition("q1", "i", "", "with_i")

    # if the following char is not e, give i back:
    for i in range(0, 26):
        if i != 4:
            if chr(i + 97) in DOUBLE:
                f.addTransition("with_i", chr(i + 97), "i" + chr(i + 97) + chr(i + 97), "doubling_last")
                f.addTransition("with_i", chr(i + 97), "i" + chr(i + 97), "q_not_doubled")
            else:
                f.addTransition("with_i", chr(i + 97), "i" + chr(i + 97), "q1")

    f.addTransition("with_i", "e", "", "with_ie")

    # changing ie to ying
    f.addTransition("with_ie", "", "ying", "q_EOW")

    # if there is sth else after ie, give ie back:
    for i in range(0, 26):
            f.addTransition("with_ie", chr(i + 97), "ie" + chr(i + 97), "q1")

    f.addSetTransition("with_ie", AZ, "q1")

    ## rule 3 for other vows like a u o
    f.addSetTransition("q1", VOWS - E - I, "with_auo")

    # if the following char is not e, give i back:
    for i in range(0, 26):
        if chr(i + 97) in DOUBLE:
            f.addTransition("with_auo", chr(i + 97), chr(i + 97) + chr(i + 97), "doubling_last")
            f.addSetTransition("with_auo", chr(i + 97), "q_not_doubled")
        else:
            f.addSetTransition("with_auo", chr(i + 97), "q1")

    # map the empty string to ing: 
    f.addTransition("q1", "", "ing", "q_EOW")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
