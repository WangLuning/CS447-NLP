I have implemented the convergence EC part.
In the trainUsingEM function, there is a variable called 'useConvergenceModel'
If set to True, we will use the convergence
If set to False, we will not use the convergence


#1. discuss where your model improved improved on the initial parameters during training (e.g. on rare words, function words, punctuation, phrases, etc.)
Answer:
For punctuation, '.':
In the first iteration every word has similar possibility towards '.' around 0.2, but with the increase of iterations of the EM algorithm we can see 'NULL' is taking on more and more possibility.
For rare words, 'PUTA':
In the first iteration it has an extremely high probability for it to be translated to 'son',
With the increase of iteration it remains far above other probabilities since it is rare, so it is unlikely to change.

#2. examine some of the alignments that your model produces (especially on sentences longer than 5 words).
Answer:
We will use the example given in our file.
With 10 iterations:
0  No                  ==>    Don'
1  pierdas             ==>    dawdle
2  el                  ==>    the
3  tiempo              ==>    dawdle
4  por                 ==>    dawdle
5  el                  ==>    the
6  camino              ==>    way
7  .                   ==>    NULL
It remains the same till convergence.

#3. Is it a usable translation system? 
Answer:
Yes, I think it is a usable system. It treats words like word salad but with enough training corpus it can produce satisfactory accuracy.
But it does not consider reordering and adding or dropping words, so it is not a really advanced model.

#4. What kinds of systematic errors does your model make? 
Answer:
The model cannot deal with the order of words and it cannot detect adding or dropping words during translation. I think it is the drawback of the whole IBM model1.
Also it cannot deal with rare words and the translation of rare words is not good.

#5. Are these errors due to (a lack of) data, or are they a result of assumptions built into the model?
Answer:
Both. I think with more data the results would be better, but this model has its own drawback regardless of the training corpus.

#6. If the latter, what changes would you make to the model to improve its performance?
Answer:
I think the model thinks of word translation too simply. It is only using bag of words. So as an improvement it should also take the order between words into account and also think about adding or dropping words during translation.