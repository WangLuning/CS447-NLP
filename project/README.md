Sentimental Analysis of Movie Review

Source data: 
http://www.cs.cornell.edu/people/pabo/movie-review-data/
Data distribution:
Total:  160867  Min:  1  Max:  62
(array([15142,  1484,   143,    18]), array([    1,    10,   100,  1000, 10000]))
. 9827.0
, 7024.0
the 7006.0
a 5108.0
and 4311.0
of 4208.0
to 3013.0
's 2521.0
is 2497.0
it 2422.0

http://ai.stanford.edu/~amaas/data/sentiment/
Data distribution:
Total:  6685964  Min:  11  Max:  2738
(array([85168, 16279,  3662,   511]), array([    1,    10,   100,  1000, 10000]))
the 334755.0
, 275887.0
. 273838.0
and 163334.0
a 162179.0
of 145429.0
to 135199.0
is 110398.0
it 95772.0
in 93250.0

Small dataset:
SVM, Naive Bayes: ~0.50
RNN accuracy: 0.7434
Naive approach (reference group) accuracy: 0.9009375

Large dataset:
SVM, Naive Bayes: ~0.50
RNN accuracy: 0.8765
Naive approach (reference group) accuracy: 0.71384