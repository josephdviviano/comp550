That/C that/N is/V, is/V .
That/C that/N is/V not/N, is/V not/N .
Is/V that/N it/N ?
Is/V it/N that/N good/J ?

lexicon:
+ that
+ is
+ not
+ it
+ good

tagset:
+ C
+ N
+ V
+ J

parameters:
+ Î  initial state probabilities)
+ A (state transition probabilities)
+ B (emission probabilities).

-- run viterbi algo to get POS for "That is good ."

new sentences:
+ "Bad is not good ."
+ "Is it bad ?"

+ use EM to update previously trained model to account for "bad".
+ show calculations for one iteration of EM.
+ solution should assign non-zero probability to unseen sentences
  involving the word "bad" and the other 5 lexical items, such as the sentence
  "That is bad ."


