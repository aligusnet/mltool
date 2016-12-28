## Multiclass Classification Sample App:

## Optical Recognition of Handwritten Digits Data Set.

Data source: https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

Data preprocessing:

    sed -i .bak 's/,/ /g' optdigits.tra 
    sed -i .bak 's/,/ /g' optdigits.tes

Learning with lambda = 30 and maximum number of iterations = 30 takes 2 minutes on Core i7 with 4 cores
and gives the following results:

Accuracy on train set (%): 99.4
Accuracy on test set (%): 96.7
