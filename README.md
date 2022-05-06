# KNN
+ Dataset :
    Iris flower dataset
+ evaluation meyhod
    use accuracy_score to evulate the result
+ We have 2 example in there
1. Use K=1 
2. UIncrease K
because the data closest to that data we want to predict can be a noise, we can use more closest data and point out which my data belong to based on the major one. This is called by "Major voting".
+ One thing can improve my process that is assign weight to the closing points. The closer point have bigger weight.  
To be easy we choose the inversion of the distance as the weight of this data point . 

*** Excersice use KNN to classify MNIST dataset
