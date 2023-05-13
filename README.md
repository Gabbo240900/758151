# Asteorid Classification 
## Will it impact the Earth ?

Team Members: Gabriele Di Palma, Francesco Di Stefano, Francesco Tramontano 

### Introduction
In this project the team had to face a classification task. We were given a dataset containing different observations of NEOs ( Near-Earth Objects ), detected by NEOWISE mission. 
Te project aims at analyzing the dataset to gain insigths on the differnt asteorids near the earth. This analysis was used to identify different patterns that could be used to detect potentially hazardouse asteroids impacts. 

### Methods

### Experimental Design

### Results
In this section, we are going to analyze the results of our research.
The logistic regression, as expected, performs amazingly well on the balanced dataset. For this reason, we chose this model as the high benchmark (the model to beat to have a satisfactory model).

![download](https://github.com/Gabbo240900/758151/assets/127876439/8880a32f-ce82-4817-b8d2-9cd74a6b98e7)


<ol>
<li>*Logistic regression precision (balanced): 0.9517241379310345*</li>
<li>*Logistic regression specificity (balanced): 0.9565217391304348*</li>
<li>*Logistic regression accuracy (balanced): 0.9668874172185431*</li>
<li>*Logistic regression recall (balanced): 0.9787234042553191*</li>
<li>*Logistic regression F-1 score (balanced): 0.965034965034965*</li>
</ol>
On the other hand, the same algorithm generates very different results on the real dataset consequently, we chose this model, performed on unbalanced data, as the low benchmark (the model that must necessarily be overcome in order to have an acceptable model).
 
Logistic regression precision (unbalanced): 0.2363013698630137
Logistic regression specificity (unbalanced): 0.8663870581186339
Logistic regression accuracy (unbalanced): 0.870939925265881
Logistic regression recall (unbalanced): 0.9787234042553191
Logistic regression F-1 score (unbalanced): 0.3806896551724137








The Random forest, as expected, performed extremely well with a really a few missclassificated objects
 

Random forest accuracy: 0.9946695095948828
Random forest precision: 0.9869281045751634
Random forest recall: 0.9805194805194806
Random forest F-1 score: 0.9837133550488599
Random forest specificity: 0.9974489795918368

The A.N.N. also performed well even if it is less accurate then the previous model
 
Ann accuracy: 0.9872068230277186
Ann precision: 0.9671052631578947
Ann recall: 0.9545454545454546
Ann F-1 score: 0.9607843137254902
Ann specificity: 0.9936224489795918+9

### Conclusions


