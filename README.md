# Asteorid Classification 
## Will it impact the Earth ?

Team Members: Gabriele Di Palma, Francesco Di Stefano, Francesco Tramontano 

### Introduction
In this project the team had to face a classification task. We were given a dataset containing different observations of NEOs ( Near-Earth Objects ), detected by NEOWISE mission. 
Te project aims at analyzing the dataset to gain insigths on the differnt asteorids near the earth. This analysis was used to identify different patterns that could be used to detect potentially hazardouse asteroids impacts. 

### Methods

![image](https://github.com/Gabbo240900/758151/assets/127833047/0cd16155-6ca9-4c92-8b98-726f32f87e50)

For the other correlation images refer to: https://github.com/Gabbo240900/758151/tree/main/images/Correlations

![FlowChart](https://github.com/Gabbo240900/758151/assets/127876439/bbf4a01b-36fe-4cdc-a580-ebe635af766f)

### Experimental Design

In this section of the report, we will describe our main experiments conducted during the realization of the project.
Starting with the Exploratory data analysis the main experiments regarded the choice of the variables. In this phase, we needed to choose which variables had to be removed from our dataset to conduct a more correct classification. The main idea behind the choice of variable removal resides in the correlation with the others and how they were expressed. We decided after looking at the correlation matrix, and distributions to remove some that were highly correlated with one another or useless since repetitive in the dataset. Others were kept based on further performance metrics of the classification models.
In the Machine Learning phase, there were many experiments conducted mainly on the hyperparameter tuning phase. 
Starting with the logistic regression, the main experiment regarded the choice of parameters for the DBScan algorithm for under-sampling. After many tries and keeping in mind that we wanted a perfectly balanced dataset, we chose the parameters that you can find in the .ipynb. 
Moreover, we chose to use an under-sampling strategy rather than oversampling after talking with the course's Teaching Assistant, following his advice. 
In the Random Forest stage, the main experiments regarded again the choice of the hyperparameters. They are fundamental for the performance of the model thus to choose them we based ourselves on two things. First, we only looked at the metrics (F1, AUC…) and tried to change them manually, then we also run a Grid Search so to have a perfect understanding of which parameters fitted the best with our dataset. 
In the ANN scenario again tuning the hyperparameters (loss-function, number of layers, epochs, optimizer …) was the key aspect of our experiments. We chose them again based mainly on the performance of the model on our dataset. 
After conducting all the ML techniques, we wanted to make our models perform better and thus we tried t remove some variables that were not significant for the classification, basing ourselves on the results from the Logistic Regression and Random Forset. 
We decided at the end to keep all of them mainly for performance reasons since removing them reduced the performance both on training and testing of all the models. 



### Results
In this section, we are going to analyze the results of our research.
The logistic regression, as expected, performs amazingly well on the balanced dataset. For this reason, we chose this model as the high benchmark (the model to beat to have a satisfactory model).

![download](https://github.com/Gabbo240900/758151/assets/127876439/aae01376-8960-45ea-b956-0194b650727b)

<ol>
<li>Logistic regression precision (balanced): 0.9517241379310345 </li>
<li>Logistic regression specificity (balanced): 0.9565217391304348</li>
<li>Logistic regression accuracy (balanced): 0.9668874172185431</li>
<li>Logistic regression recall (balanced): 0.9787234042553191</li>
<li>Logistic regression F-1 score (balanced): 0.965034965034965</li>
</ol>


On the other hand, the same algorithm generates very different results on the real dataset consequently, we chose this model, performed on unbalanced data, as the low benchmark (the model that must necessarily be overcome in order to have an acceptable model).

![download](https://github.com/Gabbo240900/758151/assets/127876439/e602aea3-6094-44b5-849a-14fed4ab95ea)

<ol> 
<li>Logistic regression precision (unbalanced): 0.2363013698630137</li>
<li>Logistic regression specificity (unbalanced): 0.8663870581186339</li>
<li>Logistic regression accuracy (unbalanced): 0.870939925265881</li>
<li>Logistic regression recall (unbalanced): 0.9787234042553191</li>
<li>Logistic regression F-1 score (unbalanced): 0.3806896551724137</li>
</ol>


The Random forest, as expected, performed extremely well with a really a few missclassificated objects
 
![download](https://github.com/Gabbo240900/758151/assets/127876439/aa888b7b-c732-4de5-ba93-5791078dca3c)

<ol> 
<li>Random forest accuracy: 0.9946695095948828</li>
<li>Random forest precision: 0.9869281045751634</li>
<li>Random forest recall: 0.9805194805194806</li>
<li>Random forest F-1 score: 0.9837133550488599</li>
<li>Random forest specificity: 0.9974489795918368</li>
</ol> 


The A.N.N. also performed well even if it is less accurate then the previous model

![download](https://github.com/Gabbo240900/758151/assets/127876439/e5bf84e5-1b42-4629-842e-b38a9efc4876)

<ol> 
<li>Ann accuracy: 0.9872068230277186</li>
<li>Ann precision: 0.9671052631578947</li>
<li>Ann recall: 0.9545454545454546</li>
<li>Ann F-1 score: 0.9607843137254902</li>
<li>Ann specificity: 0.9936224489795918+9</li>
</ol>


### Conclusions

In conclusion, as we can see from the graph below, Logistic Regression is the worst algorithm that can be applied in a real dataset.

![download](https://github.com/Gabbo240900/758151/assets/127876439/24d22e10-140e-412f-ac84-9f592998cad9)

for the other three algorithms the figure is not clear and for this reason, the graph below, a zoom of the previous one, aims to better represent the differences between the models

![download](https://github.com/Gabbo240900/758151/assets/127876439/41871c6a-bb71-43cd-bc16-3225065c182b)

As we can see both A.N.N. and Random Forest are very close to our high benchmark, but only the latter exceeds it in all the considered metrics. This is one of the reasons for considering the random forest the best of the models. In addition to having performed better than the A.N.N., the Random Forest is a simpler and faster model; therefore in our opinion it would have been the better model of the two for equal results. As far as logistic regression is concerned, we have been able to see how clearly it cannot be considered a good model in an unbalanced dataset.








