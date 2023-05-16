# Asteroid Classification 
## Will it impact the Earth ?

Team Members: Gabriele Di Palma, Francesco Di Stefano, Francesco Tramontano 

### Introduction
In this project the team had to face a classification task. We were given a dataset containing different observations of NEOs ( Near-Earth Objects ), detected by NEOWISE mission. 
The project aims at analyzing the dataset to gain insigths on the differnt asteorids near the earth. This analysis was used to identify different patterns that could be used to detect potentially hazardouse asteroids impacts. 

### Methods

<img width="2621" alt="image" src="https://github.com/Gabbo240900/758151/assets/127876439/ed371e62-5fc7-4633-adda-9d72dd9c7ba2">


All of our work has been done on Google Colab, to avoid any problem with the environment such as conflicting libraries. In order to achieve our classification task we took into account three models, the first (Logistic Regression) serving as a benchmark for the other two(Random Forest and A.N.N.). 
We started by importing all the needed libreries for the whole project.  Then we looked for NAs and duplicates, the dataset presented 0 repeted rows and 0 null values; it must be said that in the dataset there are some NEOs that have benn recorded more than once,  this is due to a posterior observation of the object, so this is not a bias augmenting factor. Lastly in the EDA section we dealed with variables, we deleted all the columns containing dates,since they are not useful to our task, and all those that contained the imperial unit of measures. We converted all the variables related to distances to AU (astronomical units, the mean distance from Earth to Sun).  We decided to use this unit of measure instead of creating our scale for two main reasons: the first is that AU is a world recognised unit of measure so that everyone can understand it instantaneously without any further explanation, moreover, we encountered troubles  working with huge numbers on some editors, this is another reason why chose to use Colab as standard for all the members.
The second stage is the exploratory data analysis.


We started by displaing all the coreelations among variables in a table, then we analized the onse that according to us are the most interasting.

![image](https://github.com/Gabbo240900/758151/assets/127833047/325e732a-215a-4502-a84c-7cfffe17f8d4)

1.	Size vs Speed, this correlation was required by the instruction and the really low correlation was predictable since the speed of an object is not influenced by its size in the void.
2.	Distances, we highlighted the correlations between the three distance features, to use only one out of three, since they all express almost the same information
3.	Size vs Brightness,  as we can see there is a not so strong negative correlation, which is unexpected because logically, the bigger an object is, the more it will be able to shine, but this is a trivial assertion, since, as shown by the data, there is no confirmation because there are many factors that determine the brightness of an object in space, for example, the material of the observed object.
4.	Size vs Speed of the object in its orbit, we wanted to investigate this correlation further to corroborate the fact that velocity does not depend on size, which is therefore not a good indicator for mass.
5.	The relationship to Jupiter's orbit vs Orbital period, the more the object has a strong relationship with the orbit of Jupiter the less its orbital period lasts. This could be misleading because we could trivially think that objects with a high relationship with the orbit of Jupiter and a short orbital period are close to the planet, indeed, the orbital period depends on other factors such as for example the speed of the object, which cannot be deduced from these data, and the length of its orbit.
6.	 The relationship to Jupiter's orbit vs Orbital speed, might be the most interesting one since it shows that if an object has a strong relationship with Jupiter, it will be also faster it is in its orbit. This may be due to the effect of gravity assist. Furthermore, this correlation allows us to improve the considerations on the previous one by adding that objects close to Jupiter actually travel their orbit faster
7.	Aphelion Distance (the farthest distance between the NEO and the Sun) vs Semi-Major Axis (the distance from the center of the NEO's orbit to its farthest point), has a strong correlation which is not surprising given that the sun contains 99.86% of the total mass of the solar system. In fact it is normal to believe that the furthest point of the orbit corresponds to the furthest point from the sun, because almost all objects are kept in orbit by the sun which obviously resides within it.
8.	The relationship to Jupiter's orbit vs Semi-Major Axis (the distance from the center of the NEO's orbit to its farthest point), this negative correlation suggests that a strong relationship with Jupiter corresponds to a shorter orbit. Therefore this correlation is the missing piece to make us conclude that all those objects that have a short orbital period, a high orbital speed and a relatively short orbit orbit around Jupiter

For the other correlation images refer to: https://github.com/Gabbo240900/758151/tree/main/images/Correlations

After the correlations we focused on the distributions of the variables and as the plot shows none of them are close to a Gaussian distribution.  As for the outliers, the plots show their presence in some variables, but given that in this specific case our classification task can be associated with an anomaly detection task, it is much more appropriate to keep them. The last exploratory analysis is to evaluate the balance of the data and as the barplot below states the hazardous NEOs are the 16.11% of the total observations.


In order to build the first model, a resampling is required since logistic regression is known to be a bad model for unbalanced datasets.
To balance the data we opted for an undersampling, performed through a clustering. First we ran a DBscan clustering algorithm and then we took at least one observation from each cluster in order to equalize the two values of the target class; the balanced dataset has 1510 observations.
We trained the LR on this dataset and then tested it both on this dataset and on the unbalanced dataset (without the observations used in the train, to avoid bias). In order to identify the most important parameters we ran another LR using a different library.  


For the random forest we used the unbalanced dataset, so that the comparison can be even more effective. Te ran the Random Forest the first time with random parameters which we then changed via a grid search, then we decided to visualize some trees in order to understand the algorithm's decision scheme and finally we focused on determining which features were the most significant.

As for our latest model, the A.N.N., we scaled the same inputs of the Random Forest and then we proceeded with the testing of the parameters, in the code there is the version with the parameters that maximize the performance of the model.
We finally decided to plot the loss function to identify any conceptual errors.

We decided to use our most accurate model to infer about some features that could potentially indicate if a NEO is hazardous.
For this reason we set the Random Forest treshold to 0.4 in order to have more false positives than false negatives because in this case we prefer to capture false positives given the importance of the event.

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

Using the Random forest which is the best model we decided to investigate which NEOs are hazardous and why. 
To do this we compared the test set for NEOs classified as hazardous and not and we compared their distributions using boxplots. 
In such a way we are able to determine which features we should look for to understand if a NEOs is hazardous or not.

*Hazardous objects*
![download](https://github.com/Gabbo240900/758151/assets/127876439/998d2b6d-e1bc-4128-8ccd-6b798daf202f)

*Non hazardous objects*
![download](https://github.com/Gabbo240900/758151/assets/127876439/dce264ca-6c48-4f2e-8be0-7f72a8109886)

There are some features that change their distribution, they are: Absolute magnitude, Orbit uncentainity, Minimum orbit intersection, Eccentricity, Perihelion distance and Mean anomaly and Est. dia in km. 
Considering a new object that has unusual values in all of those variables should be considered possibly hazardous and should be further investigated.
it is also important to specify that many objects enter the earth's atmosphere every year but this does not mean they must be classified as dangerous, many of them do not touch the ground.

### Extra

This section is dedicated to how this project could have been developed with more information. We believe that with more observations and other important variables such as mass and the type of the object, the models would have been more accurate (especially the A.N.N.). 

