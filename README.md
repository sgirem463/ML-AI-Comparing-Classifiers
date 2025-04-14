# ML-AI-Comparing-Classifiers

## Overview
The goal of this project is to compare the performance of the classifiers K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

The dataset comes from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. 


## Business Understanding
The business objective is to come up with a machine learning model that has a good performance in predicting the campaign results. The exact criteria of model performance will come from a balance of accuracy, recall, specificity...etc by examining the confusion matrix of the model.

### Data Understanding
I went over the data and did the following:

    - checked the geberal info and look at samples of the data
    - checked if there are missing or unknown values in the data
    - used the pandas value_counts(), min() and max() functions to examine values of each column


## Data Preparation

I did the following:

    - removed non-bank related features, e.g. emp.var.rate, cons.price.idx
    - removed entries with missing/unknown values
    - OneHot Encoding for categorical features
    - Scale the numerical features

## Modeling

I followed the instruction to create a baseline model with DummyClassifier(), and a simple model with LogisticRegression().

The DummyClassifier() falls back to the most frequent class, i.e. "no", campaign not successful, with an accuracy of 0.7352.

The simple LogisticRegression has an accuracy of 0.8306

I then created SVM, DecisionTree and KNN model with default setting.

The accuracy result for these classifiers is:


 Model | Train Time | Train Accuracy | Test Accuracy
--- | --- | --- | --- 
LogisticRegression |	0.0782 |	0.825165 |	0.830610
SVM	| 0.1376 |	0.850387 |	0.845228
DecisionTree |	0.0235 |	1.000000 |	0.777300
KNN |	0.0057 |	0.858985 |	0.823732


## Improving the models

I tried tuning the hypoerparameters with grid searching. On top of that decided to use a different performance metric which focus more on recall and specificity, also with recall being given higher weigh than specificity. Each FalseNegative sample means we lose the opportunity to contact and convince the customer to participate in the campaign. Slightly lower specificity is okay as long as the resource put in to contact each False Positive customer isn't very high.

Also add PolynomialFeatures with degree=2 to transform numerical features

LogisticRegression is tuned by grid searching on parameter C/regularization, SVM is tuned by grid searching on gama, DecisionTree is tuned by grid searching on max-depth, KNN is tuned by grid searching on n_neighbors.

TThey are all reasonable good, DecisionTree(max_depth == 4) has the best recall.

model      |  accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression |	0.836629 |	0.702055 |	0.665584 |	0.898246
SVM |	0.840929 |	0.722022 |	0.649351 |	0.909942
DecisionTree-4 |	0.849527 |	0.719472 |	0.707792 |	0.900585
KNN-21 |	0.837489 |	0.700337 |	0.675325 |	0.895906


## Custom threshold for prediction
I used a custom threshold of 0.3 (instead of the default 0.5) to improve recall because each False Negative would likely cost us a campaign success

LogisticRegresion is quite good for recall and specificity


   model | accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression |	0.819433 |	0.618932 |	0.827922 |	0.816374
SVM |	0.843508 |	0.686391 |	0.753247 |	0.876023
DecisionTree-4 |	0.785899 |	0.565410 |	0.827922 |	0.770760
KNN-21 |	0.805675 |	0.599034 |	0.805195 |	0.805848

## Another thought, simply the model by removing some features

Try removing features month, day_of_week, job, education and martial
These categorical features end up with more than 30 features after OneHot encoding, removing them leaves us with a much smaller table, here I also use a custom threshold of 0.3 to improve recall.

Overall seems slightly worse than models that keep all features.
DecisionTree stands out in recall, LogisticRegression and KNN have good balance between recall and specificity

model |  accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression |	0.795357 |	0.581395 |	0.811688 |	0.789474
SVM |	0.841788 |	0.694969 |	0.717532 |	0.886550
DecisionTree-4 |	0.767842 |	0.539095 |	0.850649 |	0.738012
KNN-46 |	0.794497 |	0.579310 |	0.818182 |	0.785965



## Next Steps

1. Explore further on feature engineering, can try more polynomial feature combinations on columns with numerical values, while are age, duration, campaign, pdays and previous. However the best result may not come from typical polynomial processing with degree of 2 or 3, degrees like 1/2 (square root) 1/3 or 3/2 may work better. Or maybe logarithm of some features, the possibility is endless. This can be quite time consuming given there don't seem to be library for degree of 1/2, 1/3 or 3/2.
2. grid search on more parameters with finer granularity of each model

The jupyter file is:
https://github.com/sgirem463/ML-AI-Comparing-Classifiers/blob/main/prompt_III.ipynb
