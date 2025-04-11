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

## Modeling

I followed the instruction to create a baseline model with DummyClassifier(), and a simple model with LogisticRegression().

The DummyClassifier() falls back to the most frequent class, i.e. "no", campaign not successful, with an accuracy of 0.7352.

The simple LogisticRegression has an accuracy of 0.8306

I then created SVM, DecisionTree and KNN model with default setting.

The accuracy result for these classifiers is:


 Model | Train Time | Train Accuracy | Test Accuracy
--- | --- | --- | --- 
LogisticRegression |  0.4923    |      0.825738   |     0.830610 
SVM        |          0.1498    |      0.830897    |    0.848667 
DecisionTree  |  0.0238 | 1.000000   |     0.783319
KNN          |        0.0087     |     0.847521    |    0.798796 

## Improving the models

I tried tuning the hypoerparameters with grid searching. On top of that decided to use a different performance metric which focus more on recall and specificity, also with recall being given higher weigh than specificity. Each FalseNegative sample means we lose the opportunity to contact and convince the customer to participate in the campaign. Slightly lower specificity is okay as long as the resource put in to contact each False Positive customer isn't very high.

LogisticRegression is tuned by grid searching on parameter C/regularization, SVM is tuned by grid searching on gama, DecisionTree is tuned by grid searching on max-depth, KNN is tuned by grid searching on n_neighbors.

The following is the table of test result. LogistcRegression and DecisionTree(max_depth == 6) are both quite good.

model      |  accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression | 0.843508 | 0.712838 | 0.685065 | 0.900585
SVM         |        0.857266 | 0.762963 |  0.668831 | 0.925146
DecisionTree-6  |    0.852966 | 0.735395 |  0.694805 | 0.909942
KNN-7        |       0.803095 | 0.665272 |  0.516234 | 0.906433


## Custom threshold for prediction
I used a custom threshold of 0.3 (instead of the default 0.5) to improve recall because each False Negative would likely cost us a campaign success

DecisionTree (mat-depth == 6) ends up to be the best model with this approach, with recall of 0.8344 and specificity of 0.8234, LogisticRegresion is not far behind


   model | accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression | 0.818573 | 0.623410 |  0.795455 | 0.826901
SVM           |      0.845228 | 0.703822 |  0.717532 | 0.891228
DecisionTree-6   |   0.826311 | 0.629902 |  0.834416| 0.823392
KNN-15        |      0.795357 | 0.598870 |  0.688312 | 0.833918

## Another thought, simply the model by removing some features

Try removing features month, day_of_week, job, education and martial
These categorical features end up with more than 30 features after OneHot encoding, removing them leaves us with a much smaller table, here I also use a custom threshold of 0.3 to improve recall.


model |  accuracy | precision | recall  |  specificity
--- | --- | --- | --- | ---
LogisticRegression | 0.809974 | 0.609023 |  0.788961 | 0.817544
SVM                | 0.845228 | 0.703822 |  0.717532 | 0.891228
DecisionTree-6    |  0.815993 | 0.618090 | 0.798701 | 0.822222
KNN-15          |    0.784179 | 0.566434 | 0.788961 | 0.782456


## Next Steps

1. Explore further on feature engineering, can try polynomial features on columns with numerical values, while are age, duration, campaign, pdays and previous. However the best result may not come from typical polynomial processing with degree of 2 or 3, degrees like 1/2 (square root) 1/3 or 3/2 may work better. Or maybe logarithm of some features, the possibility is endless. This can be quite time consuming given there don't seem to be library for degree of 1/2, 1/3 or 3/2.
2. grid search on more parameters with finer granularity of each model

The jupyter file is:
https://github.com/sgirem463/ML-AI-Comparing-Classifiers/blob/main/prompt_III.ipynb
