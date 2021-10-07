# Credit_Risk_Analysis
Machine Learning Classification Models to Predict Credit Risk

## Project Overview 
The purpose of this project was to evaluate the performance of selected supervised machine learning models used for classification in predicting credit risk. A total of six models will be used with focus on over- and undersampling, and ensemble algorithms. The datasest used was from LendingClub dated Q1 of 2019.

### Resources
- Data Source: [LoanStats_2019Q1.csv](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/LoanStats_2019Q1.csv)
- Language: Python 3.7.10
    - Libraries: scikit-learn, imbalanced-learn, collections, pathlib, numpy, pandas
- Software: Jupyter Notebook
- Code:
    - [credit_risk_resampling.ipynb](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/credit_risk_resampling.ipynb)
    - [credit_risk_ensemble.ipynb](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/credit_risk_ensemble.ipynb)

## Results
For the purpose of this project and to ensure consistency between tests, random state of 1 was used for each sampling algorithm. Models A through D used the resampled data to train a logistic regression model.

A. Naive Random Oversampling
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:

B. SMOTE Oversampling
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:

C. Cluster Centroids (Undersampling)
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:

D. SMOTEENN (Combination of Over and Undersampling)
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:

E. Balanced Random Forest Classifier
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:
    - Importances:

F. Easy Ensemble AdaBoost Classifier
    - Balanced Accuracy Score:
    - Precision Score:
    - Recall Score:

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If none, justify reasoning.

## Reference
(1) Trilogy Education Services. (2021, October). *Module 17 Challenge*.https://courses.bootcampspot.com/courses/626/assignments/13350?module_item_id=213727