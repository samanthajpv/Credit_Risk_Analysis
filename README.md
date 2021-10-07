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

**A. Naive Random Oversampling**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Random%20Naive%20Oversampling.png" width="500" height="300"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:

**B. SMOTE Oversampling**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/SMOTE.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:

**C. Cluster Centroids (Undersampling)**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Cluster%20Centroids.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:

**D. SMOTEENN (Combination of Over and Undersampling)**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/SMOTEENN.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:

**E. Balanced Random Forest Classifier**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Balanced%20Random%20Forest%20Classifier.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Importances.png" width="600" height="225"/>
</p>

- Importances:

**F. Easy Ensemble AdaBoost Classifier**
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Easy%20Ensemble%20Classifier.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score:
- Precision Score:
- Recall Score:

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If none, justify reasoning.

## Reference
(1) Trilogy Education Services. (2021, October). *Module 17 Challenge*.https://courses.bootcampspot.com/courses/626/assignments/13350?module_item_id=213727
