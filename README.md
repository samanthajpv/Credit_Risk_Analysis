# Credit_Risk_Analysis
Machine Learning Classification Models to Predict Credit Risk

## Project Overview 
The purpose of this project was to evaluate the performance of selected supervised machine learning models used for classification in predicting credit risk. A total of six models were used on an unbalanced dataset with focus on over- and undersampling, and ensemble algorithms. The datasest used was from LendingClub dated Q1 of 2019.

### Resources
- Data Source: [LoanStats_2019Q1.csv](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/LoanStats_2019Q1.csv)
- Language: Python 3.7.10
    - Libraries: scikit-learn, imbalanced-learn, collections, pathlib, numpy, pandas
- Software: Jupyter Notebook
- Code:
    - [credit_risk_resampling.ipynb](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/credit_risk_resampling.ipynb)
    - [credit_risk_ensemble.ipynb](https://github.com/samanthajpv/Credit_Risk_Analysis/blob/d8ce71daad78eaf3aa5305cefcf2eac6c8ee3a2d/credit_risk_ensemble.ipynb)

## Results
For the purpose of this project and to ensure consistency between tests, random state of 1 was used for each sampling algorithm.

**A. Naive Random Oversampling**
 - ```RandomOverSampler```: oversampling the minority class through random sampling with replacement ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Random%20Naive%20Oversampling.png" width="500" height="300"/>
</p>

- Balanced Accuracy Score: 0.67
- Precision Score: high_risk = 0.01 | low_risk = 1.00
- Recall Score: high_risk = 0.74 | low_risk = 0.61

**B. SMOTE**
- ```SMOTE```: synthetic minority oversampling technique, like random oversampling but new instances are interpolated ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/SMOTE.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score: 0.66
- Precision Score: high_risk = 0.01 | low_risk = 1.00
- Recall Score: high_risk = 0.63 | low_risk = 0.69

**C. Cluster Centroids**
- ```ClusterCentroids```: undersamples the majority class through replacing a cluster of samples by a centroid of a KMeans algorithm ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Cluster%20Centroids.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score: 0.54
- Precision Score: high_risk = 0.01 | low_risk = 1.00
- Recall Score: high_risk = 0.69 | low_risk = 0.40

**D. SMOTEENN**
- ```SMOTEENN```: combination of over and undersampling ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/SMOTEENN.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score: 0.66
- Precision Score: high_risk = 0.01 | low_risk = 1.00
- Recall Score: high_risk = 0.73 | low_risk = 0.59

**E. Balanced Random Forest Classifier**
- ```BalancedRandomForestClassifier```: randomly under-samples each boostrap sample ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Balanced%20Random%20Forest%20Classifier.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score: 0.79
- Precision Score: high_risk = 0.03 | low_risk = 1.00
- Recall Score: high_risk = 0.70 | low_risk = 0.87

**F. Easy Ensemble AdaBoost Classifier**
- ```EasyEnsembleClassifier```: undersamples majority class by training on different balanced bootstrap samples through bagging Adaboost learners ([see more](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html))
<p align="middle">
    <img src="https://github.com/samanthajpv/Credit_Risk_Analysis/blob/8bedee6cb71551ae3e531ef2d1acd9ecfccce801/Images/Easy%20Ensemble%20Classifier.png" width="500" height="275"/>
</p>

- Balanced Accuracy Score: 0.93
- Precision Score: high_risk = 0.09 | low_risk = 1.00
- Recall Score: high_risk = 0.92 | low_risk = 0.94

## Summary
**Scores:**
- The **Balanced Accuracy Score** of models A through D are roughly in the same range from 0.54 to 0.67 with Cluster Centroids having the lowest score. An increase was seen in the Balanced Random Forest Classifier while the highest score was seen in the Easy Ensemble Classifier with 0.93, the best model for this metric.
- All models have a **precision** of 1.00 for the low_risk applications. This means that there are no false positives for low_risk. For models A through D, a bad application will be detected correctly 1% of the time. It increases to 3% and then 9% for the ensemble models E and F respectively. 
- Out of the 6 models, the best **recall** scores are seen in the Easy Ensemble Classifier. This model is more likely to produce less false negatives for both high and low risk loan applications. 

**Recommendation:**

Precision scores across the models are approximately the same but balanced accuracy scores vary. The recommended model to use is the Easy Ensemble Classifier. Aside from having the highest average scores, for loan prediction risk analysis, the recall score outweighs precision since false negatives are coupled with high cost. It is also recommended to scale the features before training the model and further examine which features to include in order to improve the model's performance.

## Reference
(1) Trilogy Education Services. (2021, October). *Module 17 Challenge*. https://courses.bootcampspot.com/courses/626/assignments/13350?module_item_id=213727

(2) *RandomOverSampler*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html

(3) *SMOTE*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

(4) *ClusterCentroids*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html

(5) *SMOTEENN*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html

(6) *BalancedRandomForestClassifier*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

(7) *EasyEnsembleClassifier*. (n.d.). Imbalanced-Learn. Retrieved October 7, 2021, from https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html
