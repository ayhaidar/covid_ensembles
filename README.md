# COVID study notebooks and results

This repo details a feature extraction study used to detect COVID in CT image slices. The dataset was collected from an online github repo: https://github.com/UCSD-AI4H/COVID-CT (28/04/2020).

The details of the study are found in the folder named __extraction__. 

The script used to develop the transfer learning model is saved in this directory: __extraction/model/MondayDenseNet169/__

The scripts utilized to develop the ensembles of decision trees are saved in __extraction__.

The results are also saved in .csv files in __extraction__ .

In this study, the deep learning transfer learning model was trained to extract deep features, then ensembles were trained using the extracted deep features. 

Six types of ensembles of decision trees were incorporated: 

- Extreme Gradient Boosting (XGBoost)
- Gradient Boosting Decision Trees (GBDT)
- Bagged Decision Trees (BDT)
- Adaptive Boosting Decision Trees (Adaboost)
- Dropouts meet multiple Additive Regression Trees (DART)
- Random Forest (RF)

The study targeted investigating the effect of using ensembles for detecting covid-19 in CT images. 

