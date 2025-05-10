#  Newsletter Conversions

This repository contains a machine learning project developed as part of a predictive modeling challenge inspired by real-world Kaggle-style competitions. The goal is to predict whether a user will subscribe to a newsletter based on their web behavior.

---

##  Project Description

In this ML projet, the goal is to test different predictive models using a dataset provided by [Data Science Weekly](http://www.datascienceweekly.org/) and find the one with the best **F1-score**. The task is to analyze user behavior data and build a classifier that can accurately predict newsletter subscription conversions.

The dataset is split into two parts:

- `data_train.csv`: Labeled data (features + target) used to train and evaluate models.
- `data_test.csv`: Unlabeled data used to make predictions for leaderboard evaluation.

---

##  Tech Stack

- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (modeling and evaluation)
- Model optimization, XGBoost, Voting Classifier
- CSV export for leaderboard submission
---

##  Project Workflow

###  Part 1: EDA & Baseline Model
- Perform exploratory data analysis (EDA)
- Basic preprocessing (handling missing values, encoding)
- Train/test split and evaluate a baseline classifier (univariate) logistic regression

###  Part 2: Model Improvement (Optimize F1 Score)
- Feature engineering & selection
- Perform hyperparameter tuning (GridSearch, Class_weight)
- Try advanced models (e.g., XGBoost, Voting Classifier)
- Model Comparaison

###  Part 3: Submission
- Use the best model to predict labels for `data_test.csv`
- Save predictions for submission
- Submit to JEDHA leaderboard

### Part 4: Model Interpretation
- Analyze feature importance
- Discuss potential business actions to improve conversion rates
- Present insights and recommendations

---

##  Deliverables

- Jupyter notebooks showing:
  - EDA and preprocessing
  - Model training and evaluation (confusion matrix, F1-score)
- At least one submission
- Analysis of model parameters

---

## Evaluation Metric

- **Primary metric:** F1-score
---

## Key Findings:

## F1 Scores
**Baseline : Univariate Model**
- f1-score on test set :  0.7033792240300375
![image](https://github.com/user-attachments/assets/5edbbd7b-f8e7-4424-8cab-4972d04ce7e3)


**Logistic Regression: Feature Engineering**
- f1-score on test set :  0.7676461678279309
![image](https://github.com/user-attachments/assets/1095162e-bdbd-42b2-8202-7d95697979e7)

**Hyperparameter tuning with GridSearch**
- f1-score on test set :  0.7750724637681159
![image](https://github.com/user-attachments/assets/2fbf3b5c-abfd-4974-8301-f10512cc1039)

**XGBoost**
- f1-score on test set :  0.7628681939898839
![image](https://github.com/user-attachments/assets/52eddf18-3bcc-4020-84a6-5919640df92b)

**Voting Classifier**
- f1-score on test set :  0.7716303708063567
![image](https://github.com/user-attachments/assets/01aacacf-82e2-40e6-830e-2e82834d1d76)


**Model perfomance comparaison**
![image](https://github.com/user-attachments/assets/99bf6c8e-6932-4e61-b5ae-abd40d333f3c)


**Model Showdown: Which One Wins?**

1. LogReg GridSearch & Class Weight (98.63% Accuracy) → Best F1-Score (77.69%) & Recall (73.88%), meaning it catches the most positives while keeping balance.

2. LogReg Feature Engineering (98.64% Accuracy) → Highest Precision (85.33%), so it avoids false alarms, but Recall (70.07%) is a bit lower.

3.  VotingClassifier (98.63% Accuracy) → Solid across all metrics, with good Recall (71.48%) and F1-Score (77.16%), but slightly lower Precision (83.82%).

4. XGBoost (98.59% Accuracy) → High Precision (83.96%), but the lowest Recall (69.90%), meaning it plays it safe but may miss some positives.

**Final Verdict: Which one to pick?**

- Need to catch all positives? → GridSearch & Class Weight 🏆
- Hate false positives? → Feature Engineering 🎯
- Want an all-rounder? → VotingClassifier ⚖️
- Prefer conservative predictions? → XGBoost 🤖

🔥 Winner for Most Balanced Performance? → **GridSearch & Class Weight!** 🏆🚀

----
## Feature Importance
![image](https://github.com/user-attachments/assets/7f9339e6-3352-4c27-b223-10c4d308dce2)

**Features:**
- cat_country_China : -4.79 - Strongly decreases conversion likelihood.
- cat_source_Direct : -3.05 - Users coming from Direct Traffic convert much less.
- cat_source_Seo : -2.92 - Ads aren't leading to conversions effectively.
- num_total_pages_visited: 2.54 - More pages visited increases conversion probability.
- cat_country_US : - 1.47 US - US visitors convert less than other countries.
- cat_country_UK : - 1.14 - UK visitors also convert less.
- cat_country_Germany: - 0.90 - German visitors have a slight negative effect on conversion.
- num_new_user : - 0.81 - New users convert less than returning users.
- num_age : - 0.60 - Older users convert slightly less.

----
**Recommendations Based on the Model**

✅ Improve Conversion for China & US Visitors
- Localize content for these regions.
- Consider region-specific offers or promotions.
  
✅ Optimize Ad & SEO Campaigns: Since Ads & SEO users convert less, analyze why:
- Are ads targeting the wrong audience?
- Is SEO traffic not landing on the right pages?
- A/B test new ad messaging or landing pages.
  
✅ Increase Engagement for Direct Traffic Users
- Offer personalized incentives (e.g., discounts or pop-ups for direct visitors).
- Improve navigation so users engage with more pages.
  
✅ Encourage More Page Visits
- The only positive coefficient (total_pages_visited = +2.54) shows that users who explore more convert better.
- Design better content flow to encourage browsing.

✅ Focus on Returning Users
- New users convert less than existing ones → Retarget them!







