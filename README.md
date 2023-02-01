# About The Project
Building a xgboost regression model to predict the sales of forthcoming dates. information fed to the model are like store status, promotion, holiday etc. Dataset can be found [here](https://www.kaggle.com/c/rossmann-store-sales/data).

## Exploratory Data Analysis
Studying the distribution of values in each column, and their relationship with the target column Sales.</br>
let's see the Disctribution for column `Sales` and `Customers`

![distribution](https://github.com/rishabh422tiwari/Rossman-store-sales/blob/main/Images/Distribution.png?raw=true)

## Preprocessing and Feature Engineering
Data preprocessing in Machine Learning refers to the technique of preparing (cleaning and organizing) the raw data to make it suitable for a building and training Machine Learning models.

Feature engineering refers to manipulation — addition, deletion, combination, mutation — of your data set to improve machine learning model training, leading to better performance and greater accuracy
### Date column
First, let's convert Date to a date column and extract different parts of the date.

For Example :

If date is 2015-07-31 then the extracted infromation will be these additional columns.
    
   `Year`: 2015
   
   `Month`: 7 (which can be also coverted into catagorical column "July")
   
   `Day`: 31
   
   `WeekOfYear`: 31
   
Now this information makes more sense and can be fed into the model easily

### Store Open/Closed

the sales are ZERO whenever the store is closed

Instead of trying to model this relationship, it would be better to hard-code it in our predictions, and remove the rows where the store is closed. We won't remove any rows from the test set, since we need to make predictions for every row.

### Competition

Next, we can use the columns CompetitionOpenSince [Month/Year] columns from store df to compute the number of months for which a competitor has been open near the store.

`CompetitionOpenSinceYear` and `CompetitionOpenSinceYear` will be converted into another column `CompetitionOpen`

### Input and Target Columns¶

Saving all the inputs and output column

### Impute missing numerical data

`CompetitionDistance` had 2186 nan value so we have to fill those values with appropriate values.

we can simply fill it with the highest value (to indicate that competition is very far away) `max` finction can be used here

### Scale Numeric Valus

I used `MinMaxScaler` from `sklearn.preprocessing` to scale the numerical values.

### Encode Categorical Columns

We scaled the numerical values but there something has to be done, see machine only understand numerical values but we have information in the form of catogories so lets convert the Catagorical values to Numerical with `OneHotEncoder`. 

OneHotEncoding simply means there are a lot of columns but there is only one column which is "Hot" or On or 1 which represents the catagory.

![OneHotEncoding](https://miro.medium.com/max/1400/1*ggtP4a5YaRx6l09KQaYOnw.png)

---
 
## Gradient Boosting Training

We're now ready to train our gradient boosting machine (GBM) model. Here's how a GBM model works:

1. The average value of the target column and uses as an initial prediction every input.
2. The residuals (difference) of the predictions with the targets are computed.
3. A decision tree of limited depth is trained to **predict just the residuals** for each input.
4. Predictions from the decision tree are scaled using a parameter called the learning rate (this prevents overfitting)
5. Scaled predictions for the tree are added to the previous predictions to obtain the new and improved predictions.
6. Steps 2 to 5 are repeated to create new decision trees, each of which is trained to predict just the residuals from the previous prediction.

The term "gradient" refers to the fact that each decision tree is trained with the purpose of reducing the loss from the previous iteration (similar to gradient descent). The term "boosting" refers the general technique of training new models to improve the results of an existing model. 

Here's a visual representation of gradient boosting:

![](https://miro.medium.com/max/560/1*85QHtH-49U7ozPpmA5cAaw.png)

>**Prediction = Actual Value + Learning Rate * Decision tree (1) + Learning Rate * Decision tree(2) + .......**

### Training

To train a GBM, we can use the `XGBRegressor` class from the [`XGBoost`](https://xgboost.readthedocs.io/en/latest/) library.

    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)

> n_ jobs -> to configure the number of thread that it should use in the background (-1 means all threads available in machine) <br>
> n_estimators -> number of decision trees that we will create<br>

If we train an unbounded decision tree to predict the residuals then it will completely overfit the data therefore we are setting the max_depth = 4

Let's train the model using `model.fit`.

    model.fit(X,targets)

### Prediction

We can now make predictions and evaluate the model using `model.predict`.

    preds = model.predict(X)

### Evaluation

Let's evaluate the predictions using RMSE error.

    from sklearn.metrics import mean_squared_error

    def rmse(a,b):
        return mean_squared_error(a,b,squared=False)

At first and very basic model i found the RMSE is 2377 which is not very bad considering we have not fine tuned model.

### Visualization

We can visualize individual trees using `plot_tree` (note: this requires the `graphviz` library to be installed).

![Tree](https://github.com/rishabh422tiwari/Rossman-store-sales/blob/main/Images/tree.png?raw=true)

> Notice how the trees only compute residuals, and not the actual target value. We can also visualize the tree as text.

### Feature importance

Just like decision trees and random forests, XGBoost also provides a feature importance score for each column in the input.

> `feature_importances_` can be calculated in many ways, one of the way is called **information gain** which has to with how much each feature has contributed to reduction in loss over all the trees.<br>
> Second is called `weight`, which counts how many times a particular feature was used to create a split.

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
We can also Plot graph for it

---

## K Fold Cross Validation

Notice that we didn't create a validation set before training our XGBoost model. We'll use a different validation strategy this time, called K-fold cross validation :

![](https://vitalflux.com/wp-content/uploads/2020/08/Screenshot-2020-08-15-at-11.13.53-AM.png)

Scikit-learn provides utilities for performing K fold cross validation.

### Hyperparameter Tuning and Regularization

Just like other machine learning models, there are several hyperparameters we can to adjust the capacity of model and reduce overfitting.

<img src="https://i.imgur.com/EJCrSZw.png" width="480">

Now just play with the parameters like `n_estimators`, `max_depth`, `learning_rate` , `booster` etc.

### Putting it Together and Making Predictions 

Let's train a final model on the entire training set with custom hyperparameters. 

    model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=1000, 
                         learning_rate=0.2, max_depth=10, subsample=0.9, 
                         colsample_bytree=0.7)
                    
    model.fit(X, targets)
  
Feature Importance for our final model :

![feature importance](https://github.com/rishabh422tiwari/Rossman-store-sales/blob/main/Images/feature.png?raw=true)

Let's see the training and validation error for final model :

![error graph](https://github.com/rishabh422tiwari/Rossman-store-sales/blob/main/Images/error_graph.png?raw=true)

**After fine tuning error reduced to 638 which is pretty good because if we see the distribution of our Sales column most of the data point falls around 5000 to 8000 and i was able to achieve `R2_score` of 0.95 which can be interpreted as 95%** 

---

## Saving Model

    import pickle
    pickle.dump(model, open("xgb_reg.pkl", "wb"))
    xgb_reg = pickle.load(open("xgb_reg.pkl", "rb"))
    
# What can be done to improve the model : 
- Try Random Forest
- maybe try to turn off some feature
- play with parameters like n_estimators, max_depth or alpha

> Learning from the project : Machine learning is applied science so play around the parameter and do hit and trial as much as you can when training model.
    


