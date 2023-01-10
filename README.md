# Rossman-store-sales
Building a xgboost regression model to predict the sales of forthcoming dates. information fed to the model are like store status, promotion, holiday etc. Dataset can be found [here](https://www.kaggle.com/c/rossmann-store-sales/data).

## Exploratory Data Analysis
Studying the distribution of values in each column, and their relationship with the target column Sales.</br>
let's see the Disctribution for column `Sales` and `Customers`



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

OneHotEncoding simply means there are a lot of columns but there is only one column which is "Hot" or On or 1.

![convert notebook to web app](https://miro.medium.com/max/1400/1*ggtP4a5YaRx6l09KQaYOnw.png)

### 


## Gradient Boosting Training
