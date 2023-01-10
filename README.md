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
    
   `Month`: 7 ( which can be also coverted into catagorical column "July"
    
   `Day`: 31
    
    `WeekOfYear`: 31
> Now this information makes more sense and can be fed into the model easily
## Gradient Boosting Training
