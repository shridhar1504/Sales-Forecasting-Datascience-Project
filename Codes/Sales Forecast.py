#!/usr/bin/env python
# coding: utf-8

# # Sales Forecasting Project
# ***

# _**Importing the required libraries & packages**_

# In[1]:


import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from scipy.stats import uniform, randint
import pickle
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\Sales Project')
df = pd.read_csv('Train.csv')


# ## Exploratory Data Analysis (EDA) 

# _**Getting all the unique value counts from all the columns using <span style = 'background : green'> <span style = 'color : white'>  lambda   </span></span> function**_

# In[3]:


df.apply(lambda x : len(x.unique()))


# _**Checking the dataset whether it's having duplicate values or not**_

# In[4]:


df.duplicated().sum()


# _**Checking for the null values of all the columns from the dataset**_

# In[5]:


df.isnull().sum()


# _**Getting the Data types and Non-null count of all the columns from the dataset using <span style = 'background : green'><span style = 'color : white'> .info() </span> </span> statement**_

# In[6]:


df.info()


# ## Data Cleaning

# _**Getting all the columns with "object" data type from the dataset and appending it to the list**_

# In[7]:


cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
display(cat_col)


# _**Removing the columns `Item_Identifier` and `Outlet_Identifier` from the list since the both columns doesn't need any cleaning**_ 

# In[8]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')


# _**Displaying the list after removing certain columns from it to verify**_

# In[9]:


display(cat_col)


# _**Getting the unique value counts of the columns in the list**_

# In[10]:


for col in cat_col:
    print(col,len(df[col].unique()))


# _**Checking the value counts of the columns from the list and displaying it column wise**_

# In[11]:


for col in cat_col:
    print(col)
    print(df[col].value_counts(),'\n')
    print('-'*55)


# _**Getting the null values from the `Item_Weight` column for the null value treatment process and displaying the Dataset with null values in the `Item_Weight` column**_

# In[12]:


miss_bool = df['Item_Weight'].isnull()
Item_Weight_Null = df[df['Item_Weight'].isnull()]
display(Item_Weight_Null)


# _**Identifying the unique value counts in `Item_Identifier` column from the `Item_Weight` null value dataset**_

# In[13]:


Item_Weight_Null['Item_Identifier'].value_counts()


# _**Getting the mean values of the `Item_Weight` with respect to `Item_Identifier` column of the dataset using <span style = 'background : green'><span style = 'color : white'> Pivot Table </span> </span> function**_

# In[14]:


Item_Weight_Mean = df.pivot_table(values = 'Item_Weight', index = 'Item_Identifier')
display(Item_Weight_Mean)


# _**Treating the missing values of the `Item_Weight` column with the mean values we got above using <span style = 'background : green'><span style = 'color : white'> Pivot Table </span> </span> function and filling it out with respect to `Item_Identifier` column**_

# In[15]:


for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in Item_Weight_Mean:
            df['Item_Weight'][i] = Item_Weight_Mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])


# _**After treating the null values in the `Item_Weight` column, checking for the null value in the column**_

# In[16]:


df['Item_Weight'].isna().sum()


# _**Getting the unique value counts from `Outlet_Size` column from the dataset**_

# In[17]:


df['Outlet_Size'].value_counts()


# _**Checking out for the null value counts from the `Outlet_Size` column from the dataset**_

# In[18]:


df['Outlet_Size'].isnull().sum()


# _**Getting the null values from the `Outlet_Size` column for the null value treatment process and displaying the Dataset with null values in the `Outlet_Size` column**_

# In[19]:


Outlet_Size_Null = df[df['Outlet_Size'].isna()]
display(Outlet_Size_Null)


# _**Getting the value counts of `Outlet_Type` from the `Outlet_Size` null dataset**_

# In[20]:


Outlet_Size_Null['Outlet_Type'].value_counts()


# _**Grouping by `Outlet_Type` and `Outlet_Size` with the aggregate function of size of the `Outlet_Type` column values**_

# In[21]:


df.groupby(['Outlet_Type','Outlet_Size']).agg({'Outlet_Type':[np.size]})


# _**Getting the mode values of the `Outlet_Size` with respect to `Outlet_Type` column of the dataset using <span style = 'background : green'><span style = 'color : white'> Pivot Table </span> </span> function**_ 

# In[22]:


Outlet_Size_Mode = df.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x : x.mode()[0]))
display(Outlet_Size_Mode)


# _**Getting the null values of `Outlet_Size` column from the dataset and treating the null value using mode values of the `Outlet_Size` with respect to `Outlet_Type` column**_

# In[23]:


miss_bool = df['Outlet_Size'].isna()
df.loc[miss_bool,'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x : Outlet_Size_Mode[x])


# _**After missing value treatment of the `Outlet_Size` column, checking for the null values in the column**_

# In[24]:


df['Outlet_Size'].isna().sum()


# _**Checking for the null values of all the columns from the dataset after missing value treatment**_

# In[25]:


df.isna().sum()


# _**Getting the count of `Item_Visibility` column with value 0**_

# In[26]:


sum(df['Item_Visibility'] == 0)


# _**Filling out the 0 values from the `Item_Visibility` column with mean values using <span style = 'background : green'><span style = 'color : white'> replace </span> </span> function**_ 

# In[27]:


df.loc[:,'Item_Visibility'].replace([0],[df['Item_Visibility'].mean()],inplace = True)


# _**Now, again checking out for 0 values in the `Item_Visibility` column after filling it out to verify any misplacement happened**_

# In[28]:


sum(df['Item_Visibility'] == 0)


# _**Getting the unique value counts from the `Item_Fat_Content` column**_

# In[29]:


df['Item_Fat_Content'].value_counts()


# _**After seeing the unique value counts from the `Item_Fat_Content` column, there have been some mistyping occured like the same categories were typed under different names. For further processing, all the mistypings are corrected and named under a single category. Checking out for the value counts of `Item_Fat_Content` column**_

# In[30]:


df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF' : 'Low Fat', 'low fat' : 'Low Fat', 'reg' : 'Regular'})
df['Item_Fat_Content'].value_counts()


# _**Adding new column `New_Item_Type` to the dataset by getting the first two characters from the `Item_Identifier` column which represents the category of the item and getting the value counts of the `New_Item_Type` column**_

# In[31]:


df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x : x[:2])
df['New_Item_Type'].value_counts()


# _**As the `New_Item_Type` column has values which is subjected to categories for better understanding, replacing the codes with meaningful categorical item name and getting the value counts of `New_Item_Type` column**_

# In[32]:


df['New_Item_Type'] = df['New_Item_Type'].replace({'FD' : 'Food', 'NC' : 'Non-Consumables', 'DR' : 'Drinks'})
df['New_Item_Type'].value_counts()


# _**Grouping by `New_Item_Type` and `Item_Fat_Content` with the aggregate function of size of the `Outlet_Type` column values**_

# In[33]:


df.groupby(['New_Item_Type','Item_Fat_Content']).agg({'Outlet_Type':[np.size]})


# _**From the above output its clear that Non-Consumable type from `New_Item_Type` column is mapped to Low Fat category in `Item_Fat_Content` column. So marking it as Non-Edible in `Item_Fat_Content` column**_

# In[34]:


df.loc[df['New_Item_Type'] == 'Non-Consumables','Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()


# _**Getting all the unique value from `Outlet_Establishment_Year` column from the dataset**_

# In[35]:


df['Outlet_Establishment_Year'].unique()


# _**The `Outlet_Establishment_Year` column from the dataset has no significance on its own so calculating the years of outlet established until this year and adding it as `Outlet_Years` column to the dataset**_

# In[36]:


curr_time = datetime.datetime.now()
df['Outlet_Years'] = df['Outlet_Establishment_Year'].apply(lambda x: curr_time.year - x)


# ## Data Visualization

# _**Plotting the Bar Graph with count of `Item_Fat_Content`  and confirm that there are no null values and identify all unique values from the `Item_Fat_Content` and saving the PNG File**_

# In[37]:


plt.rcParams['figure.figsize'] = 15,10
plt.style.use('fivethirtyeight')
plot = sns.countplot(x = df['Item_Fat_Content'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Item_Fat_Content')
plt.savefig('Count of Item_Fat_Content.png')
plt.show()


# _**Plotting the Bar Graph with count of `Item_Type`  and confirm that there are no null values and identify all unique values from the `Item_Type` and saving the PNG File**_

# In[38]:


plot = sns.countplot(x = df['Item_Type'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.xticks(rotation = 90)
plt.title('Count of Item_Type')
plt.savefig('Count of Item_Type.png')
plt.show()


# _**Plotting the Bar Graph with count of `Outlet_Establishment_Year`  and confirm that there are no null values and identify all unique values from the `Outlet_Establishment_Year` and saving the PNG File**_

# In[39]:


plot = sns.countplot(x = df['Outlet_Establishment_Year'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Outlet_Establishment_Year')
plt.savefig('Count of Outlet_Establishment_Year.png')
plt.show()


# _**Plotting the Bar Graph with count of `Outlet_Location_Type`  and confirm that there are no null values and identify all unique values from the `Outlet_Location_Type` and saving the PNG File**_

# In[40]:


plot = sns.countplot(x = df['Outlet_Location_Type'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Outlet_Location_Type')
plt.savefig('Count of Outlet_Location_Type.png')
plt.show()


# _**Plotting the Bar Graph with count of `Outlet_Size`  and confirm that there are no null values and identify all unique values from the `Outlet_Size` and saving the PNG File**_

# In[41]:


plot = sns.countplot(x = df['Outlet_Size'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Outlet_Size')
plt.savefig('Count of Outlet_Size.png')
plt.show()


# _**Plotting the Bar Graph with count of `Outlet_Type`  and confirm that there are no null values and identify all unique values from the `Outlet_Type` and saving the PNG File**_

# In[42]:


plot = sns.countplot(x = df['Outlet_Type'])
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Outlet_Type')
plt.savefig('Count of Outlet_Type.png')
plt.show()


# _**Visualizing the data distribution of the `Item_weight` column against the density distribution using Seaborn Distplot and saving the PNG file**_

# In[43]:


sns.distplot(df['Item_Weight'],bins = 20)
plt.title('Distribution of Item_Weight')
plt.savefig('Distribution of Item_Weight.png')
plt.show()


# _**Getting the Correlation Values from all the numeric columns from the dataset using Seaborn Heatmap & saving the PNG File**_

# In[44]:


sns.heatmap(df.corr(),cmap = 'binary', cbar = True, annot = True, square = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# ## Data Preprocessing

# _**Label Encoding the `Outlet_Identifier` column and adding it as a new column `Outlet` to the dataset**_

# In[45]:


le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])


# _**Getting the data types of all the columns to find out the "object" data types columns for preprocessing before assigning it to dependent variable and independent variable**_

# In[46]:


df.dtypes


# _**Adding all the necessary column with "object" data types to the list and Label Encoding the columns**_

# In[47]:


cat_col = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type','New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])


# _**One Hot Encoding the columns `Item_Fat_Content`,`Outlet_Size`,`Outlet_Location_Type`,`Outlet_Type`,`New_Item_Type` using  <span style = 'background : green'><span style = 'color : white'> get dummies </span> </span> function**_ 

# In[48]:


df = pd.get_dummies(df,columns = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type','New_Item_Type'])


# _**Assigning the dependent and independent variable**_

# In[49]:


x = df.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Item_Outlet_Sales'],axis=1)
y=df['Item_Outlet_Sales']


# ## Model Fitting

# _**Splitting the dependent variable & independent variable into training and test dataset using  <span style = 'background : green'><span style = 'color : white'> train test split </span> </span>**_

# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)


# _**Defining the function for the ML algorithm and fitting it with the passing dependent and independent variable, predicting the dependent variable using algorithm with independent variable. With using  <span style = 'background : green'><span style = 'color : white'> cross val score [Cross Validation] </span> </span> process getting the Model Report with cv_score using "neg_mean_squared_error" as scoring and also getting the absolute average mean of cv_score. After that using  <span style = 'background : green'><span style = 'color : white'> cross val score [Cross Validation] </span> </span> process, getting the cv_score as R2 score using default scoring parameter and also again getting the mean value of cv_score with default scoring as Average R2 score. End of the function the Accuracy for full data is actually determined using  actual R2 score founded between the dependent variable and predicted dependent variable. Atlast getting the coefficient of algorithm with all the columns and plotting the graph using coefficient value of algorithm to show impact of the each column**_

# In[51]:


def train(model, x, y):
    model.fit(x, y)
    pred = model.predict(x)
    cv_score = cross_val_score(model,x,y,scoring = 'neg_mean_squared_error', cv = 10)
    print('Model Report : \n')
    print('Scoring - neg_mean_squared_error')
    print(cv_score,'\n')
    cv_score = np.abs(np.mean(cv_score))
    print('Absolute Average of neg_mean_squared_error : ',cv_score)
    cv_score = cross_val_score(model, x, y, cv = 10)
    print()
    print('R2 Score')
    print(cv_score,'\n')
    cv_score = np.mean(cv_score)
    print('Average R2 Score : ',cv_score,'\n')
    print('Accuracy for Full Data :')
    print('R2 Score : ',r2_score(y,pred),'\n')
    coef = pd.Series(model.coef_, x.columns).sort_values()
    print (coef)
    coef.plot(kind='bar', title="Model Coefficients")
    plt.show()


# _**Fitting the <span style ='color:green'> Linear Regression </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[52]:


model = LinearRegression(normalize = True)
train(model,x_train,y_train)


# _**Fitting the <span style ='color:green'> Ridge </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[53]:


model = Ridge(normalize = True)
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> Lasso </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[54]:


model = Lasso(normalize = True)
train(model, x_train, y_train)


# _**Defining the function for the ML algorithm and fitting it with the passing dependent and independent variable, predicting the dependent variable using algorithm with independent variable. With using  <span style = 'background : green'><span style = 'color : white'> cross val score [Cross Validation] </span> </span> process getting the Model Report with cv_score using "neg_mean_squared_error" as scoring and also getting the absolute average mean of cv_score. After that using  <span style = 'background : green'><span style = 'color : white'> cross val score [Cross Validation] </span> </span> process, getting the cv_score as R2 score using default scoring parameter and also again getting the mean value of cv_score with default scoring as Average R2 score. End of the function the Accuracy for full data is actually determined using  actual R2 score founded between the dependent variable and predicted dependent variable. Atlast getting the feature importance of all the columns and plotting the graph using feature importance of algorithm to show impact of the each column**_

# In[55]:


def train(model, x, y):
    model.fit(x, y)
    pred = model.predict(x)
    cv_score = cross_val_score(model,x,y,scoring = 'neg_mean_squared_error', cv = 10)
    print('Model Report : \n')
    print('Scoring - neg_mean_squared_error')
    print(cv_score,'\n')
    cv_score = np.abs(np.mean(cv_score))
    print('Absolute Average of neg_mean_squared_error : ',cv_score)
    cv_score = cross_val_score(model, x, y, cv = 10)
    print()
    print('R2 Score')
    print(cv_score,'\n')
    cv_score = np.mean(cv_score)
    print('Average R2 Score : ',cv_score,'\n')
    print('Accuracy for Full Data :')
    print('R2 Score : ',r2_score(y,pred),'\n')
    coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
    coef.plot(kind='bar', title="Feature Importance")
    plt.show()


# _**Fitting the <span style ='color:green'> Decision Tree Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[56]:


model = DecisionTreeRegressor()
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> Random Forest Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[57]:


model = RandomForestRegressor()
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> Extra Trees Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[58]:


model = ExtraTreesRegressor()
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> LGBM Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[59]:


model = LGBMRegressor()
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> XGB Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[60]:


model = XGBRegressor()
train(model, x_train, y_train)


# _**Fitting the <span style ='color:green'> Cat Boost Regressor </span> algorithm to the model and passing it to the defined function with train dependent and train independent variable and getting the output for the defined function**_

# In[61]:


model = CatBoostRegressor(verbose = 0)
train(model, x_train, y_train)


# _**Passing some of the list of parameters for the <span style ='color:green'> Random Forest Regressor </span> Model to run with Randomized Search CV Algorithm**_

# In[62]:


random_grid = {
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
               'min_samples_split':[2, 5, 10, 15, 100],
               'min_samples_leaf': [1, 2, 5, 10]
}


# _**Fitting The <span style ='color:green'>Random Forest Regressor</span> Model with the above mentioned parameters in the RandomizedSearchCV Algorithm and displaying the Best Parameters, Best Score and R2 Score between test dependent variable and predicted dependent variable**_

# In[63]:


RF = RandomForestRegressor()
RF = RandomizedSearchCV(estimator = RF, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter =10,
                       verbose = 0, cv =10, random_state = 10, n_jobs = 1)
RF.fit(x_train, y_train)
print('Best Params : ',RF.best_params_,'\n')
print('Best Score : ',RF.best_score_,'\n')
prediction = RF.predict(x_test)
print('R2 Score : ',r2_score(y_test,prediction))


# _**Visualizing the data distribution of the dependent test variable , predicted dependent variable of the Random Forest Regressor Model against the density distribution using Seaborn Distplot**_

# In[64]:


sns.distplot(y_test-prediction)
plt.show()


# _**Passing some of the list of parameters for the <span style ='color:green'> LGBM Regressor </span> Model to run with Randomized Search CV Algorithm**_

# In[65]:


params = {
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6),
    "n_estimators": randint(100, 150), 
    "subsample": uniform(0.6, 0.4)
}


# _**Fitting The <span style ='color:green'>LGBM Regressor</span> Model with the above mentioned parameters in the RandomizedSearchCV Algorithm and displaying the Best Parameters, Best Score and R2 Score between test dependent variable and predicted dependent variable**_

# In[66]:


lgb = LGBMRegressor()
lgb = RandomizedSearchCV(estimator = lgb, param_distributions = params, cv = 10, n_iter = 10, verbose = 0,
                        scoring = 'neg_mean_squared_error', n_jobs = 1, random_state = 10)
lgb.fit(x_train,y_train)
print('Best Params : ',lgb.best_params_,'\n')
print('Best Score : ',lgb.best_score_,'\n')
prediction = lgb.predict(x_test)
print('R2 Score : ',r2_score(y_test,prediction))


# _**Visualizing the data distribution of the dependent test variable , predicted dependent variable of the LGBM Regressor Model against the density distribution using Seaborn Distplot**_

# In[67]:


sns.distplot(y_test-prediction)
plt.show()


# _**Passing some of the list of parameters for the <span style ='color:green'> XGB Regressor </span> Model to run with Randomized Search CV Algorithm**_

# In[68]:


params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6),
    "n_estimators": randint(100, 150), 
    "subsample": uniform(0.6, 0.4)
}


# _**Fitting The <span style ='color:green'>XGB Regressor</span> Model with the above mentioned parameters in the RandomizedSearchCV Algorithm and displaying the Best Parameters, Best Score and R2 Score between test dependent variable and predicted dependent variable**_

# In[69]:


xgb = XGBRegressor()
xgb = RandomizedSearchCV(estimator = xgb, param_distributions = params, cv = 10, n_iter = 10, verbose = 0,
                        scoring = 'neg_mean_squared_error', n_jobs = 1, random_state = 10)
xgb.fit(x_train, y_train)
print('Best Params : ',xgb.best_params_,'\n')
print('Best Score : ',xgb.best_score_,'\n')
prediction = xgb.predict(x_test)
print('R2 Score : ',r2_score(y_test,prediction))


# _**Visualizing the data distribution of the dependent test variable , predicted dependent variable of the XGB Regressor Model against the density distribution using Seaborn Distplot**_

# In[70]:


sns.distplot(y_test-prediction)
plt.show()


# _**Passing some of the list of parameters for the <span style ='color:green'> CatBoost Regressor </span> Model to run with Randomized Search CV Algorithm**_

# In[71]:


params = {
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6),
    "n_estimators": randint(100, 150), 
    "subsample": uniform(0.6, 0.4)
}


# _**Fitting The <span style ='color:green'>CatBoost Regressor</span> Model with the above mentioned parameters in the RandomizedSearchCV Algorithm and displaying the Best Parameters, Best Score and R2 Score between test dependent variable and predicted dependent variable**_

# In[72]:


cat = CatBoostRegressor(verbose = 0)
cat = RandomizedSearchCV(estimator = cat, param_distributions = params, cv = 10, n_iter = 10, verbose = 0,
                        scoring = 'neg_mean_squared_error', n_jobs = 1, random_state = 10)
cat.fit(x_train,y_train)
print('Best Params : ',cat.best_params_,'\n')
print('Best Score : ',cat.best_score_,'\n')
prediction = cat.predict(x_test)
print('R2 Score : ',r2_score(y_test,prediction))


# _**Visualizing the data distribution of the dependent test variable , predicted dependent variable of the CatBoost Regressor Model against the density distribution using Seaborn Distplot**_

# In[73]:


sns.distplot(y_test-prediction)
plt.show()


# _**Fitting The <span style ='color:green'>CatBoost Regressor</span> Model with the best params got from the Randomized SearchCV and predicting the test dependent data to verify the r2 score with the r2 score got from Randomized SearchCV**_

# In[74]:


cat = CatBoostRegressor(learning_rate = 0.08941885942788719, max_depth = 2, n_estimators = 109, 
                        subsample = 0.6676443346250142, verbose = 0)
cat.fit(x_train,y_train)
predictions = cat.predict(x_test)
print('R2 score : ',r2_score(y_test,predictions))


# ## Model Testing

# _**Create the pickle file of the model with the highest r2 score with the name Model**_

# In[75]:


pickle.dump(cat,open('Model.pkl','wb'))


# _**Loading the pickle file and predicting the dependent variable for the whole data and getting the r2 score between the predicted dependent variable and dependent variable**_

# In[76]:


model = pickle.load(open('Model.pkl','rb'))
fpred = model.predict(x)
print('R2 Score of Full Data : ',r2_score(y,fpred))

