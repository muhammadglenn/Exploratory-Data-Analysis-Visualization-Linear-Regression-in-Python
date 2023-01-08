# Import Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='rainbow')

# Dataset
car = pd.read_excel('D:\Datasets\Kaggle\Car Price Prediction Multiple Linear Regression\CarPrice_Assignment.xlsx')
print("Car Dataset")
print(car.head(10))
print("")

# Dataset shape
print("Dataset shape")
print(car.shape)
print("")

# Dataset datatypes
print("Dataset datatypes each column")
print(car.info())
print("")

# Check Duplicates
print("Number of duplicates")
print(car.duplicated().sum())
print("")

# Change categorical variables
car.rename(columns={"symboling": "riskrating"}, inplace=True)
new_riskrating = pd.Categorical(car["riskrating"],
                                 ordered=True)
new_riskrating = new_riskrating.rename_categories(["More Safe","Safe","Normal", "Risky", "More Risky",  "Pretty Risky"])              
car["riskrating"] = new_riskrating
print("New values for riskrating variable")
print(car['riskrating'].unique())
print("")

# Change any similar values
car['manufacturer'].mask(car['manufacturer'] == 'Maxda', "Mazda", inplace=True)
car['manufacturer'].mask(car['manufacturer'] == 'Porcshce', "Porsche", inplace=True)
car['manufacturer'].mask(car['manufacturer'] == 'Toyouta', "Toyota", inplace=True)
car['manufacturer'].mask(car['manufacturer'] == 'Vokswagen', "Volkswagen", inplace=True)
car['manufacturer'].mask(car['manufacturer'] == 'Vw', "Volkswagen", inplace=True)
print("New values for manufacturer variable")
print(car['manufacturer'].unique())
print("")

# Make a list of columns
category = car.select_dtypes('object').columns
print("categorical variables")
print(category)
print("")
numerical = car.select_dtypes('number').columns
print("numerical variables")
print(numerical)
print("")

# Check unique value of categorical variables
for p in category:
    print(p)
    print(car[p].unique())
    print("");

# Data visualization using histogram and boxplot for numerical variables
for col in numerical:
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    sns.histplot(data=car, x=col, ax=ax[0])
    sns.boxplot(data=car, y=col, ax=ax[1])
    plt.show();

# Data visualization using barplot for categorical variabless
for t in category:
    sns.barplot(x = 'car_ID',y = t, data = car, errorbar=None)
    plt.show();

# Data visualization using boxplot for bivariate analysis (cat-num)
for q in category:
    sns.boxplot(x=q, y="price", data=car)
    plt.show();

# Correlation visualization using heatmap for bivariate analysis (num-num)
print(car.corr())
sns.heatmap(car.corr(), annot=True, fmt=".2f", linewidths=.5)
plt.show()

# Crostabb for bivariate analysis (cat-cat)
print("Crosstab for 2 variables")
ct1 = pd.crosstab(car.enginetype,car.aspiration, margins=True)
print(ct1)
print("")

# Crostabb 2 for bivariate analysis (cat-cat)
print("Crosstab for 3 variables")
ct2 = pd.crosstab(index= car["enginetype"],
                  columns= [car["aspiration"], 
                            car["cylindernumber"]],
                  margins=True)
print(ct2)
print("")

# Heatmap between categorical variables
sns.heatmap(pd.crosstab(car.enginetype,car.aspiration), annot=True, fmt=".1f", linewidths=.5)
plt.show()

# Grouping Variables (Pivot Table)
print(car.groupby(['manufacturer', 'carbody']).mean(numerical.all()))

# Filtering step 1: Find the index of the spesific rows
car1_index = car.index[(car['manufacturer']=="Toyota") | (car['manufacturer']=='Honda') | (car['manufacturer']=='Nissan')].tolist()
print("Index of rows that gonna be filtered")
print(car1_index)
print("")

# Filtering step 2: Select data on spesific rows and spesific column
car1 = car.iloc[[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 89, 90, 91, 92, 93, 94, 95, 96, 
                 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 150, 151, 152, 153, 154, 155, 156, 157, 
                 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 
                 176, 177, 178, 179, 180, 181]][["manufacturer", "carbody","enginesize","price"]]
print("Filtered Data")
print(car1)
print("")

# Pivot table from filtered data
print("Pivot table from filtered data")
cpv1 = pd.DataFrame(car1.groupby(['manufacturer', 'carbody']).mean())
print(cpv1)
print("")

# Data Visualization from filtered dataset
sns.barplot(x = 'price',y = 'manufacturer',hue = 'carbody',data = car1, errorbar=None)
plt.show()

# Handling outliers with natural logarithm
# logdata = car.copy()                                            #copy real dataset
# numerical_logdata = logdata.select_dtypes('number').columns     #make a list of numerical variables
# category_logdata = car.select_dtypes('object').columns          #make a list of categorical variables

# for z in numerical_logdata:
#     logdata[z] = np.log(logdata[z]);                            #change all numerical variables into natural logarithm

# print(logdata)                                                  #check if all the values are transformed

# for zi in category_logdata:                                     #visualization the tranformed variables
#     sns.boxplot(x=zi, y="price", data=logdata)
#     plt.show();

# Variables for linear regresssion
cars_lr = car[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
               'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
               'carlength','carwidth','citympg','highwaympg']]
print(cars_lr.info())
print('')

carslr_num = cars_lr.select_dtypes('number').columns

# Find outliers for all numerical variables

for nlro in carslr_num:
    print('{var} data description'.format(var=nlro))
    print(round(cars_lr[nlro].describe()),2)
    print('')
    IQR = cars_lr[nlro].quantile(0.75) - cars_lr[nlro].quantile(0.25)
    Lower_fence = cars_lr[nlro].quantile(0.25) - (IQR * 3)
    Upper_fence = cars_lr[nlro].quantile(0.75) + (IQR * 3)
    print('{outvar} outliers are values < {lowerboundary} or > {upperboundary}'.format(outvar=nlro,lowerboundary=Lower_fence, upperboundary=Upper_fence))
    print('');

# Handling outliers
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [cars_lr]:
    df3['price'] = max_value(df3, 'price', 42648)
    df3['enginesize'] = max_value(df3, 'enginesize', 273)
    df3['horsepower'] = max_value(df3, 'horsepower', 254);

print(cars_lr[['price','enginesize','horsepower']].max())

# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

# Applying the function to the cars_lr
cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)

print(cars_lr.info())

# Train-Test Split and Featue Scaling
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_train[carslr_num] = scaler.fit_transform(df_train[carslr_num])
print(df_train.describe())

# Dividing data into X and y variables

y_train = df_train.pop('price')
X_train = df_train

# Model Building
# RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Model
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm)
rfe = rfe.fit(X_train, y_train)

print(X_train.columns[rfe.support_])

X_train_rfe = X_train[X_train.columns[rfe.support_]]
print(X_train_rfe.head())

# Fuctions
def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X;
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif);

# Model 1
print('Model 1')
X_train_new = build_model(X_train_rfe,y_train)
print('')

# Model 2
X_train_new = X_train_rfe.drop(["dohcv"], axis = 1)
print('Model 2')
X_train_new = build_model(X_train_new,y_train)
print('')

# Calculating the Variance Inflation Factor Model 2
print('VIF Model 2')
print(checkVIF(X_train_new))
print('')

# Model 3
X_train_new = X_train_new.drop(["curbweight","hatchback","sedan"], axis = 1)
print('Model 3')
X_train_new = build_model(X_train_new,y_train)
print('')

# Model 4
X_train_new = X_train_new.drop(["highwaympg","hardtop","wagon"], axis = 1)
print('Model 4')
X_train_new = build_model(X_train_new,y_train)
print('')

# Calculating the Variance Inflation Factor Model 4
print('VIF Model 4')
print(checkVIF(X_train_new))
print('')

# Residual Analysis of Model
lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms
fig = plt.figure()
sns.histplot((y_train - y_train_price), kde=True)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
plt.show()

# Prediction and Evaluation
# Scaling the test set
df_test[carslr_num] = scaler.fit_transform(df_test[carslr_num])
print(df_test.describe())

#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test

# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

# Making predictions
y_pred = lm.predict(X_test_new)

# Evaluation of test via comparison of y_pred and y_test
from sklearn.metrics import r2_score 
print(r2_score(y_test, y_pred))