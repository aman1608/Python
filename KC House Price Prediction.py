## Data Preparation & Exploration
# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
import ppscore as pps
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor  
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree

# Reading Data
train = pd.read_csv(r'C:\Users\WIN8\Desktop\PGDS 2020\Assignments\KC House Prediction\wk3_kc_house_train_data.csv')
validate = pd.read_csv(r"C:\Users\WIN8\Desktop\PGDS 2020\Assignments\KC House Prediction\wk3_kc_house_valid_data.csv")
test = pd.read_csv(r"C:\Users\WIN8\Desktop\PGDS 2020\Assignments\KC House Prediction\wk3_kc_house_test_data.csv")

# Making a copy of the data
train_original = train.copy()
validate_original = validate.copy()
test_original = test.copy()

# Checking Dimensions
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('----------------------------')
print ('The validate data has {0} rows and {1} columns'.format(validate.shape[0],validate.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))

train.columns
train.head()

# Checking Data Types of the variables
train.dtypes

train.describe()

## Univariate Analysis

# Plotting a Histogram for Target Variable.
plt.hist(train.price ,bins=20, range = (80000,3000000), label = "Histogram of Price", histtype = "bar")
plt.title('Histogram of Price')
plt.xlabel('Price in US $')
plt.ylabel('Values Counts')
plt.show()

# Plotting a Normalized Cumulative Histogram for Target Variable
plt.hist(train.price ,bins=20, range = (80000,3000000), label = "Price", histtype = "bar", cumulative = True,normed = True)
plt.title('Cumulative Histogram of Price')
plt.xlabel('Price in US $')
plt.ylabel('Cumulative Values Probability')
plt.tight_layout()

# Calculating the Skewness
print ("The Skewness of Price is {}".format(train['price'].skew()))

# Log transforming the Target Variable
log_price = np.log(train['price'])
print ('The Skewness is', log_price.skew())
sns.distplot(log_price)
plt.title('Distplot of Price')
plt.xlabel('Price in US $')
plt.ylabel('Values Probability')

# Visualizing the Categorical Variables.
plt.figure(1)
plt.subplot(141)
train["waterfront"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Waterfront")
plt.xlabel('Waterfront Classes')
plt.ylabel('Values Probability')
plt.subplot(142)
train["view"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = " Bar Plot of View")
plt.xlabel('View Classes')
plt.ylabel('Values Probability')
plt.subplot(143)
train["condition"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Condition")
plt.xlabel('Condition Classes')
plt.ylabel('Values Probability')
plt.subplot(144)
train["grade"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Grade")
plt.xlabel('Grade Classes')
plt.ylabel('Values Probability')
plt.show()

# Visualizing the Discrete Numerical Variables
plt.figure(1)
plt.subplot(131)
train["bedrooms"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Bedrooms")
plt.xlabel('Number of Bedrooms')
plt.ylabel('Values Probability')
plt.subplot(132)
train["bathrooms"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = " Bar Plot of Bathrooms")
plt.xlabel('Number of Bathrooms')
plt.ylabel('Values Probability')
plt.subplot(133)
train["floors"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Floors")
plt.xlabel('Number of Floors')
plt.ylabel('Values Probability')
plt.show()

# Calculating the Skewness
print ("The Skewness of Bedrooms is {}".format(train['bedrooms'].skew()))
print ('----------------------------')
print ("The Skewness of Bathrooms is {}".format(train['bathrooms'].skew()))
print ('----------------------------')
print ("The Skewness of Floors is {}".format(train['floors'].skew()))

# Checking for Anomalies in Bedroom Variable, as seen in the describe summary
max(train.bedrooms)
train[train.bedrooms == 33]
# Correcting the Anomaly
train.bedrooms[train.bedrooms == 33] = 3

# Visualizing the Year Built Variable
train["yr_built"].value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Year Built")
plt.xlabel('Year')
plt.ylabel('Values Probability')
plt.show()

# Visualizing the Year Renovated Variable
train["yr_renovated"].value_counts(normalize = True, sort = True)

# Visualizing the Square Feet Living Variable
plt.figure(1)
plt.subplot(121)
sns.distplot(train['sqft_living'])
plt.xlabel('Square Feet Living')
plt.ylabel('Values Probability')
plt.subplot(122)
train['sqft_living'].plot.box(figsize = (16,5), notch = True)
plt.xlabel('Square Feet Living')
plt.ylabel('Number of Observations')
plt.show()

# Calculating the Skewness
print ("The Skewness of Square Feet Living is {}".format(train['sqft_living'].skew()))

# Log transforming the Variable
log_sqft_living = np.log(train['sqft_living'])
print ('The Skewness is', log_sqft_living.skew())
sns.distplot(log_sqft_living)
plt.title('Distplot of sqft_living')
plt.xlabel('Square Feet Living')
plt.ylabel('Values Probability')

# Visualizing the Square Feet Lot Variable
plt.figure(1)
plt.subplot(121)
train["sqft_lot"].hist(bins = 30)
plt.title('Histogram of sqft_lot')
plt.xlabel('Square Feet Lot')
plt.ylabel('Frequency')
plt.subplot(122)
train['sqft_lot'].plot.box(figsize = (16,5), notch = True)
plt.title('Box Plot of sqft_lot')
plt.xlabel('Square Feet Lot')
plt.ylabel('Frequency')
plt.show()

# Calculating the Skewness
print ("The Skewness of Square Feet Lot is {}".format(train['sqft_lot'].skew()))

# Log transforming the Variable
log_sqft_lot = np.log(train['sqft_lot'])
print ('The Skewness is', log_sqft_lot.skew())
log_sqft_living.hist(bins = 30)
plt.title('Histogram of sqft_lot(log)')
plt.xlabel('Square Feet Lot')
plt.ylabel('Frequency')

# Visualizing the Square Feet Above Variable
plt.figure(1)
plt.subplot(121)
train["sqft_above"].hist(bins = 30)
plt.title('Histogram of sqft_above')
plt.xlabel('Square Feet Above')
plt.ylabel('Frequency')
plt.subplot(122)
train['sqft_above'].plot.box(figsize = (16,5), notch = True)
plt.title('Box Plot of sqft_above')
plt.xlabel('Square Feet Above')
plt.ylabel('Frequency')
plt.show()

# Calculating the Skewness
print ("The Skewness of Square Feet Above is {}".format(train['sqft_above'].skew()))

# Log transforming the Variable
log_sqft_above = np.log(train['sqft_above'])
print ('The Skewness is', log_sqft_above.skew())
log_sqft_above.hist(bins = 30)
plt.title('Histogram of sqft_above(log)')
plt.xlabel('Square Feet Above')
plt.ylabel('Frequency')

# Visualizing the Square Feet Above Variable
plt.figure(1)
plt.subplot(121)
train["sqft_basement"].hist(bins = 30)
plt.title('Histogram of sqft_basement')
plt.xlabel('Square Feet Basement')
plt.ylabel('Frequency')
plt.subplot(122)
train['sqft_basement'].plot.box(figsize = (16,5), notch = True)
plt.title('Box Plot of sqft_basement')
plt.xlabel('Square Feet Basement')
plt.ylabel('Frequency')
plt.show()

# Visualizing the Latitude Variable
train["lat"].hist(bins = 20)
plt.title('Histogram of lat')
plt.xlabel('Latitude')
plt.ylabel('Frequency')

# Visualizing the Longitude Variable
train["long"].hist(bins = 30)
plt.title('Histogram of long')
plt.xlabel('Longitude')
plt.ylabel('Frequency')

# Visualizing the Square Feet Living 15 Variable.
sns.distplot(train.sqft_living15, hist = True, kde = False)
plt.title('Histogram of Sqft_Living15')
plt.xlabel('Square Feet Living15')
plt.ylabel('Frequency')

# Visualizing the Square Feet Lot 15 Variable.
sns.distplot(train.sqft_lot15, hist = True, kde = False, bins = 30)
plt.title('Histogram of Sqft_Lot15')
plt.xlabel('Square Feet Lot15')
plt.ylabel('Frequency')

# Computing the difference in sqft_living and sqft_living15 & sqft_lot and sqft_lot15
sqft_living_change = train.sqft_living - train.sqft_living15
sqft_lot_change = train.sqft_lot - train.sqft_lot15

# Encoding Sqft Living Variable -> 1 for additions, 0 for no change and -1 for reduction in square feet area.
sqft_living_15 = []
for i in sqft_living_change:
    if i < 0:
        i = -1
        sqft_living_15.append(i)
    elif i > 0:
        i = 1
        sqft_living_15.append(i)
    else:
        i = 0
        sqft_living_15.append(i)

# Encoding Sqft Lot Variable -> 1 for additions, 0 for no change and -1 for reduction in square feet area.
sqft_lot_15 = []
for i in sqft_lot_change:
    if i < 0:
        i = -1
        sqft_lot_15.append(i)
    elif i > 0:
        i = 1
        sqft_lot_15.append(i)
    else:
        i = 0
        sqft_lot_15.append(i)

# Creating a dataframe of the encoded variables.
sqft_change = pd.DataFrame()
sqft_change['sqft_living_15'] = sqft_living_15
sqft_change['sqft_lot_15'] = sqft_lot_15

# Visualizing the relationship
pd.crosstab(sqft_change.sqft_lot_15, sqft_change.sqft_living_15, normalize = True)

# Encoding Year Renovation Variable ->  0 for no renovation and 1 for renovation irrespective of the year.
year_renovation = []
for i in train.yr_renovated:
    if i == 0:
        i = 0
        year_renovation.append(i)
    else:
        i = 1
        year_renovation.append(i)
  
  # Adding the encoded variable in the dataframe created above.
sqft_change['year_renovation'] = year_renovation

# Visualizing the relationship with Year Renovated
pd.crosstab(sqft_change.year_renovation, sqft_change.sqft_living_15)

#Visualizing the relationship with Year Renovated
pd.crosstab(sqft_change.year_renovation, sqft_change.sqft_lot_15)

#Visualizing the ZipCode Variable
unique_zipcodes = []
for i in train.zipcode:
    if i not in unique_zipcodes:
        unique_zipcodes.append(i)
len(unique_zipcodes)
print ("The Number of Unique Zipcodes is {}.".format(len(unique_zipcodes)))

train.zipcode.value_counts(normalize = False).plot.bar(figsize = (20,5), title = "Bar Plot of ZipCodes")
plt.xlabel('Zipcodes')
plt.ylabel('Frequency')
plt.show()

# Visualizing the yr_renovated Variable.
train.yr_renovated.value_counts(normalize = True).plot.bar(figsize = (20,5), title = "Bar Plot of Year Renovated")
plt.xlabel('Year Renovated')
plt.ylabel('Frequency')
plt.show()

# Calculating the number of houses that were not renovated
renovate_year = []
for i in train.yr_renovated:
    if i == 0:
        renovate_year.append(i)
len(renovate_year)

## Bivariate Analysis

# Plotting PP Score Matrix.
warnings.filterwarnings("ignore")
plt.figure(figsize=(16,12))
sns.heatmap(pps.matrix(train),annot=True,fmt=".2f")

# Creating a dataframe with the transformed variables for computing the Correlation Matrix.
train_transformed = train.copy()
train_transformed.drop(["price","sqft_living","sqft_lot","sqft_above"], axis = 1)


train_transformed["price"] = log_price
train_transformed["sqft_living"] = log_sqft_living
train_transformed["sqft_lot"] = log_sqft_lot
train_transformed["sqft_above"] = log_sqft_above

# Plotting a Corelation Matrix
corrMatrix = train_transformed.corr()
plt.figure(figsize=(16,12))
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Visualizing Price & Sqft_Above
train_transformed.plot(kind = "scatter",
         x = "sqft_above", y = "price",
         color = "cornflowerblue",
         figsize=(5,5))

plt.xlabel("Square Feet Above", fontsize = 12)
plt.ylabel("Prices of Houses", fontsize = 12)
plt.title("Scatter Plot", fontsize = 14)
plt.show()

# Visualizing Price & Sqft_Living
train_transformed.plot(kind = "scatter",
         x = "sqft_living", y = "price",
         color = "cornflowerblue",
         figsize=(5,5))

plt.xlabel("Square Feet Living", fontsize = 12)
plt.ylabel("Prices of Houses", fontsize = 12)
plt.title("Scatter Plot", fontsize = 14)
plt.show()

# Visualizing Price & bathrooms
train_transformed.plot(kind = "scatter",
         x = "bathrooms", y = "price",
         color = "cornflowerblue",
         figsize=(5,5))

plt.xlabel("Number of Bathrooms", fontsize = 12)
plt.ylabel("Price of Houses", fontsize = 12)
plt.title("Scatter Plot", fontsize = 14)
plt.show()

# Visualizing Price & Grade
train_transformed.plot(kind = "scatter",
         x = "grade", y = "price",
         color = "cornflowerblue",
         figsize=(5,5))

plt.xlabel("Grade", fontsize = 12)
plt.ylabel("Prices of Houses", fontsize = 12)
plt.title("Scatter Plot", fontsize = 14)
plt.show()
        
plt.figure(figsize = (15,18))
ax = plt.subplot(521)
sns.scatterplot(x = 'bedrooms', y = 'price', data=train_transformed, ax = ax)
ax = plt.subplot(522)
sns.scatterplot(x = 'floors', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(523)
sns.scatterplot(x = 'waterfront', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(524)
sns.scatterplot(x = 'condition', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(525)
sns.scatterplot(x = 'yr_built', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(526)
sns.scatterplot(x = 'yr_renovated', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(527)
sns.scatterplot(x = 'lat', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(528)
sns.scatterplot(x = 'long', y ='price', data=train_transformed, ax = ax)
ax = plt.subplot(529)
sns.scatterplot(x = 'zipcode', y ='price', data=train_transformed)
ax = plt.subplot(5,2,10)
sns.scatterplot(x = 'sqft_lot', y ='price', data=train_transformed)
plt.show()

## Model Building
#### Simple Linear Regression

train_transformed.columns
# Dropping Variables
train_transformed = train_transformed.drop(['waterfront', 'view', 'id', 'date','floors', 'condition', 'sqft_basement',
                                           'yr_renovated', 'zipcode','lat','long','sqft_living15','sqft_lot15'], axis = 1)
                                    
# Preparing Validation Dataset and dropping variables.
validate_transformed = validate.copy()
validate_transformed['price'] = np.log(validate['price'])
validate_transformed['sqft_living'] = np.log(validate['sqft_living'])
validate_transformed['sqft_lot'] = np.log(validate['sqft_lot'])
validate_transformed['sqft_above'] = np.log(validate['sqft_above'])
validate_transformed = validate_transformed.drop(['waterfront', 'view', 'id', 'date','floors', 'condition', 'sqft_basement',
                                           'yr_renovated', 'zipcode','lat','long','sqft_living15','sqft_lot15'], axis = 1)

# Setting the Target Variable
y1 = np.array(train_transformed['price']).reshape(9761,1)
y2 = np.array(validate_transformed['price']).reshape(9635,1)

# Segregating Independent Variables in Train Dataset
x1_train = np.array(train_transformed['sqft_living']).reshape(9761,1)
x2_train = np.array(train_transformed['sqft_above']).reshape(9761,1)
x3_train = np.array(train_transformed['sqft_lot']).reshape(9761,1)
x4_train = np.array(train_transformed['bathrooms']).reshape(9761,1)
x5_train = np.array(train_transformed['bedrooms']).reshape(9761,1)
x6_train = np.array(train_transformed['yr_built']).reshape(9761,1)
x7_train = np.array(train_transformed['grade']).reshape(9761,1)

# Segregating Independent Variables in Validate Dataset.
x1_validate = np.array(validate_transformed['sqft_living']).reshape(9635,1)
x2_validate = np.array(validate_transformed['sqft_above']).reshape(9635,1)
x3_validate = np.array(validate_transformed['sqft_lot']).reshape(9635,1)
x4_validate = np.array(validate_transformed['bathrooms']).reshape(9635,1)
x5_validate = np.array(validate_transformed['bedrooms']).reshape(9635,1)
x6_validate = np.array(validate_transformed['yr_built']).reshape(9635,1)
x7_validate = np.array(validate_transformed['grade']).reshape(9635,1)

# Fitting Simple Linear Regression Models
regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
regr3 = linear_model.LinearRegression()
regr4 = linear_model.LinearRegression()
regr5 = linear_model.LinearRegression()
regr6 = linear_model.LinearRegression()
regr7 = linear_model.LinearRegression()

regr1.fit(x1_train,y1)
regr2.fit(x2_train,y1)
regr3.fit(x3_train,y1)
regr4.fit(x4_train,y1)
regr5.fit(x5_train,y1)
regr6.fit(x6_train,y1)
regr7.fit(x7_train,y1)

# Making Predictions on Validation Dataset
y_pred1 = regr1.predict(x1_validate)
y_pred2 = regr2.predict(x2_validate)
y_pred3 = regr3.predict(x3_validate)
y_pred4 = regr4.predict(x4_validate)
y_pred5 = regr5.predict(x5_validate)
y_pred6 = regr6.predict(x6_validate)
y_pred7 = regr7.predict(x7_validate)

# Calculating the RMSE for the 7 Simple Linear Models
rmse1 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred1)))**2))
rmse2 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred2)))**2))
rmse3 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred3)))**2))
rmse4 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred4)))**2))
rmse5 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred5)))**2))
rmse6 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred6)))**2))
rmse7 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred7)))**2))

print('The RMSE for Price and sqft_living is:', round(rmse1,5))
print ('----------------------------')
print('The RMSE for Price and sqft_above is:', round(rmse2,5))
print ('----------------------------')
print('The RMSE for Price and sqft_lot is:', round(rmse3,5))
print ('----------------------------')
print('The RMSE for Price and bathrooms is:', round(rmse4,5))
print ('----------------------------')
print('The RMSE for Price and bedrooms is:', round(rmse5,5))
print ('----------------------------')
print('The RMSE for Price and yr_built is:', round(rmse6,5))
print ('----------------------------')
print('The RMSE for Price and grade is:', round(rmse7,5))

# Calculating the R square score for the 7 Simple Linear Models
r1 = r2_score(y2, y_pred1, multioutput='variance_weighted')
r2 = r2_score(y2, y_pred2, multioutput='variance_weighted')
r3 = r2_score(y2, y_pred3, multioutput='variance_weighted')
r4 = r2_score(y2, y_pred4, multioutput='variance_weighted')
r5 = r2_score(y2, y_pred5, multioutput='variance_weighted')
r6 = r2_score(y2, y_pred6, multioutput='variance_weighted')
r7 = r2_score(y2, y_pred7, multioutput='variance_weighted')

print('The R-square for Price and sqft_living is:', round(r1,5))
print ('----------------------------')
print('The R-square for Price and sqft_above is:', round(r2,5))
print ('----------------------------')
print('The R-square for Price and sqft_lot is:', round(r3,5))
print ('----------------------------')
print('The R-square for Price and bathrooms is:', round(r4,5))
print ('----------------------------')
print('The R-square for Price and bedrooms is:', round(r5,5))
print ('----------------------------')
print('The R-square for Price and yr_built is:', round(r6,5))
print ('----------------------------')
print('The R-square for Price and grade is:', round(r7,5))

#### Multiple Linear Regression with Backward Selection

# Creating Variables for Multiple Linear Regression with 5 Independent Variables
# Model 1 -> All 5 Independent Variables.
x8_train = pd.DataFrame(train_transformed[['grade','bathrooms','sqft_above','sqft_living', 'sqft_lot']])
x8_validate = pd.DataFrame(validate_transformed[['grade','bathrooms','sqft_above','sqft_living', 'sqft_lot']])

# Fitting Multiple Linear Regression Model
regr8 = linear_model.LinearRegression()
regr8.fit(x8_train,y1)

# Making Predictions on Validation Dataset
y_pred8 = regr8.predict(x8_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse8 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred8)))**2))
r8 = r2_score(y2, y_pred8, multioutput='variance_weighted')
adjr1 = 1 - (((1 - r8)*(len(x8_train) - 1))/(len(x8_train) - len(x8_train.columns) - 1))

print('The RMSE is:', round(rmse8,5))
print ('----------------------------')
print('The R-square is:', round(r8,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr1,5))

# Dropping Variables from our model based on their respective Simple Linear Models
# Model 2 -> Dropping Bathrooms
# Creating Variables for Multiple Linear Regression with 4 Independent Variables
x9_train = pd.DataFrame(train_transformed[['grade','sqft_above','sqft_living', 'sqft_lot']])
x9_validate = pd.DataFrame(validate_transformed[['grade','sqft_above','sqft_living', 'sqft_lot']])

# Fitting Multiple Linear Regression Model
regr9 = linear_model.LinearRegression()
regr9.fit(x9_train,y1)

# Making Predictions on Validation Dataset
y_pred9 = regr9.predict(x9_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse9 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred9)))**2))
r9 = r2_score(y2, y_pred9, multioutput='variance_weighted')
adjr2 = 1 - (((1 - r9)*(len(x9_train) - 1))/(len(x9_train) - len(x9_train.columns) - 1))

print('The RMSE is:', round(rmse9,5))
print ('----------------------------')
print('The R-square is:', round(r9,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr2,5))

# Model 3 -> Dropping Grade
# Creating Variables for Multiple Linear Regression with 4 Independent Variables
x10_train = pd.DataFrame(train_transformed[['bathrooms','sqft_above','sqft_living', 'sqft_lot']])
x10_validate = pd.DataFrame(validate_transformed[['bathrooms','sqft_above','sqft_living', 'sqft_lot']])

# Fitting Multiple Linear Regression Model
regr10 = linear_model.LinearRegression()
regr10.fit(x10_train,y1)

# Making Predictions on Validation Dataset
y_pred10 = regr10.predict(x10_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse10 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred10)))**2))
r10 = r2_score(y2, y_pred10, multioutput='variance_weighted')
adjr3 = 1 - (((1 - r10)*(len(x10_train) - 1))/(len(x10_train) - len(x10_train.columns) - 1))

print('The RMSE is:', round(rmse10,5))
print ('----------------------------')
print('The R-square is:', round(r10,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr3,5))

# Model 4 -> Dropping sqft_living
# Creating Variables for Multiple Linear Regression with 4 Independent Variables
x11_train = pd.DataFrame(train_transformed[['bathrooms','sqft_above','sqft_living', 'sqft_lot']])
x11_validate = pd.DataFrame(validate_transformed[['bathrooms','sqft_above','sqft_living', 'sqft_lot']])

# Fitting Multiple Linear Regression Model
regr11 = linear_model.LinearRegression()
regr11.fit(x11_train,y1)

# Making Predictions on Validation Dataset
y_pred11 = regr11.predict(x11_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse11 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred11)))**2))
r11 = r2_score(y2, y_pred11, multioutput='variance_weighted')
adjr4 = 1 - (((1 - r11)*(len(x11_train) - 1))/(len(x11_train) - len(x11_train.columns) - 1))

print('The RMSE is:', round(rmse11,5))
print ('----------------------------')
print('The R-square is:', round(r11,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr4,5))

# Model 5 -> Dropping sqft_lot
# Creating Variables for Multiple Linear Regression with 4 Independent Variables
x12_train = pd.DataFrame(train_transformed[['bathrooms','sqft_above','sqft_living', 'grade']])
x12_validate = pd.DataFrame(validate_transformed[['bathrooms','sqft_above','sqft_living', 'grade']])

# Fitting Multiple Linear Regression Model
regr12 = linear_model.LinearRegression()
regr12.fit(x12_train,y1)

# Making Predictions on Validation Dataset
y_pred12 = regr12.predict(x12_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse12 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred12)))**2))
r12 = r2_score(y2, y_pred12, multioutput='variance_weighted')
adjr5 = 1 - (((1 - r12)*(len(x12_train) - 1))/(len(x12_train) - len(x12_train.columns) - 1))

print('The RMSE is:', round(rmse12,5))
print ('----------------------------')
print('The R-square is:', round(r12,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr5,5))

# Model 6 -> Dropping sqft_above
# Creating Variables for Multiple Linear Regression with 4 Independent Variables
x13_train = pd.DataFrame(train_transformed[['bathrooms','sqft_lot','sqft_living', 'grade']])
x13_validate = pd.DataFrame(validate_transformed[['bathrooms','sqft_lot','sqft_living', 'grade']])

# Fitting Multiple Linear Regression Model
regr13 = linear_model.LinearRegression()
regr13.fit(x13_train,y1)

# Making Predictions on Validation Dataset
y_pred13 = regr13.predict(x13_validate)

#Computing RMSE, R square & Adjusted R-Square
rmse13 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred13)))**2))
r13 = r2_score(y2, y_pred13, multioutput='variance_weighted')
adjr6 = 1 - (((1 - r13)*(len(x13_train) - 1))/(len(x13_train) - len(x13_train.columns) - 1))

print('The RMSE is:', round(rmse13,5))
print ('----------------------------')
print('The R-square is:', round(r13,5))
print ('----------------------------')
print('The Adjusted R-square is:', round(adjr6,5))

#### Polynomial Regression

# Preparing Data
polyfeat = PolynomialFeatures(degree = 2)
x_train_poly = polyfeat.fit_transform(train_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])
x_validate_poly = polyfeat.fit_transform(validate_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])

# Fitting Polynomial Regression -> Degree 2
poly = linear_model.LinearRegression()
poly.fit(x_train_poly,train_transformed['price'])
poly_pred = poly.predict(x_validate_poly)

# Computing the RMSE
rmse14 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(poly_pred)))**2))

# Computing the R Square
r14 = r2_score(y2, poly_pred, multioutput='variance_weighted')

print('The RMSE is:', round(rmse14,5))
print ('----------------------------')
print('The R-square is:', round(r14,5))
print ('----------------------------')
print('Intercept: ', poly.intercept_)
print ('----------------------------')
print('Coefficient:', poly.coef_)

# Preparing Data
polyfeat1 = PolynomialFeatures(degree = 3)
x_train_poly1 = polyfeat1.fit_transform(train_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])
x_validate_poly1 = polyfeat1.fit_transform(validate_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])

# Fitting Polynomial Regression -> Degree 2
poly1 = linear_model.LinearRegression()
poly1.fit(x_train_poly1,train_transformed['price'])
poly_pred1 = poly1.predict(x_validate_poly1)

# Computing the RMSE
rmse15 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(poly_pred1)))**2))

# Computing the R-squared
# Computing the R Square
r15 = r2_score(y2, poly_pred1, multioutput='variance_weighted')

print('The RMSE is:', round(rmse15,5))
print ('----------------------------')
print('The R-square is:', round(r15,5))
print ('----------------------------')
print('Intercept: ', poly1.intercept_)
print ('----------------------------')
print('Coefficient:', poly1.coef_)

#### Regression Tree

# Creating Tree Model
x14_train = pd.DataFrame(train_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])
x14_validate = pd.DataFrame(validate_transformed[['sqft_living','grade', 'sqft_lot', 'sqft_above']])
regr14 = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 5)
regr14.fit(x14_train, y1)
#Predicting
y_pred14 = regr14.predict(x14_validate)

# Plotting the Tree
data_feature_names = ['sqft_living','grade']
cn = ['price']
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(regr14, feature_names = data_feature_names, class_names = cn, filled = True)

# Computing the RMSE
rmse16 = np.sqrt(np.mean((np.array(np.exp(y2)) - (np.exp(y_pred14)))**2))

# Computing the R-squared
r16 = r2_score(y2, y_pred14, multioutput='variance_weighted')
print('The RMSE for the Regression Tree is:', round(rmse16,5))
print ('----------------------------')
print('The R-squarred value for Price and sqft_living,grade is:', round(r16,5))

## Final Model

# We select the Multiple Linear Regression Model with grade,sqft_living, sqft_lot, sqft_above as our final Model, as we got the lowest RMSE score for it
# Let us apply the model on the testing dataset and see the results
# Preparing the the test data
test_transformed = pd.DataFrame()
test_transformed['price'] = np.log(test['price'])
test_transformed['sqft_living'] = np.log(test['sqft_living'])
test_transformed['grade'] = test['grade']
test_transformed['sqft_lot'] = np.log(test['sqft_lot'])
test_transformed['sqft_above'] = np.log(test['sqft_above'])

#Setting Variables
y_test = np.array(test_transformed['price']).reshape(2217,1)
x_test = test_transformed[['grade','sqft_above','sqft_living', 'sqft_lot']]


# Making Predictions on Validation Dataset
y_pred_test = regr9.predict(x_test)

# Computing the RMSE
rmse17 = np.sqrt(np.mean((np.array(np.exp(y_test)) - (np.exp(y_pred_test)))**2))


# Computing the R-square
r17 = r2_score(y_test, y_pred_test, multioutput='variance_weighted')
                 
print('The RMSE is:', round(rmse17,5))
print ('----------------------------')                  
print('The R-square is:', round(r17,5))
print ('----------------------------')                 
print('Intercept: ', regr9.intercept_)
print ('----------------------------')
print('Coefficient:', regr9.coef_)                                  