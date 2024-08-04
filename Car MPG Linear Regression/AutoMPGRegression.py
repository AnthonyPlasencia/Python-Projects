from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score , accuracy_score
  
# fetch dataset it is from the UCI repository and it is the Auto MPG dataset
# The dataset contains 398 rows and 8 columns. The target variable is mpg 
# Using the features in the dataset, we can try to predict the miles per gallon of a car
# Using Linear Regression
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
 
# create a dataframe  
df= pd.concat([X, y], axis=1)
# save the dataframe as a csv file
df.to_csv("auto_mpg.csv", index=False)

#Method to print a new line
def newLine():
    print("\n")

#Load the data
print("The Data Head:")
print(df.head())
newLine()

#Data Info
print("The Data Info:")
print(df.info())
newLine()

#Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())
newLine()

#Drop the 6 rows with missing values this is because the missing values are only 6 
#and dropping them will not affect the data
df=df.dropna()

#Check for missing values after dropping the rows with missing values
print("Checking for missing values after dropping rows with missing values:")
print(df.isnull().sum())
newLine()

#split the data into features and target
x = df.drop("mpg", axis=1)
y = df["mpg"]

#Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Apply Linear Regression to the data to predict the miles per gallon of a car
model = LinearRegression()
model.fit(x_train, y_train)
modelPrediction = model.predict(x_test)

#Check the results of the model by checking the Mean Squared Error and the R2 Score
MSE = mean_squared_error(y_test, modelPrediction)
R2 = r2_score(y_test, modelPrediction)

#Print the results of the model and the coefficients of the features 
print("Results After applying Linear Regression:")
print(f"Mean Squared Error: {MSE}")
print(f"R2 Score: {R2}")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef}")


#The Mean Squared error is 10.71 this means that doing the square root of gives us 3.27 This means that the model is 
#off by 3.27 miles per gallon on average from the actual value. 
#The R2 score is 0.7901 which means that 79% of the variance in the target variable can be explained by the features.
#this means that the model is a good fit for the data.
#The coefficients are the weights of the features in the model. 
# As the more cylinders, horsepower, and weight the lower the miles per gallon.
# While the better the acceleration, the more recent the model year, the higher the miles per gallon.


#The key takeaways from this linear regression model are that the more cylinders, horsepower, and weight the lower the miles per gallon.
# While the better the acceleration, the more recent the model year, the higher the miles per gallon.
#automakers can use this information to design more fuel-efficient cars by focusing on the features that have a positive impact on miles per 
# gallon and reducing the ones that have a negative impact.
#this will help them meet the increasing demand for fuel-efficient vehicles and reduce their carbon footprint.
#While reducing the features that have a negative impact is a good idea, it is also important to consider the traden
# offs between fuel efficiency and other factors such as performance and cost.