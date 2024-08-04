#Anthony Plasencia
#Comp 541
#Import necessary libraries (e.g., pandas, numpy).
# 1. Setup and Initial Exploration

import csv
import pandas as pd
import numpy as nu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

#new lines
def newLine():
    print("--------------------------------------------------------------------------------------------------------------------------------------")

# Data loading
df = pd.read_csv('pizza_sales.csv')

# Preliminary Datta Exploration
# Inspect the data
print("The Data Head:")
print(df.head())
newLine()


# Summary Statistics
print("The Data Info:")
print(df.info())
newLine()


# Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())
print("There are no missing values")
newLine()


# Data clraning and Preperation
# Handle missing values
print("No missing values to handle")
newLine()

# Covert Data types
def convert_date_format(date_str):
    for fmt in ('%m/%d/%Y', '%d-%m-%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')

df["order_time"] = pd.to_datetime(df["order_time"], format='%H:%M:%S').dt.time

df["order_date"] = df["order_date"].apply(convert_date_format)
df["order_date"] = pd.to_datetime(df["order_date"])

for col in ["pizza_size" ,"pizza_category", "pizza_ingredients", "pizza_name_id" ,"pizza_name" ]:
    df[col] = df[col].astype("category")
for col in ["pizza_id", "order_id", "quantity","unit_price","total_price"]:
    df[col] = df[col].astype(int)

# Exploratory Data Analysis (EDA)--------------------------------------------------------
#Set up Visualisation Libraries see lines #5-9

# Visualise Your data
plt.rc('figure', figsize=(16, 9))
sns.countplot(x="pizza_category", data=df)
plt.title("Pizza Category")
plt.show()
#simple bar chart showing the number of pizzas sold in each category shows that the most popular category is Classic 
#and the least popular is chicken although the difference is not that big between the categories except for Classic


#Sales Trends Analysis
df.set_index('order_date', inplace=True)
monthly_sales = df['total_price'].resample('ME').sum()
monthly_sales.plot(kind='line')
plt.title("Monthly Sales")
plt.show()
#The Line chat shows monthly sales, it shows that buisness was slow in the month of february and september and peaks in march and november



#Performance by Category and Size 
sales_by_category_and_size = df.groupby(["pizza_category", "pizza_size"],observed=False)["total_price"].sum().reset_index()
sns.barplot(x="pizza_category", y="total_price", hue="pizza_size", data=sales_by_category_and_size)
plt.title("Sales Performance by Pizza Category and Size")
plt.xlabel("Pizza Category")
plt.ylabel("Total Sales")
plt.show()
#The bar chart breaks down the categories by size and shows that the most popular size is Large across all
# and the Categories across all the categories there is a clear preference for large pizzas and then medium and then small
#although the Classic has a relatiley even distribution between the sizes


#Popularity Analysis
popular_pizzas = df.groupby(["pizza_name"],observed=False)["quantity"].sum().reset_index()
sns.barplot(y="pizza_name", x="quantity", data=popular_pizzas)
plt.title("Popularity of Pizzas")
plt.ylabel("Pizza Name")
plt.xlabel("Quantity Sold")
plt.xticks(fontsize=10, rotation=90)
plt.show()
#The bar chart shows the popularity of each pizza and the most popular pizzas are the BBQ Chicken, Calfiornia Chicken, Classic Deluxe
#Hawaiian, Pepperoni, and Tai Chicken pizza. The Least popular pizza is clearlly the Bire Carre pizza

#Ingredient Analysis
df['pizza_ingredients'] = df['pizza_ingredients'].str.split(',')
ingredientList = df.explode('pizza_ingredients')
ingredients = ingredientList['pizza_ingredients'].value_counts().reset_index()
ingredients.columns = ['ingredient', 'count']
sns.barplot(x='count', y='ingredient', data=ingredients)
plt.title('Number of Pizzas That Use Each Ingredient')
plt.xlabel('Count')
plt.ylabel('Ingredient')
plt.show()
#The bar chart shows the number of pizzas that use each ingredient and the most popular ingredient is Garlic, tomatoes ,red onions
#and chicken while the least popular ingredients are thyme,pears and caramelized onions. The graph is Strongly right skewed where the top 
#are used over 10000 times and the rest are used about or less than 5000 times

#Recommendation System--------------------------------------------------------
# Data preparation
matrix = df.groupby(['order_id', 'pizza_name'],observed=False)['quantity'].sum().unstack(fill_value=0)

# Correlation Analysis
correlation_matrix = matrix.corr()
sns.heatmap(correlation_matrix)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation Matrix')
plt.show()
#the heatmap shows the correlation between the pizzas and the correlation between the pizzas is very low there basically is no correlation
#between the pizzas at all
 

def recommend_pizzas(order_id, matrix, correlation_matrix, n_recommendations):
    ordered_pizzas = list(set(matrix.loc[order_id][matrix.loc[order_id] > 0].index))
    correlation_scores = correlation_matrix[ordered_pizzas].sum(axis=1)
    correlation_scores = correlation_scores[~correlation_scores.index.isin(ordered_pizzas)]
    recommended_pizzas = correlation_scores.nlargest(n_recommendations)
    return recommended_pizzas.index.tolist()


print(recommend_pizzas(92, matrix, correlation_matrix, 5))

#Insights ationanle recommendations conlusion and futere works
#A lot of the data can inform a buisness owner on how to improve their buisness, for example the popularity of the pizzas can inform the owner
#on what toppings to order more of and what toppings to order less of. The sales trends can inform the owner on when to expect the most sales
#Also what sizes to of pizza dough to order more of and what sizes to order less of.