import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/Online Sales Data.csv')

df.info()
df.head()
df.describe()

df.isnull().sum() # No missing values
df.duplicated().sum() # No duplicate rows

# Exploratory Data Analysis
sns.countplot(x='Product Category', data=df)
plt.title('Orders by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show() #40 in each category

sns.barplot(x='Product Category', y='Total Revenue', data=df, estimator=sum)
plt.title('Total Revenue by Category')
plt.xticks(rotation=45)
plt.show() # Electronics has the highest revenue

sns.barplot(x='Product Category', y='Total Revenue', data=df, estimator=np.mean)
plt.title('Average Revenue by Category')
plt.xticks(rotation=45)
plt.show() # Electronics has the highest average revenue

df['Date'] = pd.to_datetime(df['Date'])
sales_over_time = df.groupby('Date')['Total Revenue'].sum().reset_index()
sns.lineplot(x='Date', y='Total Revenue', data=sales_over_time)
plt.title('Daily Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show() 

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show() # Unit price and total revenue are highly correlated (.93) and Unit price and Units sold are slightly correlated (-.31)

df['day_of_week'] = df['Date'].dt.day_name()
sns.countplot(x='day_of_week', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Orders by Day of Week')
plt.xticks(rotation=45)
plt.show() # Nearly equal distribution of orders across days

sns.barplot(x='day_of_week', y='Total Revenue', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], estimator=sum)
plt.title('Total Revenue by Day of Week')
plt.xticks(rotation=45)
plt.show() # Tuesday and Friday have the highest revenue

# Data Preprocessing

df['Date'] = pd.to_datetime(df['Date']) # Convert Date to datetime
df['Product Category Code'] = df['Product Category'].astype('category') # Convert Product Category to category obj in a new column
df['Product Category Code'] = df['Product Category Code'].cat.codes # Convert Product Category to numerical codes for ML models
df['day_of_week Code'] = df['day_of_week'].astype('category') # Convert day_of_week to category obj in a new column
df['day_of_week Code'] = df['day_of_week Code'].cat.codes # Convert day_of_week to numerical codes for ML models

df.describe()
df.head()

# Modeling
daily_sales = df.groupby('Date')['Total Revenue'].sum().reset_index() # Group by Date and sum sales
daily_sales = daily_sales.sort_values('Date') # Sort by Date (ie. groups by day of the week)

daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
daily_sales.set_index('Date', inplace=True)

daily_sales = daily_sales.asfreq('D').fillna(0) # Fill missing dates with 0 sales


plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales['Total Revenue'], marker='o', linestyle='-')
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show() 

import pmdarima as pm






