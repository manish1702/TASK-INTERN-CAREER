#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv(r"C:\Users\MANISH MISHRA\Downloads\internship task\youtubers_df.csv")


# In[4]:


df.info()
df.head()


# In[5]:


df.isna().sum()


# In[6]:


df.count()


# In[7]:


df.duplicated().sum()


# In[8]:


df.describe()


# In[9]:


df["Categories"].info()


# In[10]:


df["Country"].info()


# In[11]:


df["Country"].mode()


# In[12]:


country_youtubers_count = df["Country"].value_counts().head(10)
country_youtubers_count


# In[13]:


import matplotlib.pyplot as plt
country_youtubers_count.plot(kind = "bar")
plt.xlabel("Country")
plt.ylabel("Number of Youtubers")
plt.title("Top 10 Countries Having Content Creators")
plt.show()


# In[14]:


total_categories = df["Categories"].value_counts()
total_categories.head(15)


# In[15]:


categories = df['Categories'].unique()
print("Categories in the dataset:")
for categorie in categories:
    print(categorie)


# In[16]:


total_categories = df["Categories"].value_counts()
total_categories.head(15)


# In[17]:


categories = df["Categories"].value_counts().head(12)
categories.plot(kind = "bar")
plt.xlabel("category")
plt.ylabel("count")
plt.title("Top 12 categories")
plt.show()


# In[21]:


suscribers = df.head(5)
plt.figure(figsize=(10,6))
plt.bar(suscribers['Username'],suscribers['Suscribers'])
plt.xlabel='suscribers'
plt.ylabel='Youtuber'
plt.title='Top 5 Youtube Suscribers'
plt.show()


# In[22]:


col_names = df.columns
print(col_names)


# In[23]:


summary = df.describe(include='all')
print(summary)


# In[25]:


numerical_df = df[['Suscribers', 'Visits', 'Likes', 'Comments']]
numerical_df.head()


# In[27]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots


fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Box(y=numerical_df['Suscribers'], name="Suscribers"))
fig.add_trace(go.Box(y=numerical_df['Visits'], name="Visits"))
fig.add_trace(go.Box(y=numerical_df['Likes'], name="Likes"))
fig.add_trace(go.Box(y=numerical_df['Comments'], name="Comments"))

fig.update_layout(title="BOX PLOT SHOWING THE OUTLIERS ACROSS VARIABLES")

fig.show()


# In[28]:


# Columns that we want to check for outliers
columns = ['Suscribers', 'Visits', 'Likes', 'Comments']

# Calculate the z score for each data point
z_scores = (numerical_df[columns] - numerical_df[columns].mean()) / numerical_df[columns].std()

# Setting a threshold for outliers (let's say 2 s.t.d from the mean)
threshold = 2

# Creating a data frame with NA for non-outliers and actual values for outliers
zscore_outliers = numerical_df[columns].copy()

# Replacing non-outliers with NA
zscore_outliers[(z_scores <= -threshold) | (z_scores >= threshold)] = np.nan


# In[30]:


zscore_outliers.head()


# In[31]:


# Counting outliers for each variable
subscriber_outliers = zscore_outliers['Suscribers'].notna().sum()
print('suscriber_outliers:', subscriber_outliers)

visit_outliers = zscore_outliers['Visits'].notna().sum()
print('visit_outliers:', visit_outliers)

likes_outliers = zscore_outliers['Likes'].notna().sum()
print('likes_outliers:', likes_outliers)

comment_outliers = zscore_outliers['Comments'].notna().sum()
print('comment_outliers:', comment_outliers)


# In[33]:


# Finding the number of streamers by category
popular_category = df['Categories'].value_counts()

# Converting the result to a data frame
popular_category_df = popular_category.reset_index()

# Renaming the columns
popular_category_df.columns = ['Category', 'Total No. of Streamers / Category']

# Inspecting our data
print(popular_category_df.head())


# In[41]:


import matplotlib.pyplot as plt

# Creating a bar graph
plt.figure(figsize=(10, 6))
plt.bar(popular_category_df['Category'][:10], popular_category_df['Total No. of Streamers / Category'][:10], color='skyblue')
# plt.title('BAR GRAPH SHOWING THE POPULAR CATEGORIES BY STREAMERS')
# plt.xlabel('Category')
# plt.ylabel('Total No. of Streamers / Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

# Extracting the relevant columns for correlation
correlation_matrix = df[['Suscribers', 'Likes', 'Comments']].corr()

# Creating a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('CORRELATION MATRIX BETWEEN SUBSCRIBERS, LIKES AND COMMENTS')
plt.show()


# In[44]:


# Grouping by 'Country' and 'Username' and counting the number of streamers
streamers_country = df.groupby(['Country', 'Username']).size().reset_index(name='Number of Streamers')

# Renaming columns
streamers_country.rename(columns={'Username': 'Streamers'}, inplace=True)

# Displaying the first few rows
print(streamers_country.head())


# In[47]:


import plotly.graph_objects as go

# Creating the bar chart
fig = go.Figure(data=[go.Bar(
    x=streamers_country['Country'],
    y=streamers_country['Number of Streamers'],
    marker=dict(color=streamers_country['Number of Streamers']),  # Using number of streamers as color
    text=['Streamers: {}<br>Number of Streamers: {}'.format(a, b) for a, b in zip(streamers_country['Streamers'], streamers_country['Number of Streamers'])],
    hoverinfo='text'
)])

# Updating layout
fig.update_layout(title="TOP STREAMERS BY COUNTRY")

# Displaying the plot
fig.show()


# In[49]:


# Grouping by 'Country' and 'Categories' and counting the number of categories
country_category_df = df.groupby(['Country', 'Categories']).size().reset_index(name='Number of categories')

# Sorting by the number of categories in descending order
country_category_df = country_category_df.sort_values(by='Number of categories', ascending=False)

# Displaying the first few rows
print(country_category_df.head())


# In[51]:


import plotly.graph_objects as go

# Creating the bar chart
fig = go.Figure(data=[go.Bar(
    x=country_category_df['Categories'][:10],
    y=country_category_df['Number of categories'][:10],
    marker=dict(color=country_category_df['Number of categories'][:10]),  # Using number of categories as color
    text=['Country: {}<br>Number of categories: {}'.format(a, b) for a, b in zip(country_category_df['Country'][:10], country_category_df['Number of categories'][:10])],
    hoverinfo='text'
)])

# Updating layout
fig.update_layout(title="TOP CATEGORIES BY COUNTRY")

# Displaying the plot
fig.show()



# In[57]:


# Calculate the mean metrics
var_average = df.agg({
    'Suscribers': 'mean',
    'Visits': 'mean',
    'Likes': 'mean',
    'Comments': 'mean'
}).rename({
    'Subcribers': 'mean_subscribers',
    'Visits': 'mean_visits',
    'Likes': 'mean_likes',
    'Comments': 'mean_comments'
}).to_frame().T

# Display the mean metrics
print(var_average)


# In[58]:


import plotly.graph_objects as go

# Creating the bar plot
fig = go.Figure(data=go.Bar(
    x=var_average.columns,
    y=var_average.iloc[0],
    marker=dict(color='purple')
))

# Updating layout
fig.update_layout(title="Average Metrics", xaxis=dict(title="Metrics"), yaxis=dict(title="Mean Value"))

# Displaying the plot
fig.show()


# In[59]:


import plotly.graph_objects as go

# Selecting the top 10 categories
popular_category_df_top10 = popular_category_df.head(10)

# Creating the bar plot
fig = go.Figure(data=go.Bar(
    x=popular_category_df_top10['Category'],
    y=popular_category_df_top10['Total No. of Streamers / Category'],
    marker=dict(color='black')
))

# Updating layout
fig.update_layout(title="BAR GRAPH SHOWING THE TOP CATEGORIES BY STREAMERS",
                  xaxis=dict(title="Category"),
                  yaxis=dict(title="Total No. of Streamers / Category"))

# Displaying the plot
fig.show()


# In[65]:


# Creating the performance metric
df['Performance_Metric'] = (df['Suscribers'] + df['Visits'] + df['Likes'] + df['Comments']) / 4

# Calculating the total mean of the performance metric by category
category_metrics = df.groupby('Categories')['Performance_Metric'].mean().reset_index(name='Total_Mean_Performance')

# Displaying the result
print(category_metrics)


# In[66]:


# Finding the top 10 categories by performance metrics
top_category_metrics = category_metrics.sort_values(by='Total_Mean_Performance', ascending=False).head(10)

# Displaying the result
print(top_category_metrics)


# In[67]:


import plotly.graph_objects as go

# Creating the bar plot
fig = go.Figure(data=go.Bar(
    x=top_category_metrics['Categories'],
    y=top_category_metrics['Total_Mean_Performance'],
    marker=dict(color='red')
))

# Updating layout
fig.update_layout(title="TOP CATEGORIES BY PERFORMANCE METRIC",
                  xaxis=dict(title="Categories"),
                  yaxis=dict(title="Performance Metric"))

# Displaying the plot
fig.show()


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculating average metrics by categories
category_avg_metrics = df.groupby('Categories').agg(
    avg_subscribers=('Suscribers', 'mean'),
    avg_visits=('Visits', 'mean'),
    avg_likes=('Likes', 'mean'),
    avg_comments=('Comments', 'mean')
).reset_index()

# Merging data into a long format suitable for seaborn
category_avg_metrics_long = category_avg_metrics.melt(id_vars=['Categories'], var_name='Metric', value_name='Value')

# Creating a pivot table
pivot_table = category_avg_metrics_long.pivot_table(index='Metric', columns='Categories', values='Value')

# Creating a heatmap with seaborn
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(data=pivot_table, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
# plt.title('Average Metrics by Categories (Seaborn Heatmap)')  # Renaming plt.title
# plt.xlabel('Categories')
# plt.ylabel('Metric')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Displaying the plot
plt.show()


# In[77]:


# Pivot table to sum metrics for each streamer
streamer_metrics = df[['Username', 'Categories', 'Country', 'Suscribers', 'Visits', 'Likes', 'Comments']]

# Calculating the average metrics
metrics_avg = streamer_metrics[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Identifying streamers with above-average performance
above_avg_streamers = streamer_metrics[
    (streamer_metrics['Suscribers'] > metrics_avg['Suscribers']) &
    (streamer_metrics['Visits'] > metrics_avg['Visits']) &
    (streamer_metrics['Likes'] > metrics_avg['Likes']) &
    (streamer_metrics['Comments'] > metrics_avg['Comments'])
]


# In[78]:


print("The above average streamers are:")
print(above_avg_streamers)


# In[79]:


streamers = len(above_avg_streamers)
print("There are", streamers, "above-average YouTube streamers")


# In[81]:


# Sorting above-average streamers by subscribers
top_content_creators_subscribers = above_avg_streamers.sort_values(by='Suscribers', ascending=False).head(10)[['Username', 'Suscribers']]

# Printing the top performing content creators by subscribers
print("The top performing content creators by subscribers are:")
print(top_content_creators_subscribers)


# In[82]:


# Sorting above-average streamers by visits
top_content_creators_visits = above_avg_streamers.sort_values(by='Visits', ascending=False).head(10)[['Username', 'Visits']]

# Printing the top performing content creators by visits
print("The top performing content creators by visits are:")
print(top_content_creators_visits)


# In[83]:


# Sorting above-average streamers by likes
top_content_creators_likes = above_avg_streamers.sort_values(by='Likes', ascending=False).head(10)[['Username', 'Likes']]

# Printing the top performing content creators by likes
print("The top performing content creators by likes are:")
print(top_content_creators_likes)


# In[84]:


# Sorting above-average streamers by comments
top_content_creators_comments = above_avg_streamers.sort_values(by='Comments', ascending=False).head(10)[['Username', 'Comments']]

# Printing the top performing content creators by comments
print("The top performing content creators by comments are:")
print(top_content_creators_comments)


# In[ ]:




