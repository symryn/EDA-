#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all required libraries for the EDA project
import pandas as pd
import numpy as np
import statsmodels.api as sm
from bokeh.io import output_notebook
output_notebook()
import seaborn as sb 
import matplotlib.pyplot as plt
from scipy import stats
from bokeh.io import show
import sys
import researchpy as rp
from bokeh.plotting import figure


# In[2]:


# Creates a DataFrame to load the data in the file and preview it.

wage_df = pd.read_csv('wage.csv')
wage_df.head()


# In[47]:


# Encode marital status as a categorical variable 

#category = pd.cut(wages_df.married, bins=[0,1], labels=['single', 'married'])
#wages_df['married'] = category.values


# The dataset encodes marital status as <b>0 - Single</b> and <b>1 - married</b> hence we have to initially Encode the marital status as a categorical variable to enable us display accurate descriptive statistics.

# <h3> 1. Displaying Descriptive statistics on the dataset
# 

# In[4]:


# Displaying descriptive stats on the dataset

wage_df.describe(include='all')


# In[5]:


# Display relevant information on the dataset.

wage_df.info()


# <h3> 2. Checking if any records in the data have any missing values; handling the missing data as appropriate.

# In[6]:


# Returns number of empty/null values in the dataset.

print(wage_df.isnull().sum())


# In[7]:


# Replace empty/null numerical variables with the mean.

data = wage_df
data['hourly_wage'] = data['hourly_wage'].fillna(data.hourly_wage.mean())
data['years_in_education'] = data['years_in_education'].fillna(data.years_in_education.mean())
data['years_in_employment'] = data['years_in_employment'].fillna(data.years_in_employment.mean())
data['num_dependents'] = data['num_dependents'].fillna(data.num_dependents.mean())


# In[8]:


# Printing results
print(wage_df.isnull().sum())


# In[9]:


# Replace missing gender values with the modal value

gender_mode = data.gender.mode()[0]
data.gender.fillna(gender_mode, inplace=True)
race_mode = data.race.mode()[0]
data.race.fillna(race_mode, inplace=True)


# In[10]:


# Replacing the empty cells with the modal value
data.married.fillna(data.married.mode()[0], inplace=True)


# In[12]:



print(wage_df.isnull().sum())


# In[13]:


# displaying the information about the datase
data.info()


# In[14]:


# Displayin gthe statistical information of the dataset
data.describe(include='all')


# In[15]:


# counting the number of white and non white race from the data
data.race.value_counts(normalize=True).plot.barh()



# The graph above is a frequency distribution graph that shows the distribution of white people and non-white people in the data set. From the graph above, the number of white people in the data set is significantly more than the number of non-white people.

# <h3> 3  Build a graph visualizing the distribution of one or more individual continuous variables of the dataset

# In[16]:


# Displaying the histogram for the hourly wage, years in education and number of dependents
data.hist(column=['hourly_wage', 'years_in_education', 'num_dependents'], bins=8, figsize=(10,6))


# The three graphs above are histograms showing the distributions of the hourly wage, years in education and number of dependents. For the hourly wage distribution, it is seen that a large proportion of people in the data set earn 10 pounds an hour or less. For the years in education distribution, most of the people in the dataset have been in education for 12 years or more. For the number of dependents distribution, most of the demographic, have between 0 and 2 dependents.

# In[17]:


# Plotting a scatter graph of hourly wage to years in employment
plt.scatter(data.hourly_wage, data.years_in_employment)
plt.show()


# From the scatter plot above, we see that most of the people earn at most 10 pounds an hour and have been in employment for less than 10 years. The data is aggregated majorly towards the bottom left region of the graph.

# <h3>  4-Building a relationship between a pair of continious variables and determing the correlation between them

# In[18]:


# Finding the correlation between years in employment and hourly wage
data['years_in_employment'].corr(data['hourly_wage'], method='pearson')


# The correlation between years in employment and hourly wage is just 0.34097528485222717. This implies that both variables are not closely related and little to no information about one can be gotten from the other.

# In[19]:


# Displaying the count of distribution of white and non-white people
data.race.value_counts()


# In[20]:


# Plotting a pie chart to show the ratio of white to non-white
data.race.value_counts().plot.pie(autopct='%1.1f%%')


# The pie chart above gives a visual representation of what is seen in the count of the racial distribution that the proportion of white people to non-white people is 89.7:10.3.

# <h3> 5 Display unique values of a categorical variable

# In[63]:


# Displaying the count of distribution of male to female
data.gender.value_counts()


# In[62]:


# Plotting a pie chart to show the ratio of male to female
data.gender.value_counts().plot.pie(autopct='%1.1f%%')


# The pie chart shows that the gender distribution of male to female is almost equal as the ratio of male to female is 52.6 to 47.4.

# In[23]:


# Displaying the count of distribution of married to single
data.married.value_counts()


# In[24]:


# Plotting a pie chart to show the ratio of married to single
data.married.value_counts().plot.pie(autopct='%1.1f%%')


# The pie chart above shows that there are more married people than single people in the data set as the ratio of the married to single people is 61.1 to 38.9.

# <h3> 6 Build a contingency table of two potentially related categorical variables. Conduct a statistical test of the independence between them.

# In[25]:


#  Display a contingency table showing the relationshiop between marital status
# and gender.

contingency_table = pd.crosstab(data['married'], data['gender'])
contingency_table


# In[26]:


# Plotting a graph of the contingency table
contingency_table.plot(kind='bar', stacked=True, rot=0)


# In[27]:


# Running a chi-squared test on the marital status and gender data
chi2,p_val,dof,expected = stats.chi2_contingency(contingency_table)
print('P value is:', p_val)


# After conducting a chi squared test on the marital status and gender, the p-value of 0.001700859727490584 was observed. This is significantly less than the usual significance level of 0.05. Therefore, **we reject the null hypothesis that there is no dependence between the gender and marital status**. 
# 
# We therefore found evidence that a higher proportion of males are more likely to be married while the females were more likely to be single.

# <h3> 7 Retrieve one or more subset of rows based on two or more criteria and present descriptive statistics on the subset(s).

# In[ ]:


# Display a subset based on speified criteria. 

data1 = data[(data.years_in_employment > 6) & (data.hourly_wage > 6)]
data1


# There are 68 people in the data set whose experience is above the average years of experience whose hourly wage also exceed the average.

# In[29]:


# Displaying descriptive statistics on the subset.

data1.describe()


# In[30]:


# Plotting a pie chart of the distribution of the genders in the subset
gender_count = data1['gender'].value_counts()
gender_df = pd.DataFrame({'gender': gender_count}, 
                        index = ['male', 'female'])
gender_df.plot.pie(y='gender', figsize=(8,8), autopct='%1.1f%%')
print(gender_count)


# In[31]:


# out of 68 people 


# In[32]:


# Plotting a pie chart of the distribution of the races in the subset
race_count = data1['race'].value_counts()
race_df = pd.DataFrame({'race': race_count}, 
                        index = ['white', 'nonwhite'])
race_df.plot.pie(y='race', figsize=(8,8), autopct='%1.1f%%')
print(race_count)


# Of the 68 people whose hourly wage rate and years in employment exceeded the average, the gender distribution was 85.3% (or 58 people) being male and 14.7% (or 13 people) being female. 

# In[34]:


# Creating a subset
data2 = data[(data.married==0) & (data.gender=='male') 
                 & (data.years_in_education > 13)]
data2.head()


# In[35]:


# displaying the subset 
data2.describe()


#  <h3>8. Conducting a statistical test of the significance of the difference between the means of the two subsets of the data</h3>

# In[36]:


# Conducts a test to check the significance between the difference of the average wage 
# for white and Nonwhite people.
summary, results = rp.ttest(group1= data["hourly_wage"][data['race']=='white'],group1_name ="white",
        group2= data['hourly_wage'][data['race']=='nonwhite'], group2_name ="nonwhite")
print(summary)


# In[37]:


# printing the results
print(results)


# The p-value of 0.5657 is higher than the confidence interval p-value of 0.05, therefore the null hypothesis is rejected. We can then say that the means of the two subsets of the data are not the same.

# <h3> 9 Create one or more tables that group the data by a certain categorical variable and displays summarized information for each group (e.g. the mean or sum within the group).</h3>

# In[38]:


# Group the data by gender.

gender_group = data.groupby("gender")
gender_group.size()


# In[39]:


# Aggregate and summarize info of each variable for the grouped data into a table.

display_grouped = gender_group.agg({"years_in_education":["count","min","max","mean","median","skew"],
                  "years_in_employment":["count","min","max","mean","median","skew"],
                     "hourly_wage":["count","min","max","mean","median","skew"],
                     "num_dependents":["count","min","max","mean","median","skew"]
                    })
display_grouped


# <h3>10. Model Fitting</h3>

# In[41]:


# Encode Categorical Variables as numerical

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
data['gender'] = number.fit_transform(data['gender'].astype('object'))
data['race'] = number.fit_transform(data['race'].astype('object'))
data['married'] = number.fit_transform(data['married'].astype('float'))
data.head()


# In[43]:


# Fits a model to determine hourly wage using OLS method.

hourly_wage_model = sm.OLS.from_formula(
    "hourly_wage ~ married + years_in_education + years_in_employment + num_dependents + gender + race", data = data).fit()
hourly_wage_model.summary()


# 
# 
# (1) **Coefficients on the variables**. The estimated coefficients are specified in the second table. Our model is thus described by the line: 
# 
# $Hourly Wage = -2.6107 + 0.7030*married + 0.5214*YearsInEducation + 0.0957*NumDependennts + 0.1531*YearsInEmployment + 1.6732*gender - 0.0665*race + e$.
# 
# Considering the signs on the coefficients it is hard to say categorically which of the variables would have the most effect on the hourly wage. 
# 
# (2) **Significance of the variables**. The p-values on the coefficients indicate that two of the variables which were the Number of dependents and race were insignificant to the model outcome. However, the remaining variables seemed significant with p_values of 0.000 which were less than the significance level of 0.05.
# 
# (3) **Quality of the model**. The inital $R^2$ and the adjusted $R^2$ values were 0.35 and 0.342 respectively. The two insignificant variables (Number of dependents and race) were removed from the model and the model yielded $R^2$ and adjusted $R^2$ values of 0.349 and 0.344 respectively which seemed to further reduce the quality of the model. Hence, we can conclude that there were most likely other factors affecting the hourly wage that our model didn't take into account.

# # Checking the assumptions of multicollinearity, normality and zero mean of residuals
# 
# As with the simple regression model, we can plot the standardized residuals and their histogram to confirm the validity of the assumptions of normality of the distribution of residuals and of the zero mean of residuals are valid with this model.

# In[44]:


fig = figure(height=400, width=400)

# the x axis is the fitted values
# the y axis is the standardized residuals
st_resids = hourly_wage_model.get_influence().resid_studentized_internal
fig.circle(hourly_wage_model.fittedvalues, st_resids)

show(fig)


# In[45]:


# Plot an histogram of 10 bins to show the standardized residual values.

hist, edges = np.histogram(st_resids, bins=10)
res_hist = figure(height=400, width=400)
res_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
show(res_hist)


# 
# 
# The scatterplot and the histogram suggest the residuals are not equally distributed around 0 and thus not normally distributed. The results of the Jarque-Bera test on the residuals (881.856 - in the third table of the summary) also indicate that the errors are not distributed normally as the value was very high. Also, the p-value equals 3.22e-192 therefore we reject the null hypothesis of normal distribution.
# 
# As with the simple linear regression model, we do not test for independence of the errors and for homoskedasticity, as these assumptions are likely to be violated only if the observations are ordered along a temporal dimension.
# 
# Thus, the removal of the insignificant variables did not improve the quality of the model, and thus the assumptions of the classical linear regression method did not hold with this particular model.
# 
# The method could possibly be improved by addition of other variables/factors we haven't considered due to inavailability in the dataset provided.

# In[68]:



get_ipython().run_cell_magic('javascript', '', 'var nb = IPython.notebook;\nvar kernel = IPython.notebook.kernel;\nvar command = "NOTEBOOK_FULL_PATH = \'" + nb.notebook_path + "\'";\nkernel.execute(command);')


# In[69]:


import io
from nbformat import read, NO_CONVERT

with io.open(NOTEBOOK_FULL_PATH.split("/")[-1], 'r', encoding='utf-8') as f:
    nb = read(f, NO_CONVERT)

word_count = 0
for cell in nb.cells:
    if cell.cell_type == "markdown":
        word_count += len(cell['source'].replace('#', '').lstrip().split(' '))
print(f"Word count: {word_count}")


# In[ ]:




