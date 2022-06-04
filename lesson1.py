#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## 1.1. Data Parsing

# In[33]:


df=pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')


# In[34]:


df.head()


# In[35]:


df['date']=pd.to_datetime(df['date'])


# In[36]:


df.info()


# Consider the number of empty values

# In[37]:


pd.set_option('display.max_rows',None)
df.isnull().sum()


# In[38]:


pd.set_option('display.max_rows',10)


# In[39]:


df.shape


# All data was included when parsing from the repository. The dimension of the dataset is `183348 rows` and `67 columns`
# 
# We will supplement the data set with new information so that, if necessary, the accuracy of the model during training is greater. Also, new data can be useful in order to include them in data analysis, from which some dependencies can be extracted, if they are present. 
# 
# As new data, let's take the average number of deaths and infected per region.

# In[40]:


df[['location', 'new_cases', 'new_deaths']]=df[['location', 'new_cases', 'new_deaths']].fillna(0)


# In[41]:


grouped_cases=df[['location', 
                  'new_cases', 
                  'new_deaths']].groupby(by="location").mean().rename(columns={'new_cases':'mean_new_cases', 
                                                                               'new_deaths':'mean_new_deaths'})


# In[42]:


grouped_cases


# In[43]:


df=df.merge(grouped_cases, on='location')


# In[44]:


df.head()


# New data has been generated

# ## 1.2. Data preprocessing and highlighting of significant attributes
# Fill in all empty values with zeros

# In[45]:


df=df.fillna(0)


# ### Determining the most significant attributes
# To find the most significant attributes, let's build the Pearson correlation on the heat map

# In[46]:


corr=df.corr()
plt.figure(figsize=(70, 70))

heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20)


# As we can see above, quite a lot of features have a high correlation coefficient, but the most significant attributes are: `total_casem, new_case, new_cases_smoothed, total_deaths, new_deaths и new_deaths_smoothed`

# ## 1.3. Description of the data set structure

# In[47]:


df.info()


# `total_cases` - Total confirmed cases of COVID-19. The counts may include probable cases reported.
# 
# `new_cases` - New confirmed cases of COVID-19. The counts may include probable cases reported. In rare cases, when our source reports a negative daily change due to data adjustments, we set the NAME value for this metric.
# 
# `new_cases_smoothed` - New confirmed cases of COVID-19 (7-day period smoothed out). The counts may include probable cases reported.
# 
# `total_cases_per_million` - The total number of confirmed COVID-19 cases per 1,000,000 people. The counts may include probable cases reported.
# 
# `new_cases_per_million` - New confirmed cases of COVID-19 per 1,000,000 people. The counts may include probable cases reported.
# 
# `new_cases_smoothed_per_million` - New confirmed COVID-19 cases (smoothed over 7 days) per 1,000,000 people. The counts may include probable cases reported.
# 
# `total_deaths` - The total number of deaths related to COVID-19. It has been reported that the counts may include probable deaths.
# 
# `new_deaths` - New deaths related to COVID-19. It has been reported that the counts may include probable deaths. In rare cases, when our source reports a negative daily change due to data adjustments, we set the NAME value for this metric.
# 
# `new_deaths_smoothed` - New deaths related to COVID-19 (7-day period smoothed out). It has been reported that the counts may include probable deaths.
# 
# `total_deaths_per_million` - The total number of COVID-19-related deaths per 1,000,000 people. It has been reported that the counts may include probable deaths.
# 
# `new_deaths_per_million` - New COVID-19-related deaths per 1,000,000 people. It has been reported that the counts may include probable deaths.
# 
# `new_deaths_smoothed_per_million` - New deaths related to COVID-19 (smoothed over 7 days) per 1,000,000 people. It has been reported that the counts may include probable deaths.
# 
# `excess_mortality` is the percentage difference between the registered number of weekly or monthly deaths in 2020-2021 and the projected number of deaths for the same period based on previous years. For more information, see https://github.com/owid/covid-19-data/tree/master/public/data/excess_mortality 
# 
# `excess_mortality_cumulative` is the percentage difference between the cumulative number of deaths since January 1, 2020 and the cumulative projected number of deaths for the same period based on previous years. For more information, see https://github.com/owid/covid-19-data/tree/master/public/data/excess_mortality
# 
# `excess_mortality_cumulative_absolute` is the cumulative difference between the registered number of deaths since January 1, 2020 and the projected number of deaths for the same period based on previous years. For more information, see https://github.com/owid/covid-19-data/tree/master/public/data/excess_mortality excess
# 
# `excess_mortality_cumulative_per_million` is the cumulative difference between the registered number of deaths since January 1, 2020 and the projected number of deaths for the same period based on previous years per million people. For more information, see https://github.com/owid/covid-19-data/tree/master/public/data/excess_mortality
# 
# `icu_patients` - The number of patients with COVID-19 in intensive care units (ICU) on a given day
# 
# `icu_patients_per_million` - The number of patients with COVID-19 in intensive care units (ICU) on a given day per 1,000,000 people
# 
# `hosp_patients` - The number of COVID-19 patients in the hospital on a given day
# 
# `hosp_patients_per_million` - The number of COVID-19 patients in the hospital on a given day per 1,000,000 people
# 
# `weekly_icu_admissions` - The number of COVID-19 patients admitted to the intensive care unit (ICU) for the first time in a given week
# 
# `weekly_icu_admissions_per_million` - The number of COVID-19 patients admitted to the intensive care unit (ICU) for the first time in a given week per 1,000,000 people
# 
# `weekly_hosp_admissions` - The number of COVID-19 patients admitted to hospitals for the first time in a given week
# 
# `weekly_hosp_admissions_per_million` - The number of COVID-19 patients admitted to hospitals for the first time in a given week per 1,000,000 people
# 
# `stringency_index` - Government Response Severity Index: a composite indicator based on 9 response indicators, including school closures, job closures and travel bans, scaled to a value from 0 to 100 (100 = the strictest response)
# 
# `reproduction_rate` - Real-time estimate of the effective reproduction rate (R) of COVID-19. https://github.com/crondonm/TrackingR/tree/main/Estimates-Database
# 
# `total_tests` - Total number of tests for COVID-19
# 
# `new_tests` - New tests for COVID-19 (calculated only for consecutive days)
# 
# `total_tests_per_thousand` - Total number of COVID-19 tests per 1000 people
# 
# `new_tests_per_thousand` - New COVID-19 tests per 1000 people
# 
# `new_tests_smoothed` - New tests for COVID-19 (7-day smoothed). For countries that do not report testing data on a daily basis, we assume that testing changed the same daily for any periods during which data was not reported. This gives a full range of daily indicators, which are then averaged over a rolling 7-day window.
# 
# `new_tests_smoothed_per_thousand` - New COVID-19 tests (7-day smoothed) for 1000 people
# 
# `positive_rate` - The percentage of positive tests for COVID-19, given as a 7-day moving average (this is the inverse of tests_per_case)
# 
# `tests_per_case` - The tests performed for each new confirmed case of COVID-19 are given as a 7-day moving average (this is the inverse of positive_rate)
# 
# `tests_units` - Units of measurement used by the location to represent its testing data
# 
# `total_vaccinations` - The total number of doses of vaccination against COVID-19 administered to vaccinated people
# 
# `people_vaccinated` - The total number of people who received at least one dose of the vaccine
# 
# `people_fully_vaccinated` - The total number of people who received all the doses prescribed by the initial vaccination protocol 
# 
# `total_boosters` - The total number of administered booster doses of vaccination against COVID-19 (doses administered in excess of the amount prescribed by the vaccination protocol)
# 
# `new_vaccinations` - Introduced new doses of vaccination against COVID-19 (calculated only for consecutive days)
# 
# `new_vaccinations_smoothed` - New doses of vaccination against COVID-19 have been introduced (the 7-day period has been smoothed out). For countries that do not report vaccination data on a daily basis, we assume that vaccination varied the same daily for any periods during which data were not reported. This gives a full range of daily indicators, which are then averaged over a rolling 7-day window.
# 
# `total_vaccinations_per_hundred` - The total number of doses of vaccination against COVID-19 administered per 100 people in the general population
# 
# `people_vaccinated_per_hundred` - The total number of people who received at least one dose of the vaccine per 100 people in the total population
# 
# `people_fully_vaccinated_per_hundred` - The total number of people who received all doses prescribed by the primary vaccination protocol per 100 people in the total population
# 
# `total_boosters_per_hundred` - The total number of booster doses of vaccination against COVID-19 administered per 100 people in the total population
# 
# `new_vaccinations_smoothed_per_million` - New doses of vaccination against COVID-19, introduced (smoothed for 7 days) per 1,000,000 people of the total population
# 
# `new_people_vaccinated_smoothed` - The daily number of people receiving their first dose of the vaccine (smoothed by 7 days)
# 
# `new_people_vaccinated_smoothed_per_hundred` - The daily number of people receiving the first dose of the vaccine (7-day smoothed) per 100 people in the total population
# 
# `iso_code` - ISO 3166-1 alpha-3 – three-letter country codes
# 
# `continent` - The continent of the geographical location
# 
# `location` - Geographical location
# 
# `date` - Date of observation
# 
# `population` - Population size (latest available values). https://github.com/owid/covid-19-data/blob/master/scripts/input/un/population_latest.csv for a complete list of sources
# 
# `population_density` - The number of people divided by the land area measured in square kilometers, the most recent year available
# 
# `medier_age` - Average age of the population, UN forecast for 2020
# 
# `aged_65_older` - Percentage of the population aged 65 and older, the last available year
# 
# `aged_70_older` - Percentage of the population aged 70 and older in 2015 
# 
# `gdp_per_capita` - Gross domestic Product at purchasing power parity (constant international dollar 2011), the last available year
# 
# `extreme_poverty` - The proportion of the population living in extreme poverty, the last year available since 2010 
# 
# `cardiovasc_death_rate` - Mortality rate from cardiovascular diseases in 2017 (annual number of deaths per 100,000 people)
# `diabetes_prevalence` - Prevalence of diabetes (% of the population aged 20 to 79 years) in 2017
# 
# `female_smokers` - Percentage of women smokers available over the last year
# 
# `male_smokers` - Percentage of male smokers available in the last year
# 
# `handwashing_facilities` - The proportion of the population with basic hand washing facilities in the premises, over the past year the number of available
# 
# `hospital_beds_per_thousand` - Hospital beds for 1000 people, last available year since 2010
# 
# `life_expectancy` - Life expectancy at birth in 2019
# 
# 

# ### Empty values
# 
# Empty values were preprocessed earlier, after preprocessing they are no longer left

# In[48]:


pd.set_option('display.max_rows',None)
df.isnull().sum()


# In[49]:


pd.set_option('display.max_rows',10)


# ### Data distribution density
# Let's form density graphs for each feature

# In[50]:


df.head()


# In[51]:


plt.figure(figsize=(10, 5))
sns.kdeplot(df['iso_code'].value_counts())
plt.title('Distribution iso_code')
plt.xlabel('Values')
plt.ylabel('Distribution')
plt.show()


# In[52]:


plt.figure(figsize=(10, 5))
sns.kdeplot(df['continent'].value_counts())
plt.title('Distribution continent')
plt.xlabel('Values')
plt.ylabel('Distribution')
plt.show()


# In[53]:


plt.figure(figsize=(10, 5))
sns.kdeplot(df['location'].value_counts())
plt.title('Distribution location')
plt.xlabel('Values')
plt.ylabel('Distribution')
plt.show()


# In[54]:


def plot(column):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[column])
    plt.title('Distribution '+column)
    plt.xlabel('Values')
    plt.ylabel('Distribution')
    plt.show()


# In[55]:


for column in df[:100].select_dtypes(exclude=['object']).columns:
    plot(column)


# ## 1.4. Formation of additional attributes
# Let's form a new attribute according to the formula given in the task:
# 
# *`Rt = number of registered diseases in the last 4 days / number of registered diseases in the previous 4 days`*

# In[56]:


df['Rt']=None
data=pd.DataFrame()
for country in df['location'].value_counts().keys():
    r=df[df['location']==country].copy()
    da=pd.DataFrame()
    for i in range(0, len(r), 8):
        tida=pd.DataFrame()
        su=r['new_cases'].tail(8).tail(4).sum()/r['new_cases'].tail(8).head(4).sum()
        tida=r.tail(8)
        tida['Rt']=su
        r.drop(r.tail(8).index,inplace=True)
        da=da.append(tida)
    data=data.append(da)


# In[57]:


data=data.fillna(0)


# In[58]:


data.reset_index(drop=True, inplace=True)
df=data


# In[59]:


df.head()


# **The result of the algorithm above demonstrates the formation of an additional attribute Rt - the coefficient of infection spread**

# ### Analysis of the possibility of determining changes in the epidemiological
# Let's analyze the data obtained for several countries

# In[60]:


d=pd.DataFrame({'India': [list(df[df['location']=='India']['Rt'])[0]], 
                'Mexico':[list(df[df['location']=='Mexico']['Rt'])[0]], 
                'France': [list(df[df['location']=='France']['Rt'])[0]], 
                'Taiwan':[list(df[df['location']=='Taiwan']['Rt'])[0]], 
                'United States':[list(df[df['location']=='United States']['Rt'])[0]], 
                'Japan':[list(df[df['location']=='Japan']['Rt'])[0]], 
                'Canada':[list(df[df['location']=='Canada']['Rt'])[0]], 
                'Singapore':[list(df[df['location']=='Singapore']['Rt'])[0]],}).T


# **We will output the current Rt values for each country**

# In[61]:


plt.rcParams.update({'font.size': 15,})
plt.figure(figsize=(15, 8))
plots = sns.barplot(x=d.index, y=d[0], data=df)

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.title('Analysis of the epidemiological situation')
plt.ylabel('Rt - value')
plt.xlabel('Contry')
plt.show()


# From the results obtained, we can say that the maximum Rt value of the proposed countries is Singapore`(2.27)`. The minimum is Mexico `(0.56)`. India currently has an Rt value of `0.80`

# ## Conclusion
# 
# * 1.1 Data parsing - The data set is loaded from the repository, and some attributes are supplemented
# * 1.2 Data preprocessing and allocation of significant attributes - The data set is processed from empty values, and the most "important" attributes are determined by the Pearson correlation
# * 1.3 Description of the data set structure - for each attribute, a description and density of the data distribution are presented
# * 1.4 Formation of additional attributes - An additional Rt attribute has been formed, which determines the coefficient of infection spread over the last 8 days

# In[62]:


df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

