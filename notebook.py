#!/usr/bin/env python
# coding: utf-8

import requests 
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = "https://www.worldometers.info/coronavirus/"
data = requests.get(url)

dataset_folder = "dataset/"
graphs_folder = "graphs/"
incidences_folder = "incidences/"
deaths_folder = "deaths/"

parsed = BeautifulSoup(data.content, "html.parser")
raw_table = parsed.find_all('table', id = "main_table_countries_yesterday")[0]
table_df = pd.read_html(str(raw_table), displayed_only = False)[0]
table_df = table_df[8:]

from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
table_df.to_csv(dataset_folder+'worldometer-'+ today +'.csv')

def obtain_data(file_name):    
    # the csv file was converted in panda dataframe
    df_table = pd.read_csv(file_name)
    # only relevant data from the csv was obtained to filter the noise
    relevant_data = df_table[['Country,Other', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered', 'ActiveCases', 'Population', 'Continent']]
    # data not available was replaced by 0
    relevant_data = relevant_data.fillna(0)
    # Inorder to obtain data based on the country, the data was remodelled to use country as the row index
    country_names = relevant_data['Country,Other']
    index_country_mapping = {}
    for i in range(len(country_names)):
        index_country_mapping[i] = country_names[i]
    relevant_data = relevant_data.rename(index = index_country_mapping)
    return relevant_data

def get_five_countries_data(data):
    # For this project, five countries are chosen
    five_countries_data = data.loc[['USA','China','India','UK','Germany']]
    return five_countries_data

eleven_days_data_files = ["worldometer-2022-11-24.csv","worldometer-2022-11-25.csv","worldometer-2022-11-26.csv",                          "worldometer-2022-11-27.csv","worldometer-2022-11-28.csv","worldometer-2022-11-29.csv",                          "worldometer-2022-11-30.csv","worldometer-2022-12-01.csv","worldometer-2022-12-02.csv",                          "worldometer-2022-12-03.csv"]


#NewCases of COVID-19 for each day for selected selected countries 
def barplot_new_cases(data, date):
    new_cases_five_countries = sns.barplot(x=data.index, y=data['NewCases']).set_title('Covid-19 New Cases by Countries on ' + date)
    return new_cases_five_countries 

def barplot_new_deaths(data, date):
    new_deaths_five_countries = sns.barplot(x=five_countries_data.index, y=five_countries_data['NewDeaths']).set_title('Covid-19 Deaths by Countries on '+ date)
    return new_deaths_five_countries

for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    new_cases_plot = barplot_new_cases(five_countries_data, date)
    barplot_new_cases_fig = new_cases_plot.get_figure()
    barplot_new_cases_fig.savefig(graphs_folder+incidences_folder+'new_cases_plot_' + date + '.jpg')


for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    new_deaths_plot = barplot_new_deaths(five_countries_data, date)
    barplot_death_cases_fig = new_deaths_plot.get_figure()
    barplot_death_cases_fig.savefig(graphs_folder+deaths_folder+'new_deaths_plot_' + date + '.jpg') 

data_for_pie_chart = obtain_data(dataset_folder+'worldometer-2022-11-24.csv')
pie_dict = {}
for index, row in data_for_pie_chart.iterrows():
    active_cases = int(row[6])
    continent = row[8]
    if continent == 0 or continent == "All":
        continue
    pie_dict[continent]= pie_dict.get(continent,0) + active_cases

data = []
labels = []
for key,value in pie_dict.items():
    data.append(value)
    labels.append(key)
    
plt.clf()
pie_chart_colors = sns.color_palette('pastel')[0:5]
plt.pie(data, labels=labels, colors=pie_chart_colors, autopct='%.0f%%')
plt.title('Active Cases by Continent')
plt.savefig(graphs_folder+'active_cases_by_continent_pie.jpg')
plt.show()


usa_data_by_date = []
for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    usa_data = five_countries_data.loc[['USA']]
    dict_data = [date[5:], usa_data['NewCases'][0]]
    usa_data_by_date.append(dict_data)

usa_data_for_lineplot = pd.DataFrame(usa_data_by_date, columns=["Date", "NewCases"])
plt.clf()
line_plt_usa = sns.lineplot(data=usa_data_for_lineplot, x="Date", y="NewCases").set(title="Trends in incidence of COVID-19 New Cases in USA")
plt.savefig(graphs_folder+incidences_folder+'usa_new_cases_trend.jpg')


china_data_by_date = []
for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    china_data = five_countries_data.loc[['China']]
    dict_data = [date[5:], china_data['NewCases'][0]]
    china_data_by_date.append(dict_data)

china_data_for_lineplot = pd.DataFrame(china_data_by_date, columns=["Date", "NewCases"])
plt.clf()
sns.lineplot(data=china_data_for_lineplot, x="Date", y="NewCases").set(title="Trends in incidence of COVID-19 New Cases in China")
plt.savefig(graphs_folder+incidences_folder+'china_new_cases_trend.jpg')


usa_data_by_date = []
for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    usa_data = five_countries_data.loc[['USA']]
    dict_data = [date[5:], usa_data['NewDeaths'][0]]
    usa_data_by_date.append(dict_data)

usa_data_for_lineplot = pd.DataFrame(usa_data_by_date, columns=["Date", "NewDeaths"])
usa_data_for_lineplot
plt.clf()
sns.lineplot(data=usa_data_for_lineplot, x="Date", y="NewDeaths").set(title="Trends in Deaths of COVID-19 in USA")
plt.savefig(graphs_folder+deaths_folder+'usa_deaths_trend.jpg')


china_data_by_date = []
for filename in eleven_days_data_files:
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    five_countries_data = get_five_countries_data(data)
    china_data = five_countries_data.loc[['China']]
    dict_data = [date[5:], china_data['NewDeaths'][0]]
    china_data_by_date.append(dict_data)

china_data_for_lineplot = pd.DataFrame(china_data_by_date, columns=["Date", "NewDeaths"])
plt.clf()
sns.lineplot(data=china_data_for_lineplot, x="Date", y="NewDeaths").set(title="Trends in Deaths of COVID-19 in China")
plt.savefig(graphs_folder+deaths_folder+'china_deaths_trend.jpg')


# Performing linear regression to predict new cases

def obtain_data(file_name):
    df_table = pd.read_csv(file_name)
    # only relevant data from the csv was obtained to filter the noise
    relevant_data = df_table[['Country,Other', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered','ActiveCases', 'Population', 'Continent']]
    # data not available was replaced by 0
    relevant_data = relevant_data.fillna(0)
    # Inorder to obtain data based on the country, the data was remodelled to use country as the row index
    country_names = relevant_data['Country,Other']
    index_country_mapping = {}
    for i in range(len(country_names)):
        index_country_mapping[i] = country_names[i]
    relevant_data = relevant_data.rename(index = index_country_mapping)
    return relevant_data

eleven_days_data_files = ["worldometer-2022-11-24.csv","worldometer-2022-11-25.csv","worldometer-2022-11-26.csv",                          "worldometer-2022-11-27.csv","worldometer-2022-11-28.csv","worldometer-2022-11-29.csv",                          "worldometer-2022-11-30.csv","worldometer-2022-12-01.csv","worldometer-2022-12-02.csv",                          "worldometer-2022-12-03.csv"]


all_days = []
for day, filename in enumerate(eleven_days_data_files):
    date = filename[12:22]
    data = obtain_data(dataset_folder+filename)
    all_days.append([day, data])


def get_usa_data(date, data):
    usa_data = data.loc[['USA']]
    usa_new_cases = usa_data['NewCases'][0]
    l = [date, usa_new_cases]
    return l

all_days_usa_data = []
for one_day in all_days:
    usa = get_usa_data(one_day[0], one_day[1])
    all_days_usa_data.append(usa)

usa_data_dataframe = pd.DataFrame(all_days_usa_data, columns = ["Day", "NewCases"])

X = np.array(usa_data_dataframe[["Day"]])
y = np.array(usa_data_dataframe[["NewCases"]])
plt.clf()
plt.scatter(X,y)
plt.title("USA COVID-19 cases by day")
plt.xlabel('Days')
plt.ylabel("NewCases")
plt.savefig(graphs_folder+'scatterplot_cases_day_by_day.jpg')
plt.show()


# Linear regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)


y_predict = linear_regression.predict(X_test)


for i in range(len(y_predict)):
    print("Actual Cases are {} and predicted cases are {}".format(y_test[i], y_predict[i]))



