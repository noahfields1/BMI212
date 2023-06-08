import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# Read the data from the CSV file
data = pd.read_csv('PRS_values_with_race_sex.csv')

def plot_histogram_by_race(data, t2d_status):
    race_categories = [ 'White', 'Black or African American','Asian']
    
    # Filter data for selected race categories and T2D_Status
    filtered_data = data[(data['Race'].isin(race_categories)) & (data['T2D_Status'] == t2d_status)]
    
    # Plot histogram of PRS for each race category
    plt.figure(figsize=(8, 6))
    for race in race_categories:
        race_data = filtered_data[filtered_data['Race'] == race]
        plt.hist(race_data['PRS'], bins=20, alpha=0.5, label=race)
    
    plt.xlabel('PRS')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of PRS by Race (T2D_Status = {t2d_status})')
    plt.legend()
    plt.savefig(f'pictures/histogram_race_t2d_{t2d_status}.png')
    plt.show()

def plot_histogram_by_sex(data, t2d_status):
    sex_categories = ['Male', 'Female']
    
    # Filter data for selected sex categories and T2D_Status
    filtered_data = data[(data['Sex'].isin(sex_categories)) & (data['T2D_Status'] == t2d_status)]
    
    # Plot histogram of PRS for each sex category
    plt.figure(figsize=(8, 6))
    for sex in sex_categories:
        sex_data = filtered_data[filtered_data['Sex'] == sex]
        plt.hist(sex_data['PRS'], bins=20, alpha=0.5, label=sex)
    
    plt.xlabel('PRS')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of PRS by Sex (T2D_Status = {t2d_status})')
    plt.legend()
    plt.savefig(f'pictures/histogram_sex_t2d_{t2d_status}.png')
    plt.show()

def perform_ranksums_test(data, group1, group2, t2d_status):
    group1_data = data[(data['Race'] == group1) & (data['T2D_Status'] == t2d_status)]['PRS']
    group2_data = data[(data['Race'] == group2) & (data['T2D_Status'] == t2d_status)]['PRS']
    
    # Perform Wilcoxon rank sum test
    stat, p_value = ranksums(group1_data, group2_data)
    
    print(f"Ranksums Test Results: {group1} vs {group2} (T2D_Status = {t2d_status})")
    print(f"Statistic: {stat}")
    print(f"P-value: {p_value}\n")


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Given a df and two PRS_Quartile Values: Find odds ratios
def calculate_odds_ratios(subset_df,low,high):
    a = len(subset_df[(subset_df['PRS Quartile'] == high) & (subset_df['T2D_Status'] == 1)])
    b = len(subset_df[(subset_df['PRS Quartile'] == high) & (subset_df['T2D_Status'] == 0)])

    c = len(subset_df[(subset_df['PRS Quartile'] == low) & (subset_df['T2D_Status'] == 1)])
    d = len(subset_df[(subset_df['PRS Quartile'] == low) & (subset_df['T2D_Status'] == 0)])
    OR = (a*d)/(b*c)
    return OR
def plot_odds_ratios():
    # Read the CSV file into a pandas DataFrame
    df = data
    sections = 5
    # Create quartiles based on the PRS values
    df['PRS Quartile'] = pd.qcut(df['PRS'], q=sections, labels=False, duplicates='drop')

    min_values = df.groupby('PRS Quartile')['PRS'].min()
    print(min_values)
    


    # Calculate the odds ratios for T2D status based on the PRS quartiles
    overall_odds_ratios = []
    quartiles = []
    for i in range(sections):
        OR = calculate_odds_ratios(df,0,i)

        quartiles.append(i+1)
        overall_odds_ratios.append(OR)


    # Plot the odds ratios for the overall population and stratified by race
    plt.figure(figsize=(10, 8))
    sns.pointplot(x=quartiles, y=overall_odds_ratios, color='black', label='Overall')

    plt.xlabel('PRS Quintile')
    plt.ylabel('Odds Ratio')
    plt.title('Odds Ratios by PRS Quintiles')
    plt.show()






def plot_histograms_and_Wilcoxon():
    # Call the functions for T2D_Status = 1
    plot_histogram_by_race(data, 1)
    plot_histogram_by_sex(data, 1)

    perform_ranksums_test(data, 'White', 'Asian', 1)
    perform_ranksums_test(data, 'Asian', 'Black or African American', 1)
    perform_ranksums_test(data, 'Black or African American', 'White', 1)
    perform_ranksums_test(data, 'Male', 'Female', 1)

    # Call the functions for T2D_Status = 1
    plot_histogram_by_race(data, 0)
    plot_histogram_by_sex(data, 0)

    perform_ranksums_test(data, 'White', 'Asian', 0)
    perform_ranksums_test(data, 'Asian', 'Black or African American', 0)
    perform_ranksums_test(data, 'Black or African American', 'White', 0)
    perform_ranksums_test(data, 'Male', 'Female', 0)

if __name__ == "__main__":
    plot_odds_ratios()
    #plot_histograms_and_Wilcoxon()