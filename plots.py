import matplotlib.pyplot as plt
import pandas as pd

#Maddie wrote this code
def plot_positive_negative_chart_lasso_importance(dataframe):
    snp_col = 'Feature'
    importance_col = 'Importance'

    
    # Select top 7 most negative and top 7 most positive importance values
    top_negative = dataframe.nsmallest(7, importance_col)
    top_positive = dataframe.nlargest(7, importance_col)
    combined_data = pd.concat([top_positive, top_negative])
    dataframe = combined_data
    """
    Generates a positive-negative bar chart based on SNP importance values.


    Args:
        dataframe (pd.DataFrame): The input dataframe.
        snp_col (str): The column name for SNP names.
        importance_col (str): The column name for importance values.
    """
    snp_names = dataframe[snp_col].str.replace('_interaction', '')
    importance_values = dataframe[importance_col]
    colors = ['blue' if val >= 0 else 'red' for val in importance_values]

    plt.figure(figsize=(20, 12))
    plt.barh(snp_names, importance_values, color=colors)
    plt.xlabel('Importance',fontsize=16,fontweight='bold')
    plt.ylabel('SNP (Most Negative and Most Positive)',fontsize=16,fontweight='bold')
    plt.title('Positive-Negative Bar Chart of SNP Importance', fontsize=20, fontweight='bold')
    # Increase font size of snp_names labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)

    # Save the figure to the specified directory
    save_dir = 'pictures/LASSO_Feature_Weights_Positive_Negative_Chart.png'
    plt.savefig(save_dir, bbox_inches='tight')
    #plt.show()

#Written by Maddie
def plot_PRS_Histogram():

    # Read the Excel sheet into a pandas DataFrame
    df = pd.read_excel('/Users/noah/Desktop/BMI212/PRS_values_no_interactions.xlsx')

    # Extract values from the first column
    values = df['PRS']

    # Extract classifications from the second column
    classification = df['T2D_Status']

    # Separate values based on classifications
    T2D_values = values[classification == 1]
    ctrl_values = values[classification == 0]

    # Plot histogram colored by classification
    plt.hist(T2D_values, bins=10, color='blue', alpha=0.5, label='Case')
    plt.hist(ctrl_values, bins=10, color='red', alpha=0.5, label='Control')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of PRS')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #df = pd.read_csv("LASSO_Feature_Weights.tsv", delimiter='\t')
    #print(df.head())
    #exit()
    plot_PRS_Histogram()
    #plot_positive_negative_chart_lasso_importance(df)
