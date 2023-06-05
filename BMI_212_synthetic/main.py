import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Set the random seed for reproducibility
np.random.seed(42)

# Number of rows in the DataFrame
num_rows = 800


def create_dataframe():
    # Create the synthetic data
    data = {
        'Person ID': np.random.randint(100000, 999999, size=num_rows),  # Random 6-digit numbers
        'SNP_0': np.random.randint(0, 2, size=num_rows),  # Random binary values (0 or 1)
        'SNP_1': np.random.randint(0, 2, size=num_rows),
        'SNP_2': np.random.randint(0, 2, size=num_rows),
        'SNP_3': np.random.randint(0, 2, size=num_rows),
        'SNP_4': np.random.randint(0, 2, size=num_rows),
        'SNP_5': np.random.randint(0, 2, size=num_rows),
        'SNP_6': np.random.randint(0, 2, size=num_rows),
        'SNP_7': np.random.randint(0, 2, size=num_rows),
        'SNP_8': np.random.randint(0, 2, size=num_rows),
        'SNP_9': np.random.randint(0, 2, size=num_rows),
        'SNP_10': np.random.randint(0, 2, size=num_rows),
        'SNP_11': np.random.randint(0, 2, size=num_rows),
        'SNP_12': np.random.randint(0, 2, size=num_rows),
        'SNP_13': np.random.randint(0, 2, size=num_rows),
        'SNP_14': np.random.randint(0, 2, size=num_rows),
        'SNP_15': np.random.randint(0, 2, size=num_rows),
        'SNP_16': np.random.randint(0, 2, size=num_rows),
        'SNP_17': np.random.randint(0, 2, size=num_rows),
        'SNP_18': np.random.randint(0, 2, size=num_rows),
        'SNP_19': np.random.randint(0, 2, size=num_rows),
        'SNP_20': np.random.randint(0, 2, size=num_rows),
        'SNP_21': np.random.randint(0, 2, size=num_rows),
        'SNP_22': np.random.randint(0, 2, size=num_rows),
        'SNP_23': np.random.randint(0, 2, size=num_rows),
        'SNP_24': np.random.randint(0, 2, size=num_rows),
        'SNP_25': np.random.randint(0, 2, size=num_rows),
        'SNP_26': np.random.randint(0, 2, size=num_rows),
        'SNP_27': np.random.randint(0, 2, size=num_rows),
        'SNP_28': np.random.randint(0, 2, size=num_rows),
        'SNP_29': np.random.randint(0, 2, size=num_rows),
        'SNP_30': np.random.randint(0, 2, size=num_rows),
        'T2D_Status': np.random.randint(0, 2, size=num_rows),
        'race': np.random.choice(['Black', 'White'], size=num_rows),
        'sex': np.random.choice(['Male', 'Female'], size=num_rows),
    }

    # Calculate interactions between SNP columns
    snp_columns = [col for col in data if col.startswith('SNP_')]
    interaction_columns = []
    for i in range(len(snp_columns)):
        for j in range(i + 1, len(snp_columns)):
            col_i = snp_columns[i]
            col_j = snp_columns[j]
            interaction_col = f'{col_i}_{col_j}_interaction'
            data[interaction_col] = data[col_i] * data[col_j]
            interaction_columns.append(interaction_col)

    column_order = snp_columns + interaction_columns + ['T2D_Status', 'race', 'sex']

    # Create the DataFrame
    df = pd.DataFrame(data)[column_order]

    return df


def run_random_forest(X, y, features_list):
    y = np.ravel(y)

    # Create a Random Forest classifier
    rf = RandomForestClassifier()

    # Fit the model to the data
    rf.fit(X, y)

    # Get the feature importances
    importances = rf.feature_importances_

    # Create a DataFrame to store the feature importances
    feature_importances = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    })

    # Sort the DataFrame by importance (descending order)
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    feature_importances.to_excel("RF Feature Weights.xlsx")

    return feature_importances


def calculate_prs(X_test, features_list, feature_importances):

    # Extract the feature importances for SNP columns
    snp_importances = feature_importances[feature_importances['Feature'].isin(features_list)]

    # Calculate the PRS for the test set
    prs_test = np.zeros(len(X_test))
    print(X_test)
    for snp in features_list:
        importance = snp_importances[snp_importances['Feature'] == snp]['Importance'].values[0]
        genotype = X_test[snp]
        prs_test += importance * genotype

    return prs_test


def find_optimal_f1_threshold(y_test, prs_scores):

    # list of potential thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    f1_scores = []

    # Iterate through each threshold
    for threshold in thresholds:

        # Convert predicted probabilities into binary predictions based on the threshold
        y_pred = (prs_scores >= threshold).astype(int)

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)

        # Append F1 score array
        f1_scores.append(f1)

    # Find the threshold that maximizes the F1 score -- might want to change how we determine this!
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    optimal_f1_score = f1_scores[np.argmax(f1_scores)]

    return optimal_threshold


def find_optimal_auc_threshold(y_test, prs_scores):
    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, prs_scores)

    # Compute the area under the ROC curve (AUC-ROC)
    auc_roc = auc(fpr, tpr)

    # Find the index of the threshold that maximizes the AUC-ROC
    max_index = np.argmax(auc_roc)

    # Retrieve the threshold that maximizes the AUC-ROC
    optimal_threshold = thresholds[max_index]

    print("Optimal Threshold:", optimal_threshold)


def general_evaluate(feature_importances, features_list, X_test, y_test):

    # remove race from testing data
    X_test = X_test.drop(['race'], axis=1)
    y_test = y_test.drop(['race'], axis=1)
    X_test = X_test.drop(['sex'], axis=1)
    y_test = y_test.drop(['sex'], axis=1)

    prs_test = calculate_prs(X_test, features_list, feature_importances)

    # find optimal threshold given f1 or auc scores
    optimal_f1_threshold = find_optimal_f1_threshold(y_test, prs_test)
    optimal_auc_threshold = find_optimal_auc_threshold(y_test, prs_test)

    # find predicted y values based on PRS and optimal threshold
    y_pred = (prs_test >= optimal_f1_threshold).astype(int)
    # or
    y_pred = (prs_test >= optimal_auc_threshold).astype(int)

    # Evaluate the performance of the PRS
    auc_roc_prs = roc_auc_score(y_test, prs_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return auc_roc_prs, f1, accuracy


def race_stratified_evaluate(feature_importances, features_list, X_test, y_test):
    # Split the data by race
    racial_groups = X_test['race'].unique()

    f1_scores = []
    auc_roc_scores = []
    accuracy_scores = []
    for race in racial_groups:
        race_subset_X = X_test[X_test['race'] == race]
        race_subset_y = y_test[y_test['race'] == race]

        auc_roc_prs, f1, accuracy = general_evaluate(feature_importances, features_list, race_subset_X, race_subset_y)
        auc_roc_scores.append(auc_roc_prs)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    df = pd.DataFrame({'Race': racial_groups, 'AUC-ROC': auc_roc_scores, 'F1': f1_scores})
    return df


def sex_stratified_evaluate(feature_importances, features_list, X_test, y_test):
    # Split the data by sex
    sexes = X_test['sex'].unique()

    auc_roc_scores = []
    f1_scores = []
    accuracy_scores = []
    for sex in sexes:
        sex_subset_X = X_test[X_test['sex'] == sex]
        sex_subset_y = y_test[y_test['sex'] == sex]

        auc_roc_prs, f1, accuracy = general_evaluate(feature_importances, features_list, sex_subset_X, sex_subset_y)
        auc_roc_scores.append(auc_roc_prs)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    df = pd.DataFrame({'Sex': sexes, 'AUC-ROC': auc_roc_scores, 'F1': f1_scores})
    return df


def rf_eval(feature_importances, features_list, X_test, y_test):
    # evaluate RF for overall population
    auc_roc_prs, f1, accuracy = general_evaluate(feature_importances, features_list, X_test, y_test)
    print("RF AUC-ROC overall: ", auc_roc_prs)
    print("RF F1 overall: ", f1)
    print("RF accuracy overall: ", accuracy)

    '''
    # evaluate RF stratified by race
    output_df = race_stratified_evaluate(feature_importances, features_list, X_test, y_test)
    output_df.to_excel('(synthetic) evaluation of RF by race.xlsx', index=False)

    # evaluate RF stratified by sex
    output2_df = sex_stratified_evaluate(feature_importances, features_list, X_test, y_test)
    output2_df.to_excel('(synthetic) evaluation of RF by sex.xlsx', index=False)
    '''

def run_lasso(X_train, y_train, features_list):
    # Initialize the LASSO regression model
    lasso = Lasso(alpha=0.1)

    # Fit the LASSO model
    lasso.fit(X_train, y_train)

    # Get the coefficients (weights) of the LASSO model
    coefficients = lasso.coef_

    # Create a DataFrame to store the feature importances
    feature_coefficients = pd.DataFrame({
        'Feature': features_list,
        'Importance': coefficients
    })

    # Sort the DataFrame by importance (descending order)
    feature_coefficients = feature_coefficients.sort_values(by='Importance', ascending=False)

    feature_coefficients.to_excel('LASSO Feature Weights.xlsx')

    return feature_coefficients


def lasso_eval(lasso_coefficients, features_list, X_test, y_test):
    # evaluate RF for overall population
    auc_roc_prs, f1, accuracy = general_evaluate(lasso_coefficients, features_list, X_test, y_test)
    print("lasso AUC-ROC overall: ", auc_roc_prs)
    print("lasso F1 overall: ", f1)
    print("lasso accuracy overall: ", accuracy)

    '''
    # evaluate RF stratified by race
    output_df = race_stratified_evaluate(lasso_coefficients, features_list, X_test, y_test)
    output_df.to_excel('(synthetic) evaluation of lasso by race.xlsx', index=False)

    # evaluate RF stratified by sex
    output2_df = sex_stratified_evaluate(lasso_coefficients, features_list, X_test, y_test)
    output2_df.to_excel('(synthetic) evaluation of lasso by sex.xlsx', index=False)
    '''

def organize_data():
    # Create the DataFrame
    df = create_dataframe()

    features_list = df.columns.tolist()
    features_list.remove('T2D_Status')

    # X = SNP data, race and sex
    X = df[features_list]

    # y = diabetes diagnosis, race and sex
    y = df[['T2D_Status', 'race', 'sex']]

    features_list.remove('sex')
    features_list.remove('race')

    return features_list, X, y


def main():
    features_list, X, y = organize_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Drop the race and sex columns from the training data
    X_train = X_train.drop(['race'], axis=1)
    y_train = y_train.drop(['race'], axis=1)
    X_train = X_train.drop(['sex'], axis=1)
    y_train = y_train.drop(['sex'], axis=1)

    # Run and evaluate Random Forest
    rf_importances = run_random_forest(X_train, y_train, features_list)
    rf_eval(rf_importances, features_list, X_test, y_test)

    # Run and evaluate LASSO
    lasso_coefficients = run_lasso(X_train, y_train, features_list)
    lasso_eval(lasso_coefficients, features_list, X_test, y_test)


# Run the main function
if __name__ == "__main__":
    main()
