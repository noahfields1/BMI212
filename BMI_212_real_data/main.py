import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def create_dataframe():
    # read in file
    data = pd.read_csv('T2D_df.csv')
    data = data.replace(2, 1)

    # Calculate interactions between SNP columns
    snp_columns = [col for col in data if col.startswith('rs')]
    interaction_columns = []
    for i in range(len(snp_columns)):
        for j in range(i + 1, len(snp_columns)):
            col_i = snp_columns[i]
            col_j = snp_columns[j]
            interaction_col = f'{col_i}_{col_j}_interaction'
            data[interaction_col] = data[col_i] * data[col_j]
            interaction_columns.append(interaction_col)

    data = data.copy()

    column_order = snp_columns + interaction_columns + ['race', 'sex', 'T2D_Status']

    # Create the DataFrame
    df = pd.DataFrame(data)[column_order]
    
    return df



def optimize_rf_hyperparams(X, y):

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 4, 6],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a Random Forest classifier
    rf = RandomForestClassifier()

    # Perform grid search cross-validation with F1 score as the evaluation metric
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X, y)

    best_hyperparams = grid_search.best_params_

    # Print the best hyperparameters and corresponding F1 score
    print("Best Hyperparameters: ", best_hyperparams)


def run_random_forest(X, y, features_list):
    y = np.ravel(y)

    # uncomment following line to optimize hyperparameters for RF -- runtime is long, so we did this
    # once and hardcoded the resulting values:

    # optimize_rf_hyperparams(X, y)

    # Create a Random Forest classifier
    rf = RandomForestClassifier(class_weight='balanced', max_depth=None, max_features='auto', min_samples_split=4, n_estimators=200)

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


def general_evaluate(feature_importances, features_list, X_test, y_test):

    race = X_test['race']
    sex = X_test['sex']

    # remove race from testing data
    X_test = X_test.drop(['race'], axis=1)
    y_test = y_test.drop(['race'], axis=1)
    X_test = X_test.drop(['sex'], axis=1)
    y_test = y_test.drop(['sex'], axis=1)

    print("X_test: ", X_test)
    print("y_test: ", y_test)

    prs_test = calculate_prs(X_test, features_list, feature_importances)

    # print("PRS: ", prs_test)

    # find optimal threshold given f1 or auc scores
    optimal_f1_threshold = find_optimal_f1_threshold(y_test, prs_test)
    optimal_auc_threshold = find_optimal_auc_threshold(y_test, prs_test)

    # find predicted y values based on PRS and optimal threshold

    print("optimal f1 threshold: ", optimal_f1_threshold)
    y_pred = (prs_test >= optimal_f1_threshold).astype(int)
    # or
    # y_pred = (prs_test >= optimal_auc_threshold).astype(int)

    T2D_Status = y_test['T2D_Status']
    df = pd.DataFrame({'PRS': prs_test, 'Predicted T2D_Status': y_pred, 'T2D_Status': T2D_Status, 'Race': race,
                       'Sex': sex})
    df.to_excel('PRS values with race_sex.xlsx', index=False)

    # Evaluate the performance of the PRS
    auc_roc_prs = roc_auc_score(y_test, prs_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    print(report)

    # uncomment below to plot overall ROC curve
    '''
    # Plotting the ROC curve

    fpr, tpr, thresholds = roc_curve(y_test, prs_test)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    '''

    return auc_roc_prs, f1, accuracy, prs_test


def race_stratified_evaluate(feature_importances, features_list, X_test, y_test):
    # Split the data by race
    racial_groups = X_test['race'].unique()

    f1_scores = []
    auc_roc_scores = []
    accuracy_scores = []

    races_with_one_classif = ['None Indicated', 'Middle Eastern or North African', 'PMI: Skip', 'None of these',
                                 '0tive Hawaiian or Other Pacific Islander', 'I prefer not to answer']


    races_used = []
    colors = ['red', 'green', 'blue', 'purple']
    plt.figure()
    i = 0
    for race in racial_groups:
        if race in races_with_one_classif:
            continue
        races_used.append(race)

        race_subset_X = X_test[X_test['race'] == race]
        race_subset_y = y_test[y_test['race'] == race]

        print("race: ", race)

        auc_roc_prs, f1, accuracy, prs_test = general_evaluate(feature_importances, features_list, race_subset_X, race_subset_y)
        auc_roc_scores.append(auc_roc_prs)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

        # uncomment below to plot ROC curve by race


        y_test_no_race = race_subset_y.drop(['race'], axis=1)
        y_test_only = y_test_no_race.drop(['sex'], axis=1)

        fpr, tpr, thresholds = roc_curve(y_test_only, prs_test)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for current race
        plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(race, roc_auc), color=colors[i])
        i = i + 1


    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # Set labels and title
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # Add legend
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()


    df = pd.DataFrame({'Race': races_used, 'AUC-ROC': auc_roc_scores, 'F1': f1_scores, 'Accuracy': accuracy_scores})
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

        auc_roc_prs, f1, accuracy, prs_test = general_evaluate(feature_importances, features_list, sex_subset_X, sex_subset_y)
        auc_roc_scores.append(auc_roc_prs)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    df = pd.DataFrame({'Sex': sexes, 'AUC-ROC': auc_roc_scores, 'F1': f1_scores})
    return df


def training_evaluate(feature_importances, features_list, X_train, y_train):
    prs_train = calculate_prs(X_train, features_list, feature_importances)

    print("PRS: ", prs_train)

    # find optimal threshold given f1 or auc scores
    optimal_f1_threshold = find_optimal_f1_threshold(y_train, prs_train)
    optimal_auc_threshold = find_optimal_auc_threshold(y_train, prs_train)

    # find predicted y values based on PRS and optimal threshold
    y_pred = (prs_train >= optimal_f1_threshold).astype(int)
    # or
    # y_pred = (prs_test >= optimal_auc_threshold).astype(int)

    # Evaluate the performance of the PRS
    auc_roc_prs = roc_auc_score(y_train, prs_train)
    f1 = f1_score(y_train, y_pred)
    accuracy = accuracy_score(y_train, y_pred)

    return auc_roc_prs, f1, accuracy


def rf_eval(feature_importances, features_list, X_test, y_test, X_train, y_train):

    # evaluate RF on training data (compare w testing data to test for overfitting)
    auc_roc_prs, f1, accuracy = training_evaluate(feature_importances, features_list, X_train, y_train)
    print("RF (training) AUC-ROC overall: ", auc_roc_prs)
    print("RF (training) F1 overall: ", f1)
    print("RF (training) accuracy overall: ", accuracy)


    # evaluate RF for overall population
    auc_roc_prs, f1, accuracy, prs_test = general_evaluate(feature_importances, features_list, X_test, y_test)
    print("RF (testing) AUC-ROC overall: ", auc_roc_prs)
    print("RF (testing) F1 overall: ", f1)
    print("RF (testing) accuracy overall: ", accuracy)

    '''
    # evaluate RF stratified by race
    output_df = race_stratified_evaluate(feature_importances, features_list, X_test, y_test)
    output_df.to_excel('(synthetic) evaluation of RF by race.xlsx', index=False)

    # evaluate RF stratified by sex
    output2_df = sex_stratified_evaluate(feature_importances, features_list, X_test, y_test)
    output2_df.to_excel('(synthetic) evaluation of RF by sex.xlsx', index=False)
    '''


def optimize_alpha(X_train, y_train):
    # Define a list of alpha values to explore
    alpha_values = [0.001, 0.01, 0.1, 1, 10]

    # Create an empty dictionary to store the F1 scores for each alpha
    f1_scores = {}

    for alpha in alpha_values:

        model = Lasso(alpha=alpha)

        # Perform cross-validation and calculate the F1 score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        # Store the mean F1 score for the current alpha value
        f1_scores[alpha] = cv_scores.mean()

    # Find the alpha value with the highest F1 score
    best_alpha = max(f1_scores, key=f1_scores.get)

    print("best alpha: ", best_alpha)
    return best_alpha


def run_lasso(X_train, y_train, features_list):
    # determine optimal alpha value
    print("y-train: ", y_train)
    best_alpha = optimize_alpha(X_train, y_train)

    # Initialize the LASSO regression model
    lasso = Lasso(alpha=best_alpha)

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

    feature_coefficients.to_excel('LASSO Feature Weights(1).xlsx')

    return feature_coefficients


def lasso_eval(lasso_coefficients, features_list, X_test, y_test, X_train, y_train):
    '''
    # evaluate lasso on training data (compare w testing data to test for overfitting)
    auc_roc_prs, f1, accuracy = training_evaluate(lasso_coefficients, features_list, X_train, y_train)
    print("lasso (training) AUC-ROC overall: ", auc_roc_prs)
    print("lasso (training) F1 overall: ", f1)
    print("lasso (training) accuracy overall: ", accuracy)
    '''

    # evaluate lasso for overall population on testing data
    auc_roc_prs, f1, accuracy, prs_test = general_evaluate(lasso_coefficients, features_list, X_test, y_test)
    print("lasso (testing) AUC-ROC overall: ", auc_roc_prs)
    print("lasso (testing) F1 overall: ", f1)
    print("lasso (testing) accuracy overall: ", accuracy)

    '''
    # evaluate lasso stratified by race
    output_df = race_stratified_evaluate(lasso_coefficients, features_list, X_test, y_test)
    output_df.to_excel('evaluation of lasso by race.xlsx', index=False)
    '''
    '''
    # evaluate lasso stratified by sex
    output2_df = sex_stratified_evaluate(lasso_coefficients, features_list, X_test, y_test)
    output2_df.to_excel('evaluation of lasso by sex.xlsx', index=False)
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

    '''
    # Run and evaluate Random Forest
    rf_importances = run_random_forest(X_train, y_train, features_list)
    rf_eval(rf_importances, features_list, X_test, y_test, X_train, y_train)
    '''

    # Run and evaluate LASSO
    lasso_coefficients = run_lasso(X_train, y_train, features_list)
    lasso_eval(lasso_coefficients, features_list, X_test, y_test, X_train, y_train)


# Run the main function
if __name__ == "__main__":
    main()
