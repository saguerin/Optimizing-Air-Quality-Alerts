from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, \
                                    GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, \
                            recall_score, precision_score

def main():
    # Load data
    # --------------------------------------------------------------------------
    data_fname  = '../local_analysis/dow_avg_all_tables_DISTINCT.csv'
    space_fname = '../local_analysis/space_type.csv'

    print('Loading data...')
    df_main_trimmed, space_type_df, df_main_unlabeled  = load_data(
                                                        data_fname, space_fname)
    # Generate arrays
    # --------------------------------------------------------------------------
    print('Generating arrays...')
    X, X_validate, y, y_validate, feature_labels  = get_scikit_arrays(
                                                    df_main_trimmed,
                                                    space_type_df,diff=False)
    
    # Run GridSearch to optimize hyper-parameters
    # ------------------------------------------------------------------------
    print('Running GridSearch...')
    best_params = RandomForestGridSearch(X, y, cv = 3)
    pd.DataFrame(estimator.best_params_, index = [0]).to_csv(
        'GridSearch_best_params.csv')
    
    # Apply optimized model to the left out test set
    # --------------------------------------------------------------------------
    print('Running machine learning model...')
    best_params_df = pd.read_csv('GridSearch_best_params.csv', index_col = 0)
    model_performance, confusion, feature_importances, probability = \
            RandomForestValidate(X, X_validate, y, y_validate, feature_labels,
                                 best_params_df, randomize_y = False)
    
    # Write results to disk
    pd.DataFrame(model_performance, index = [0]).to_csv(
        'model_performance.csv')
    pd.DataFrame(confusion, index = ['Home', 'Office'],
                 columns = ['Home', 'Office']).to_csv('confusion.csv')
    feature_importances.to_csv('feature_importances.csv')
    probability.to_csv('probability.csv')
    df_main_trimmed.to_csv('df_main_trimmed.csv')


def load_data(data_fname, space_fname):
    """Load data and filter users with missing data (i.e., 7 days a week and
    24 hours a day).
    
    Parameters
    ----------
    data_fname : str
        File path to csv file containing SQL query with user averages for every
        hour in the week
    space_fname : str
        File path to csv file containing SQL query with space_type (home/office/
        NULL) data for each user.
    
    Returns
    ----------
    df_main_trimmed : DataFrame
        Filtered day of week and hour of day data for each user. Only
        includes users with "House" or "Office" labels.
    space_type_df : DataFrame
        DataFrame with space type (house/office/NULL) data for each user
    df_main_unlabeled : str
        Same as df_main_trimmed but includes users without "House" or
        "Office" labels
    """
    expected_n_hrs = 24 * 7
    df = pd.read_csv(data_fname)
    space_type_df = pd.read_csv(space_fname)

    # Merge time course and space data
    df_main = df.merge(space_type_df, left_on='device_uuid', right_on='uuid')

    # Group by id to count each subjects # of hours (filter users with
    # missing data)
    df_main_grouped = df_main.loc[:,['device_uuid', 'avg_co2',
                                       'avg_voc']].groupby(by='device_uuid')

    # Identify subjects where we have 7 days and 24 hours for each day
    df_user_N = df_main_grouped.aggregate(pd.DataFrame.count)
    df_user_N = df_user_N.rename(columns={'avg_co2'  : 'n_hrs_co2',
                                          'avg_voc'  : 'n_hrs_voc',
                                          'avg_dust' : 'n_hrs_dust'}
                                 ).reset_index()

    # Add observation counts to df_main for filtering
    df_main = df_main.merge(df_user_N, on='device_uuid')

    # Filter based on expected # of hours (filter users with missing data)
    df_main_trimmed = df_main[df_main.n_hrs_co2==expected_n_hrs]
    df_main_trimmed = df_main_trimmed[df_main_trimmed.n_hrs_voc==expected_n_hrs]
    df_main_trimmed = df_main_trimmed[df_main_trimmed.n_hrs_dust
                                      ==expected_n_hrs]
    df_main_unlabeled = df_main_trimmed.copy()

    # Filter users without space_type data
    df_main_trimmed = df_main_trimmed[np.logical_or(
                                        df_main_trimmed.space_type=='Home',
                                        df_main_trimmed.space_type=='Office')]
    
    # Ensure that values are properly sorted
    df_main_trimmed.sort_values(by = ['device_uuid', 'day_of_week',
                                      'hour'])
    
    return df_main_trimmed, space_type_df, df_main_unlabeled
    
def get_scikit_arrays(df_main_trimmed, space_type_df):
    """Get data arrays for machine learning model with scikit-learn
    Parameters
    ----------
    df_main_trimmed : DataFrame
        Filtered day of week and hour of day data for each user returned by
        load_data.
    space_type_df : DataFrame
        DataFrame with space type (house/office/NULL) data for each user
        
    Returns
    ----------
    X : numpy array (float)
        X matrix for scikit-learn where each row is a user and each column is
        a feature (a hour in the week). 80% train/test data
    X_validate : numpy array (float)
        Same as X but 20% validate data
    y :  numpy array (float)
        y vector of binary category labels for scikit-learn (0 = home,
        1 = office). 80% train/test data.
    y_validate : DataFrame
       Same as y but 20% validate data
    feature_labels : DataFrame
       Hour, day of week, and metric labels for columns of X
    """
    # Need to create an X matrix such that each row is a user and each
    # column is a feature.
    # ..........................................................................
   
    # Group by users, then use apply to unstack each user's data and concatenate
    # across users
    X_df = df_main_trimmed.loc[:,['device_uuid', 'day_of_week',
                                  'hour', 'avg_co2', 'avg_voc',
                                  'avg_dust']].groupby(by='device_uuid')
    
    # Get feature labels
    feature_labels_list = list()
    for metric in ['avg_co2', 'avg_voc', 'avg_dust']:

        temp_df = X_df.get_group(df_main_trimmed.device_uuid.iloc[0]).loc[:,
                  ['day_of_week','hour']]
        temp_df['metric'] = metric
        feature_labels_list.append(temp_df)
    
    feature_labels = pd.concat(feature_labels_list,axis=0)

    X_df = X_df.apply(lambda df : df.drop(['device_uuid', 'day_of_week',
                                           'hour'], axis = 1).reset_index(
                                                        drop = True)).unstack()
    # Create array of X
    X = np.array(X_df)

    # Now we want to get the category labels (y vector) for all of these users
    # ..........................................................................
    y_df = pd.DataFrame(columns=['device_uuid'])
    y_df.device_uuid = X_df.index
    y_df = y_df.merge(space_type_df, left_on='device_uuid', right_on='uuid')
    
    # Verify that rows of y_df and X_df still match
    if not np.array_equal(y_df.device_uuid,X_df.index):
        ValueError('Rows of data matrix X and category labels y do not match')

    # Create binary y
    y = get_binary_y(y_df['space_type'])

    # See if we have NaN values from including dust
    rows2keep = np.logical_not(np.isnan(X.sum(axis = 1)))
    X = X[rows2keep, :]
    y = y[rows2keep]
    
    # Important! Split off sacred left-out validation set that we don't touch
    # until we have fully optimized the model
    X, X_validate, y, y_validate = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 123,
                                                    shuffle=True,stratify=y)
    
    return X, X_validate, y, y_validate, feature_labels

def RandomForestGridSearch(X, y, cv = 5):
    """Optimize hyper-parameters using Grid Search
    Parameters
    ----------
    X : numpy array (float)
        X matrix for scikit-learn where each row is a user and each column is
        a feature (a hour in the week). 80% train/test data

    y :  numpy array (float)
        y vector of binary category labels for scikit-learn (0 = home,
        1 = office). 80% train/test data
    cv : int
        Number of cross-validation folds.

    Returns
    ----------
    best_params : dict
        Optimal hyper-parameters based on f1 score
    """

    # Set up stratified k fold with scikit-learn
    skf = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 123)
    skf.get_n_splits(X, y)

    sm = SMOTE(random_state = 123, ratio = 'minority')
    RandomForest = RandomForestClassifier(random_state=123)

    pipe = Pipeline(steps = [('sm', sm), ('RFC', RandomForest)])

    param_grid = {'RFC__n_estimators' : [10, 20, 30, 40, 50, 60],
                  'RFC__max_depth' : [3, 6, 10, 20, 30, 40],
                  'RFC__min_samples_split' : [2, 4, 6, 8, 10, 12]}

    # Results:
    # {'RFC__max_depth' : 30, 'RFC__min_samples_split' : 6,
    #  'RFC__n_estimators' : 40}
    
    estimator = GridSearchCV(pipe, param_grid, scoring = 'f1',
                             fit_params = None, n_jobs = 1, iid = True,
                             refit = True, cv = skf, verbose = 1,
                             pre_dispatch ='2 * n_jobs', error_score = 100,
                             return_train_score = True)
    estimator.fit(X, y)

    return estimator.best_params_

def RandomForestValidate(X, X_validate, y, y_validate, feature_labels,
                         best_params_df, randomize_y = False):
    """Apply random forest model to left out test data using hyper-parameters
    optimized with Grid Search
    
    Parameters
    ----------
    X : numpy array (float)
        X matrix for scikit-learn where each row is a user and each column is
        a feature (a hour in the week). 80% train/test data
    X_validate : numpy array (float)
        Same as X but 20% validate data
    y :  numpy array (float)
        y vector of binary category labels for scikit-learn (0 = home,
        1 = office). 80% train/test data
    y_validate : DataFrame
       Same as y but 20
    feature_labels : DataFrame
       Hour, day of week, and metric labels for columns of X
    best_params_df : DataFrame
        Optimal hyper-parameters based on f1 score in pandas dataframe
    randomize_y : bool
        Whether to randomize labels (y vector) before estimation for
        randomized control.

    Returns
    ----------
    model_performance : dict
        Dictionary reporting performance metrics
    confusion : numpy array (float)
        Confusion matrix
    feature_importances :  DataFrame
        Feature importances for each feature
    probability : DataFrame
       Estimated probabilities for each sample in the validation set
       (for ROC analysis)
    """

    # Randomize training labels if requested
    if randomize_y :
        rand_rows = np.random.permutation(y.shape[0])
        y = y[rand_rows].copy()

    # Set up pipeline
    sm = SMOTE(random_state=123, ratio = 'minority')
    RandomForest = RandomForestClassifier(
           max_depth = best_params_df.loc[0, 'RFC__max_depth'],
           min_samples_split = best_params_df.loc[0, 'RFC__min_samples_split'],
           n_estimators = best_params_df.loc[0, 'RFC__n_estimators'],
           random_state = 123)

    pipe = Pipeline(steps = [('sm', sm), ('RFC', RandomForest)])

    # Fit pipeline to training data
    pipe.fit(X, y)

    # Get feature importance
    feature_importances = pd.DataFrame(
        {'day_of_week' : feature_labels['day_of_week'],
         'hour' : feature_labels['hour'],
         'metric' : feature_labels['metric'],
         'importance' : pipe.steps[1][1].feature_importances_})
    
    # Get decision variable for ROC plot
    probs = pipe.steps[1][1].predict_proba(X_validate)[:,1]
    probability = pd.DataFrame(
        {'y' : y_validate,
         'prob_office' : probs})

    # Apply to validation data and get performance
    y_pred = pipe.predict(X_validate)

    # Get scoring on left out validation data
    auc       = roc_auc_score(y_validate, probs)
    f1        = f1_score(y_validate, y_pred),
    precision = precision_score(y_validate, y_pred)
    recall    = recall_score(y_validate, y_pred)
    
    model_performance = {'auc'       : auc,
                         'f1'        : f1,
                         'precision' : precision,
                         'recall'    : recall}
    
    confusion = confusion_matrix(y_validate, y_pred)

    return model_performance, confusion, feature_importances, probability

def get_binary_y(y_str):
    """Converts string category labels to binary category labels needed by
    scikit-learn.
    String labels are sorted in alphanumeric order. The first label is assigned
    0 and the second label is assigned 1.
    Parameters
    ----------
    y_str : numpy array or list
        Category labels for each sample (i.e., trial). Length should equal #
        of trials.
    Returns
    ----------
    y : numpy array
        A binary version of y_str."""

    y_str = np.array(y_str)
    labels = np.unique(y_str)

    if len(labels) > 2:
        raise ValueError('More than 2 category labels found in y_str.')

    y = np.zeros(y_str.shape)
    y[y_str==labels[0]], y[y_str==labels[1]] = 0, 1

    return y

if __name__ == "__main__":
    main()