# prepare numeric and categorical feature names for pipelines
def separate_raw_num_cat(df_X, verbose=True):
    '''
    separate numeric and categorical column names by their raw data type in the dataframe df_X
    suitable for dataset with huge number of features, otherwise can pick manually
    return: lists of separated num and cat column names, in that order
    '''
    raw_num_cols = []
    raw_cat_cols = []
    strange = []
    
    for col in df_X.columns:
        if df_X[col].dtype=='O':
            raw_cat_cols.append(col)
        elif df_X[col].dtype=='int64' or df_X[col].dtype=='float64':
            raw_num_cols.append(col)
        else:
            strange.append(col)
    assert len(raw_num_cols+raw_cat_cols+strange)==len(df_X.columns), f'lengths mismatch for separate_raw_num_cat, {len(raw_num_cols+raw_cat_cols+strange)} != {len(df_X.columns)}'
    assert len(strange)==0, f'found strange variable names {", ".join(strange)}, should go back and check'
    if verbose:
        print(raw_num_cols, '\n')
        print(raw_cat_cols, '\n')
        print(f'raw_num_cols: {len(raw_num_cols)}\nraw_cat_cols: {len(raw_cat_cols)}')
    return raw_num_cols, raw_cat_cols


def adjust_raw_num_cat_dtypes(num_to_cat, df_X, verbose=True):
    '''
    adjust the two lists so that some should-be categorical columns are removed from the num_cols, and added to cat_cols
    note this does not change the data type in the dataset. it just ensure we feed the correct features into the pipelines, will change the data type in the pipeline
    return: adjusted num_cols and cat_cols
    '''
    num_cols, cat_cols = separate_raw_num_cat(df_X, verbose=False)
    # removing num_to_cat from num_cols to cat_cols
    num_cols = [num_col for num_col in num_cols if num_col not in num_to_cat]
    cat_cols.extend(num_to_cat)
    assert len(num_cols+cat_cols)==len(df_X.columns), f'lengths mismatch for adjust_raw_num_cat_dtypes, {len(num_cols+cat_cols)} != {len(df_X.columns)}'
    if verbose:
        print(num_cols, '\n')
        print(cat_cols, '\n')
        print(f'adjusted_num_cols: {len(num_cols)}\nadjusted_cat_cols: {len(cat_cols)}')
    return num_cols, cat_cols

def save_num_cat_to_csv(num_cols, cat_cols, csv_path_num, csv_path_cat, verbose=True):
    '''
    save both the numeric and categorical feature names lists into csv files
    return: None
    '''
    assert type(num_cols)==list, 'num_cols must be a list'
    assert type(cat_cols)==list, 'cat_cols must be a list'
    pd.DataFrame(num_cols).to_csv(csv_path_num)
    pd.DataFrame(cat_cols).to_csv(csv_path_cat)
    if verbose:
        print(f'num_cols is saved to {csv_path_num}\ncat_cols is saved to {csv_path_cat}')

def load_num_cat_csv_to_list(csv_path_num, csv_path_cat):
    '''
    load the saved csv files of numeric and categorical feature names back into lists
    '''
    return pd.read_csv(csv_path_num, index_col=0).values.flatten().tolist(), pd.read_csv(csv_path_cat, index_col=0).values.flatten().tolist()
    
def prepare_num_cat_for_pipeline(df_X, num_to_cat, csv_path_num=None, csv_path_cat=None, save_and_load=False):
    '''
    master function handling the entire process of numeric and categorical feature separation, the returned lists of names will be inputs for pipelines
    Note: process flow can be controlled using save_and_load to check the separated lists of features before saving them into csv files,
    return: separated lists of numeric and categorical features
    '''
    assert type(df_X)==pd.core.frame.DataFrame, 'df_X must be a dataframe'
    assert type(num_to_cat)==list, 'num_to_cat must be a list'
    assert type(csv_path_num)==str, 'csv_path_num must be a string'
    assert type(csv_path_cat)==str, 'csv_path_cat must be a string'
    
    if save_and_load==False:
        return adjust_raw_num_cat_dtypes(num_to_cat, df_X, verbose=True)
    else:
        assert csv_path_num != None, 'must provide csv_path for num'
        assert csv_path_cat != None, 'must provide csv_path for cat'
        num_cols, cat_cols = adjust_raw_num_cat_dtypes(num_to_cat, df_X, verbose=False)
        save_num_cat_to_csv(num_cols, cat_cols, csv_path_num, csv_path_cat, verbose=True)
        numeric_X, categorical_X = load_num_cat_csv_to_list(csv_path_num, csv_path_cat)
        #print(len(numeric_X))
        #print(len(categorical_X))
        assert len(numeric_X+categorical_X)==len(df_X.columns), f'lengths mismatch when read from csv, {len(numeric_X+categorical_X)} != {len(df_X.columns)}'
        return numeric_X, categorical_X
    
def remove_features_intersection(features, features_to_remove):
    '''
    remove useless features that are in the numeric or categorical feature lists
    return: a clean list of numeric or categorical feature names
    '''
    return [feature for feature in features if feature not in list(set(features_to_remove).intersection(set(features)))]

def remove_features_from_num_cat(numeric_X, categorical_X, X_to_remove):
    '''
    remove unwanted features from lists of numeric and categorical features
    return: clean lists of numeric and categorical feature names
    '''
    assert type(numeric_X)==list, 'numeric_X must be a list'
    assert type(categorical_X)==list, 'categorical_X must be a list'
    assert type(X_to_remove)==list, 'X_to_remove must be a list'
    numeric_X_clean = remove_features_intersection(features=numeric_X, features_to_remove=X_to_delete)
    categorical_X_clean = remove_features_intersection(features=categorical_X, features_to_remove=X_to_delete)
    #print(len(numeric_X_clean))
    #print(len(categorical_X_clean))
    assert len(numeric_X_clean+categorical_X_clean)==(len(numeric_X+categorical_X)-len(X_to_remove)), f'lengths mismatch after removing features, {len(numeric_X_clean+categorical_X_clean)} != {(len(numeric_X+categorical_X)-len(X_to_remove))}'
    return numeric_X_clean, categorical_X_clean
