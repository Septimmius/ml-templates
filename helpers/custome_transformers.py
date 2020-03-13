
class DeleteFeature(BaseEstimator, TransformerMixin):
    def __init__(self, X_to_delete):
        # create class variables
        self.X_to_delete = X_to_delete
    
    def fit(self, X, y=None):
        return self
    
    def delete_X(self, X, X_to_delete):
        assert type(X_to_delete)==list, 'X_to_delete must be a list of feature names'
        return X.drop(columns=X_to_delete)
    
    def transform(self, X, y=None):
        X_cpy = X.copy()
        return self.delete_X(X_cpy, self.X_to_delete)

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, X_to_impute, impute_strategy, fill_value=None):
        # create class variables
        self.X_to_impute = X_to_impute
        self.impute_strategy = impute_strategy
        self.fill_value = fill_value
        
    def fit(self, X, y=None):
        return self
    
    def impute_X(self, X, X_to_impute, strategy, fill_value=None):
        assert type(X_to_impute)==list, 'X_to_impute must be a list of feature names'
        assert strategy in ['mean', 'median', 'most_frequent', 'constant'], f'invalid strategy {strategy}' 
        if strategy == 'mean':
            X.fillna(dict(X[X_to_impute].mean()), inplace=True)
        elif strategy == 'median':
            X.fillna(dict(X[X_to_impute].median()), inplace=True)
        elif strategy == 'most_frequent':
            mode = X[X_to_impute].mode()
            X.fillna(dict(zip(mode.columns, mode.values.tolist()[0])), inplace=True)
        elif strategy == 'constant':
            assert fill_value != None, 'fill_value must not be None when strategy is constant'
            X[X_to_impute] = X[X_to_impute].fillna(fill_value)
        return X   
    
    def transform(self, X, y=None):
        X_cpy = X.copy()
        return self.impute_X(X_cpy, self.X_to_impute, self.impute_strategy, self.fill_value)

class Binarize(BaseEstimator, TransformerMixin):
    def __init__(self, X_to_binarize, threshold):
        # create class variables
        self.X_to_binarize = X_to_binarize
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
    
    def binarize(self, X, X_to_binarize, threshold):
        s = X[X_to_binarize].copy()
        #assert s.dtypes!='O', 'cannot binarize str'
        s.where(s > threshold, 1, inplace=True)
        s.where(s <= threshold, 0, inplace=True)
        X[X_to_binarize] = s.copy()
        return X
    
    def transform(self, X, y=None):
        X_cpy = X.copy()
        return self.binarize(X_cpy, self.X_to_binarize, self.threshold)
    
class KBinDiscretize(BaseEstimator, TransformerMixin):
    def __init__(self, X_to_kbin, n_bins, labels, strategy='quantile'):
        # create class variables
        self.X_to_kbin = X_to_kbin
        self.n_bins = n_bins
        self.labels = labels
        self.strategy = strategy
        
    def fit(self, X, y=None):
        return self

    def kbindiscretize(self, X, X_to_kbin, n_bins, labels, strategy='quantile'):
        assert type(X_to_kbin)==str, 'X_to_kbin must be a single str, which is the column name'
        assert type(n_bins)==int, 'n_bins should be an integer'
        assert type(labels)==list, 'labels must be a list of label names'
        assert len(labels)==n_bins, 'length of labels must equal n_bins'
        if strategy=='quantile':
            X[X_to_kbin] = pd.qcut(x=X[X_to_kbin].copy(), q=n_bins, labels=labels, retbins=False, duplicates='drop')
        return X
    def transform(self, X, y=None):
        X_cpy = X.copy()
        return self.kbindiscretize(X_cpy, self.X_to_kbin, self.n_bins, self.labels, self.strategy)
