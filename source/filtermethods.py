
# Function that returns a dataframe containing the results of the ANOVA test between each feature and the classes.

class FilterMethods:
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def anova_oneway(self):
        import pandas as pd
        import numpy as np
        from scipy.stats import f_oneway
        
        anova_df = {'features':[], 'F_test_stat':[], 'p_value':[]}
        dataset = self.X_train.copy()
        dataset['labels'] = self.y_train
        for name in self.X_train.columns:
            income_groups = [dataset.loc[dataset['labels']==subclass, name].values for subclass in dataset['labels'].dropna().unique()]
            for i in list(range(len(income_groups))):
                income_groups[i] = income_groups[i][np.logical_not(np.isnan(income_groups[i]))]
            stat, p_value = f_oneway(*income_groups)
            anova_df["features"].append(name)
            anova_df["F_test_stat"].append(stat)
            anova_df["p_value"].append(p_value)

        return pd.DataFrame(anova_df)


    # Function that returns a dataframe of mutual_information scores.
    def mutual_info(self, imputer):
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_selection import mutual_info_classif

        # Subclass ColumnTransformer to return a Dataframe with columns instead of just an array (what is usually returned)
        # This is useful after imputation.
        class ColumnTransformerPandas(ColumnTransformer):

            def fit(self, X, y=None):
                self.columns = X.columns
                return super().fit(X, y)

            def transform(self, X):
                return pd.DataFrame(super().transform(X), columns=self.columns)
            
            def fit_transform(self, X, y=None):
                self.columns = X.columns
                return pd.DataFrame(super().fit_transform(X, y), columns=self.columns)

        # Define the imputer for missing data.
        impute = ColumnTransformerPandas([
            ('imputer', imputer, self.X_train.columns)],
            remainder='passthrough')
        
        # Compute feature importance.
        importances = mutual_info_classif(impute.fit_transform(self.X_train), self.y_train)
        importances_df = pd.DataFrame()
        importances_df['features'] = self.X_train.columns.to_list()
        importances_df['mutual_info_score'] = importances
        importances_df['mutual_info_score_rank'] =  importances_df['mutual_info_score'].rank(method='average', ascending=False).astype('int')
        
        return importances_df


    # Function for calculating vif and returning a dataframe of values.
    def calc_vif(self, imputer, new_df=None):
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.compose import ColumnTransformer

        # Subclass ColumnTransformer to return a Dataframe with columns instead of just an array (what is usually returned)
        # This is useful after imputation.
        class ColumnTransformerPandas(ColumnTransformer):

            def fit(self, X, y=None):
                self.columns = X.columns
                return super().fit(X, y)

            def transform(self, X):
                return pd.DataFrame(super().transform(X), columns=self.columns)
            
            def fit_transform(self, X, y=None):
                self.columns = X.columns
                return pd.DataFrame(super().fit_transform(X, y), columns=self.columns)

        # Define the imputer for missing data.
        impute = ColumnTransformerPandas([
            ('imputer', imputer, self.X_train.columns)],
            remainder='passthrough')
        
        

        # Calculating VIF
        vif = pd.DataFrame()

        if isinstance(new_df, pd.DataFrame):
            vif["variables"] = new_df.columns
            vif["VIF"] = [variance_inflation_factor(new_df.values, i) for i in range(new_df.shape[1])]
        
        else:
            X_train_imp = impute.fit_transform(self.X_train)
            vif["variables"] = self.X_train.columns
            vif["VIF"] = [variance_inflation_factor(X_train_imp.values, i) for i in range(self.X_train.shape[1])]

        return(vif)
    
    # Function to return the columns to remove after iteratively removing feature with highest vif until
    # value of vif is below a threshold.
    def vif_threshold_reduction(self, threshold, imputer):
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer

        # Subclass ColumnTransformer to return a Dataframe with columns instead of just an array (what is usually returned)
        # This is useful after imputation.
        class ColumnTransformerPandas(ColumnTransformer):
            
            def fit(self, X, y=None):
                self.columns = X.columns
                return super().fit(X, y)

            def transform(self, X):
                return pd.DataFrame(super().transform(X), columns=self.columns)
            
            def fit_transform(self, X, y=None):
                self.columns = X.columns
                return pd.DataFrame(super().fit_transform(X, y), columns=self.columns)

        # Define the imputer for missing data.
        impute = ColumnTransformerPandas([
            ('imputer', imputer, self.X_train.columns)],
            remainder='passthrough')

        remove = {'feature': [], 'vif':[]}
        new_df = impute.fit_transform(self.X_train.copy())
        maxvif = np.inf
        n_cols = self.X_train.shape[1]
        while (maxvif > threshold) & (n_cols>1):
            if new_df.shape[1] == self.X_train.shape[1]:
                vif = self.calc_vif(imputer=imputer).sort_values(by='VIF', ascending=False).reset_index(drop=True)
            else:
                vif = self.calc_vif(new_df=new_df, imputer=imputer).sort_values(by='VIF', ascending=False).reset_index(drop=True)
            remove['feature'].append(vif['variables'][0])
            remove['vif'].append(vif['VIF'][0])
            new_df = new_df.drop(remove['feature'][-1], axis=1)
            maxvif = remove['vif'][-1]
            n_cols = new_df.shape[1]
            print(f"Num columns after removal: {n_cols}; removed: {remove['feature'][-1]}; VIF: {maxvif}")
        return remove


# # Define the column transformer from feature reduction using VIF. This can be used within a pipeline.
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class vif_reduction(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.vif_df = pd.read_csv('../processed_data/vif_df.csv')
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.vif_df[self.vif_df['vif']>self.threshold]['feature'].tolist(), axis=1, inplace=False)