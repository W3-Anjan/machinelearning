import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    # an abstract method that is used to fit the step and to learn by examples
    # X – features data_frame and y – your target
    def fit(self, X, y=None):
        return self

    # an abstract method that is used to transform according to what happend in the fit method
    def transform(self, X):
        return X[self.attribute_names].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataAnalysis():

    def __init__(self, dir):
        self.directory = dir

    # when you call fetchHousingData(), it creates a datasets/housing directory in
    # your workspace, downloads the housing.tgz file, and extracts the housing.csv from it in
    # this directory.
    def fetchHousingData(self, housing_url):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            tgz_path = os.path.join(self.directory, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=self.directory)
            housing_tgz.close()

    def load_housing_data(self):
        csv_path = os.path.join(self.directory, "housing.csv")
        return pd.read_csv(csv_path)

    def split_train_test(self, data, test_ratio):
        # The permutation() method returns a re-arranged array (and leaves the original array un-changed).
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())


def main():
    PARENT_DIR = "E:/Machine learning/datasets/housing"

    ml = DataAnalysis(PARENT_DIR)
    ml.fetchHousingData(housing_url=HOUSING_URL)
    # load to pandas data frame
    housing_df = ml.load_housing_data()
    #print(housing_df.head)


    # The info() method is useful to get a quick description of the data, in particular the
    # total number of rows, and each attribute’s type and number of non-null values
    housing_df.info()
    print('\n')
    # from the data set ocean_proximity is type object and other attributes are numerical
    # ocean_proximity is a categorical attribute
    #print(housing_df['ocean_proximity'].value_counts())

    #The describe() method shows a summary of the numerical attributes
    #print(housing_df.describe())

    # display the numerical attributes on histogram
    #housing_df.hist(bins=50, figsize=(20, 15))
    #plt.show()

    # OWN SAMPLING
    # (obsolete) 20% of the dataset will be treated as test set
    #train_set, test_set = ml.split_train_test(housing_df, 0.2)
    #print(len(train_set), "train +", len(test_set), "test")

    '''
    RANDOM SAMPLING
    '''
    # (Latest) Sklearn itself provides a split method
    #train_set, test_set = train_test_split(housing_df, test_size= 0.2, random_state=42)
    #print(len(train_set), "train +", len(test_set), "test")

    '''
    STRATIFIED SAMPLING
    '''
    # Note: the reason of dividing with 1.5 is the income categories needs to be less in number.
    # So that the stratified sampling can be applied. that means the overall
    # population will represent less number of sample.
    # Here the overall sampling can be reduced to 1.5 times less (Previously sample was 16 but now 11)
    housing_df["income_cat"] = np.ceil(housing_df["median_income"] / 1.5 )
    housing_df["income_cat"].where(housing_df["income_cat"] < 5, 5.0, inplace=True)

    # Stratified Sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
        strat_train_set = housing_df.loc[train_index]
        strat_test_set = housing_df.loc[test_index]

    # drop income_cat attribute from both train and test set
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True) # The drop() function is used to drop specified labels from rows or columns.

    # Display train & test set data length
    # 16512 train set + 4128 test set= 20640 data set
    #print(print(len(strat_train_set), "train +", len(strat_test_set), "test"))

    """
    # DISPLAY SCATTER PLOT
    housing_df = strat_train_set.copy()
    y = housing_df["latitude"]
    x = housing_df["longitude"]
    # Plot labels
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    # Visualize the data
    plt.scatter(x, y, alpha=0.4,
                s=housing_df["population"]/100, label="population",
                c=housing_df['median_house_value'],
                cmap=plt.get_cmap("jet")
                )
    # Color bar
    cbar = plt.colorbar()
    cbar.set_label('Median House Value', fontsize=16)
    plt.show() # this will show the plot
    """

    """
    # CALCULATE CORRELATIONS
    # Combined attributes
    # In the below we have done attribute combination with custom transformer
    housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
    housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["total_rooms"]
    housing_df["population_per_household"]=housing_df["population"]/housing_df["households"]

    corr_matrix = housing_df.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    """

    '''
    DATA PREPARING
    '''
    # drop the Label variable and keep the predictors/ features variables
    # Since we are predicting the median house value so we will drop the attribute as it will be treated as Label
    housing_predictors = strat_train_set.drop("median_house_value", axis=1) #axis =1 represents column
    # label is the data that we want to predict based on other attributes
    # Here we need to predict the median_house_value
    housing_labels = strat_train_set["median_house_value"].copy()


    """
    # DATA CLEANING
    # WITHOUT PIPELINE TRANSFORM
    # TRANSFORM MISSING VALUES
    # You noticed earlier that the total_bedrooms
    # attribute has some missing values, so let’s fix this.
    imputer = SimpleImputer(strategy='mean')

    #Since the median can only be calculated on numerical attributes
    #So we need to create a copy of the data without ocean proximity text attribute
    housing_num = housing_predictors.drop("ocean_proximity", axis=1)
    #The imputer has simply computed the median of each attribute and stored the result
    # in its statistics_ instance variable.
    imputer.fit(housing_num)

    # train data with numerical attributes
    X = imputer.transform(housing_num)
    # train set
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

    # TRANSFORM CATEGORICAL ATTRIBUTES
    # handling text and categorical attributes
    # Data transform
    encoder = LabelEncoder()
    housing_cat = housing_predictors["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    #print(housing_cat_encoded)

    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    #print(housing_cat_1hot.toarray())

    # This will do the 2 upper tasks of transforms in a single time
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    #print(housing_cat_1hot)

    # TRANSFORM COMBINED ATTRIBUTES
    # Custom transformers to automatically combine the attributes
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing_df.values)
    #np.set_printoptions(threshold=np.inf)
    print(housing_extra_attribs)
    """

    '''
    DATA CLEANING
    '''
    #PIPELINE TRANSFORMATION
    # We can do upper data cleaning steps of
    # 1. Transform Missing values of Numerical attributes with Median value
    # 2. Transform to combined attributes
    # 3. Transform similar scaling of numerical attributes
    # 4. Transform Categorical attributes of one hot vector
    housing_num = housing_predictors.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        # ('label_binarizer', MyLabelBinarizer() We can use this also
        ('one_hot_encoder', OneHotEncoder(sparse=False)),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing_predictors)
    print(housing_prepared.shape)

    '''
    EVALUATING TRAIN SET & PREPARE MODEL & RMSE
    '''

    '''
    LINEAR REGRESSION
    '''
    """
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    
    # calculate Root Mean Square Error (RMSE)
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("RMSE:\t\t", lin_rmse)
    # so a typical prediction error of $68,628 is not very satisfying.
    # it is much more likely that the model has badly underfit the data.
    """

    '''
    DECISION TREE REGRESSION
    '''
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # calculate RMSE
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # so a typical prediction error of $0 is not very satisfying.
    # it is much more likely that the model has badly overfit the data.
    print("RMSE:\t\t", tree_rmse)

    # cross validation with small train data
    # create 10 fold of train data
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    ml.display_scores(tree_rmse_scores)






if __name__== "__main__":
    main()