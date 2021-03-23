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

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
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


def main():
    PARENT_DIR = "E:/Machine learning/datasets/housing"

    ml = DataAnalysis(PARENT_DIR)
    ml.fetchHousingData(housing_url=HOUSING_URL)
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

    # (obsolete) 20% of the dataset will be treated as test set
    #train_set, test_set = ml.split_train_test(housing_df, 0.2)
    #print(len(train_set), "train +", len(test_set), "test")

    # (Latest) Sklearn itself provides a split method
    #train_set, test_set = train_test_split(housing_df, test_size= 0.2, random_state=42)
    #print(len(train_set), "train +", len(test_set), "test")

    # Note: the reason of dividing with 1.5 is the income categories needs to be less in number.
    # So that the stratified sampling can be applied. that means the overall
    # population will represent less number of sample.
    # Here the overall sampling can be reduced to 1.5 times less (Previously sample was 16 but now 11)
    housing_df["income_cat"] = np.ceil(housing_df["median_income"] / 1.5 )
    housing_df["income_cat"].where(housing_df["income_cat"] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
        strat_train_set = housing_df.loc[train_index]
        strat_test_set = housing_df.loc[test_index]
        
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True) # The drop() function is used to drop specified labels from rows or columns.


    #display scatter plot
    housing_df = strat_train_set.copy()
    y = housing_df["latitude"]
    x = housing_df["longitude"]
    # Plot labels
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    # Setting alpha value will present the data more visually. Now more density areas are clear
    #plt.scatter(x, y, alpha=0.1)
    #plt.show()

    # Visualize the data
    plt.scatter(x, y, alpha=0.4,
                s=housing_df["population"]/100, label="population",
                c=housing_df['median_house_value'],
                cmap=plt.get_cmap("jet")
                )
    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label('Median House Value', fontsize=16)
    plt.show() # this will show the plot


    #Calculate correlations among manually attribute combinations
    # In the below we have done attribute combination with custom transformer
    #housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
    #housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["total_rooms"]
    #housing_df["population_per_household"]=housing_df["population"]/housing_df["households"]

    #corr_matrix = housing_df.corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Data Prepare
    # drop the Label variable and keep the predictors/ features variables
    # Since we are predicting the median house value so we will drop the attribute as it will be treated as Label
    housing_predictors = strat_train_set.drop("median_house_value", axis=1) #axis =1 represents column
    # label is the data that we want to predict based on other attributes
    housing_labels = strat_train_set["median_house_value"].copy()

    #Data Cleaning
    # You noticed earlier that the total_bedrooms
    # attribute has some missing values, so let’s fix this.
    imputer = SimpleImputer(strategy='mean')

    #Since the median can only be calculated on numerical attributes
    #So we need to create a copy of the data without ocean proximity text attribute
    housing_num = housing_predictors.drop("ocean_proximity", axis=1)
    #The imputer has simply computed the median of each attribute and stored the result
    # in its statistics_ instance variable.
    imputer.fit(housing_num)
    #print(housing_num.median().values)

    # train data with numerical attributes
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

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

    # Custom transformers to automatically combine the attributes
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing_df.values)
    #np.set_printoptions(threshold=np.inf)
    print(housing_extra_attribs)


if __name__== "__main__":
    main()