import numpy
import pandas as pd
class CustomDataFrame:
    def __init__(self, dataset, numerical_features, categorical_features, label, train_split=0.8):
        self.train_split = train_split
        self.ds_size = len(dataset)
        self.labels = []
        self.numerical_features = []
        self.categorical_features = []

        df = pd.DataFrame(dataset)
        unique_category_values = [df[cat_feature].unique() for cat_feature in categorical_features]

        for i in range(len(df[label])):
            temp_array = []

            if len(numerical_features) == 0:
                temp_array.append(1)
            else:
                for num_feature in numerical_features:
                    temp_array.append(df[num_feature][i])

                self.numerical_features.append(temp_array)

            temp_array = []

            if len(categorical_features) == 0:
                temp_array.append(1)

            else:
                for cat_feature in categorical_features:
                    for j in range(len(unique_category_values[categorical_features.index(cat_feature)])):
                        if unique_category_values[categorical_features.index(cat_feature)][j] == df[cat_feature][i]:
                            temp_array.append(1)
                        else:
                            temp_array.append(0)

            self.categorical_features.append(temp_array)

            self.labels.append(df[label][i])

    def get_train_categorical_features(self):
        return numpy.array(self.categorical_features)[:int(self.ds_size * self.train_split)]

    def get_train_numerical_features(self):
        return numpy.array(self.numerical_features)[:int(self.ds_size * self.train_split)]

    def get_train_labels(self):
        return numpy.array(self.labels)[:int(self.ds_size * self.train_split)]

    def get_test_categorical_features(self):
        return numpy.array(self.categorical_features)[int(self.ds_size * self.train_split):]

    def get_test_numerical_features(self):
        return numpy.array(self.numerical_features)[int(self.ds_size * self.train_split):]

    def get_test_labels(self):
        return numpy.array(self.labels)[int(self.ds_size * self.train_split):]
def determine_features(df):
    cat_features = []
    num_features = []
    for feature in df.columns:
        if ('float' in str(df[feature].dtype)) or ('int' in str(df[feature].dtype)):
                num_features.append(feature)
        else:
            cat_features.append(feature)

    return cat_features, num_features


def load_csv2df(file_path, label_column):
    df = pd.read_csv(file_path)
    columns_data = determine_features(df)
    return CustomDataFrame(df, columns_data[1] , columns_data[0] ,label_column)