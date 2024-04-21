import os
import random
import math
import pandas

list = []
cats = []
result = []

dataset = {
    'X': [],
    'Y': [],
    'Z': [],
    'C': [],
    'K': []
}

for i in range(1, 10):
    i1 = random.random()
    i2 = random.random()
    i3 = random.random()

    list.append([i1, i2])
    dataset['X'].append(i1)
    dataset['Y'].append(i2)
    if i % 3 == 0:
        dataset['C'].append('A')
        dataset['K'].append('T')
        result.append(2 * i1 + i2)
        dataset['Z'].append(2 * i1 + i2)
    elif i % 3 == 1:
        cats.append([0, 1, 0])
        dataset['C'].append('B')
        dataset['K'].append('O')
        result.append(3 * i1 + 5 * i2)
        dataset['Z'].append(3 * i1 + 5 * i2)
    else:
        cats.append([0, 0, 1])
        dataset['C'].append('C')
        dataset['Z'].append(6 * i1 + 2 * i2)
        dataset['K'].append('O')
        result.append(6 * i1 + 2 * i2)

import numpy
import pandas as pd

df = pd.read_csv('data.csv')
cat_features = ['Make','Vehicle Class', 'Cylinders', 'Transmission','Fuel Type',]
num_features = ['Engine Size(L)', 'Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)']
label = 'CO2 Emissions(g/km)'



class CustomDataFrame:
    def __init__(self, dataset, numerical_features, categorical_features, label):
        import pandas as pd
        import numpy
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

    def get_categorical_features(self):
        return numpy.array(self.categorical_features)

    def get_numerical_features(self):
        return numpy.array(self.numerical_features)

    def get_labels(self):
        return numpy.array(self.labels)


my_df = CustomDataFrame(df, num_features, cat_features, label)

class LinearRegression:
    def __init__(self, dataframe):
        self.features = dataframe.get_numerical_features()
        self.categories = dataframe.get_categorical_features()
        self.labels = dataframe.get_labels()
        self.temp_weights = []
        self.epochs = []
        self.errors = []
        self.weights = numpy.random.randn(len(self.categories[0]), len(self.features[0]))

    def predict(self, dataframe = None):
        if dataframe is not None:
            pass
            #return (((dataframe.get_numerical_features() @ self.weights) * dataframe.get_categorical_features()).sum(1))
        else:
            return (((self.categories @ self.weights) * self.features).sum(1))

    def error(self):
        return sum((self.labels - self.predict()) ** 2)

    def get_weights_differentials(self, current):
        residules = (current - self.labels)
        return 2 * (numpy.expand_dims(self.categories, self.categories.ndim) * numpy.expand_dims(self.features,
                                                                                                 1) * numpy.expand_dims(
            numpy.expand_dims(residules, residules.ndim), residules.ndim + 1)).sum(0)

    def validate_step(self, steps, error_treshold):
        step = None
        step_error = None
        current_error = self.error()
        for _step in steps:
            self.backpropagate(_step)
            temp_cur_error = self.error()
            if step is None:
                step = _step
                step_error = temp_cur_error
            else:
                if step_error > temp_cur_error:
                    step = _step
                    step_error = temp_cur_error
            self.reverse_step()

        if current_error > step_error and ((current_error - step_error)/current_error) > error_treshold:
            print(((current_error - step_error)/current_error))
            return step
        else:
            return -1

    def backpropagate(self, step):
        current = self.predict()
        self.temp_weights = self.weights

        weight_chng = (self.get_weights_differentials(current) * step)

        self.weights = self.weights - weight_chng

    def reverse_step(self):
        self.weights = self.temp_weights

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.plot(self.epochs, self.errors, color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Epochs Vs Error")
        plt.legend(["Error"])
        plt.show()

    def get_R_squared(self):
        pass

    def get_F_value(self):
        pass

    def train(self, steps_count=10, error_chng_threshold = 0.01):
        steps = [1 / (10 ** (x + 1)) for x in range(steps_count)]
        ephoc = 0
        while True:
            step = self.validate_step(steps, error_chng_threshold)
            if step == -1:
                break

            self.backpropagate(step)
            self.errors.append(self.error())
            self.epochs.append(ephoc)
            ephoc += 1
            print(f"{math.sqrt(self.error()/len(df))} - ({self.error()})")

    def save_model(self, filename):
        if 'models' not in os.listdir():
            os.system('mkdir models')
        numpy.save(f'models/{filename}.npy', self.weights)

    def load_model(self, filename):
        self.weights = numpy.load(f'models/{filename}.npy')


        # while True:
        #     for step in steps:
        #         current_error = self.error()
        #         self.backpropagate(step)
        #         stepped_error = self.error()
        #
        #         if stepped_error >= current_error:
        #             if step == steps[-1]:
        #                 break
        #             else:
        #                 self.reverse_step()
        #
        #         else:
        #             print(stepped_error, current_error)
        #             print(self.error())
        #             continue

    # def error(self):
    #     return


n = LinearRegression(my_df)
# n.load_model('model0')
n.train(10)
n.save_model('model0')
# n.predict(df)
# n.visualize()
# for i in range(10):
#     n.backpropagate(0.01)


# 1 0 0 | 1 0 | 1         a1 a2 a3
# 0 1 0 | 0 1 | 0         b1 b2 b3
# 0 0 1 | 1 0 | 1         c1 c2 c3
# 		        A1 A2 A3
# 		        B1 B2 B3
#                         Q1 Q2 Q3
#
#
#
#
# 1 0 0  -> 2x + 1
# 0 1 0  -> 5x + 4
# 0 0 1  -> 9x + 3
#
#
# 2 1
# 5 4
# 9 3
#
#
# numerical -


# import numpy
#
# a = numpy.array(
#     [
#         [[2, 4]],
#         [[5, 4]]
#     ]
# )
#
# p = numpy.array(
#     [[[0], [1]],
#      [[1], [0]]]
# )
#
# k = numpy.ones((2, 2, 2))
# print(a * k * p)
