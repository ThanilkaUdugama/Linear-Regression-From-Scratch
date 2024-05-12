import os
import math
import os
import numpy
import pandas as pd

class LinearRegression:
    def __init__(self, dataframe):
        self.features = dataframe.get_train_numerical_features()
        self.categories = dataframe.get_train_categorical_features()

        self.eval_features = dataframe.get_test_numerical_features()
        self.eval_categories = dataframe.get_test_categorical_features()
        
        self.labels = dataframe.get_train_labels()
        self.eval_labels = dataframe.get_test_labels()

        self.temp_weights = []
        self.epochs = []
        self.errors = []
        self.weights = numpy.random.randn(len(self.categories[0]), len(self.features[0]))

    def predict(self, eval = False):
        if eval:
            return (((self.eval_categories @ self.weights) * self.eval_features).sum(1))
        else:
            return (((self.categories @ self.weights) * self.features).sum(1))

    def error(self, eval = False):
        if eval:
            return sum((self.eval_labels - self.predict(eval=True)) ** 2)
        else:
            return sum((self.labels - self.predict()) ** 2)

    def evaluate(self):
        print(f"TRAIN - {math.sqrt(self.error()/len(self.features))}")
        print(f"TEST - {math.sqrt(self.error()/len(self.eval_features))}\n-------------------")

    def get_weights_differentials(self, current):
        residules = (current - self.labels)
        return 2 * (numpy.expand_dims(self.categories, self.categories.ndim) * numpy.expand_dims(self.features,
                                                                                                 1) * numpy.expand_dims(
            numpy.expand_dims(residules, residules.ndim), residules.ndim + 1)).sum(0)

    def validate_step(self, steps, step_index):
        temp_stepindex = step_index
        prev_error = self.error()
        while True:
            self.backpropagate(steps[temp_stepindex])
            temp_cur_error = self.error()
            if temp_cur_error < prev_error:
                return temp_stepindex
            else:
                self.reverse_step()
                if temp_stepindex == len(steps):
                    return -1
                else:
                    temp_stepindex += 1



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

    def train(self, steps_count=10):
        steps = [1000 / (10 ** (x + 1)) for x in range(steps_count)]
        ephoc = 0
        step_index = 0
        while True:
            step = self.validate_step(steps, step_index)
            if step == -1:
                break
            else:
                step_index = step

            # self.backpropagate(step)
            self.errors.append(self.error())
            self.epochs.append(ephoc)
            ephoc += 1
            self.evaluate()

    def save_model(self, filename):
        if 'models' not in os.listdir():
            os.system('mkdir models')
        numpy.save(f'models/{filename}.npy', self.weights)

    def load_model(self, filename):
        self.weights = numpy.load(f'models/{filename}.npy')

