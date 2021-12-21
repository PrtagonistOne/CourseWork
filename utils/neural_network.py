import math
import random
import numpy as np


class NeuronsHidden:
    def __init__(self, speed, dim):
        self.weights = np.array(
            [[[-round(random.uniform(-0.5, 0.5), 6)] for i in range(dim[0])] for j in range(dim[1])])
        self.speed = speed
        self.b = -round(random.uniform(-0.5, 0.5), 6)

    def activate(self, x):
        S = 0
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                S += x[i][j] * self.weights[i][j]

        S += self.b
        return 1.0 / (1.0 + math.exp(-S))

    def weights_adjustment(self, err, y):
        self.b = self.b + self.speed * err
        row, col = y.shape
        for i in range(row):
            for j in range(col):
                self.weights[i][j] += self.speed * err * y[i][j]


class NeuronsOutput:
    def __init__(self, weights, speed):
        self.weights = np.array([[-round(random.uniform(-0.5, 0.5), 6)] for i in range(weights)])
        self.speed = speed
        self.b = -round(random.uniform(-0.5, 0.5), 6)

    def activate_out(self, x):
        S = sum(x[i] * self.weights[i] for i in range(len(x)))
        S += self.b
        return 1.0 / (1.0 + math.exp(-S))

    def weights_adjustment_out(self, err, y):
        self.b = self.b + self.speed * err
        for i in range(len(self.weights)):
            self.weights[i] += self.speed * err * y[i]


class Perceptron:
    def __init__(self, amount, speed, data, corr_res, dim):
        self.HiddenNeurons = []
        self.Output = []
        self.data = data
        self.corrRes = corr_res
        self.dim = dim
        self.amount = amount
        for _ in range(10):
            self.Output.append(NeuronsOutput(self.amount, speed))
        for _ in range(self.amount):
            self.HiddenNeurons.append(NeuronsHidden(speed, dim))
        self.HiddenNeurons = np.load("hidden_weights.npy", allow_pickle=True)
        self.Output = np.load("output_weights.npy", allow_pickle=True)

    def check(self, input_data):
        results = np.zeros(self.amount, dtype=float)
        for i in range(len(results)):
            results[i] = self.HiddenNeurons[i].activate(input_data)

        final_res = np.zeros(10, dtype=float)
        for i in range(10):
            final_res[i] = self.Output[i].activate_out(results)

        return final_res

    def learn(self, epoch):
        # z, y, x = self.data.shape
        # while epoch != 0:
        #     for k in range(z):
        #         hid_res = np.zeros(z, dtype=float)
        #         out_res = np.zeros(10, dtype=float)
        #         out_delta = np.zeros(10, dtype=float)
        #         for i in range(len(self.HiddenNeurons)):
        #             hid_res[i] = self.HiddenNeurons[i].activate(self.data[k])
        #
        #         for i in range(len(self.Output)):
        #             out_res[i] = self.Output[i].activate_out(hid_res)
        #
        #         for i in range(len(self.Output)):
        #             out_delta[i] = out_res[i] * (1 - out_res[i]) * (self.corrRes[i][k] - out_res[i])
        #             self.Output[i].weights_adjustment_out(out_delta[i], hid_res)
        #
        #         for i in range(len(self.HiddenNeurons)):
        #             hid_delta = sum(
        #                 out_delta[j] * self.Output[j].weights[i]
        #                 for j in range(len(self.Output))
        #             )
        #
        #             hid_error = hid_res[i] * (1 - hid_res[i]) * hid_delta
        #             self.HiddenNeurons[i].weights_adjustment(hid_error, self.data[k])
        print('Used previous learn data')
            # print("Epoch#", epoch)
            # epoch -= 1
