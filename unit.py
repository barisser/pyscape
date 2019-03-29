import numpy as np
import util


class Unit:
    def __init__(self, input_n, output_n, hidden_width, hidden_depth):
        self.inputs = np.zeros((1,input_n))
        self.hidden = np.zeros((hidden_width, hidden_depth))
        self.input_hidden_axons = util.random_array_range(input_n, hidden_width)
        self.hidden_hidden_axons = util.random_array_range3d(hidden_depth-1, hidden_width, hidden_width)
        self.hidden_output_axons = util.random_array_range(hidden_width, output_n)
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.input_n = input_n
        self.output_n = output_n

    def feed_inputs(self, input_data):
        self.inputs = input_data
        self.hidden[:, 0] = np.tanh(np.dot(self.inputs, self.input_hidden_axons))[0]

    def run_hidden(self):
        for i in range(0, self.hidden_depth-1):
            layer = self.hidden[:, i]
            new_values = np.dot(layer, self.hidden_hidden_axons[i, :, :])
            self.hidden[:, i+1] = np.tanh(np.add(new_values, self.hidden[:, i+1]))

    def read_outputs(self):
        return np.tanh(np.dot(self.hidden[:, self.hidden_depth-1], self.hidden_output_axons))

    def run_once(self, input_data):
        self.feed_inputs(input_data)
        self.run_hidden()
        return self.read_outputs()

    def run(self, input_stream):
        a = []
        for x in input_stream:
            a.append(self.run_once(x))
        return a

    def backpropragate(self, output_values, correct_values):
        error = np.subtract(correct_values, output_values)
        diff = backpropragate_output_axons(error)
        diff = backpropragate_hidden_axons(diff)

    def backpropragate_output_axons(self, errors):
        diff = np.dot(errors, self.hidden_output_axons.transpose())
        diff = util.tanh_derivative_array(diff)
        self.hidden_output_axons = np.add(self.hidden_output_axons, diff)
        return diff

    def backpropragate_hidden_axons(self, diff):
        for i in range(0, self.hidden_depth-1):
            d = self.hidden_depth - 1 - i
            layer = self.hidden_hidden_axons[d, :, :]
            diff = np.dot(diff, layer.transpose())
            diff = util.tanh_derivative_array(diff)
            self.hidden_hidden_axons[d, :, :] = np.add(layer, diff)

    def backpropragate_input_axons(self, diff):
        
