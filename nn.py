"""This module defines different neural networks"""


class Perceptron:
    """defines a single perceptron"""

    def __init__(self, weight, bias, nlf):
        self.weight = weight
        self.bias = bias
        self.nlf = nlf

    def update_weight(self, weight):
        """updates a percpetrons weight"""
        self.weight = weight

    def update_bias(self, bias):
        """updates a perceptrons bias"""
        self.bias = bias

    def get_weight(self):
        """returns a perceptrons weight"""
        return self.weight

    def get_bias(self):
        """returns a perceptrons bias"""
        return self.bias


class FFNNLayer:
    """defines a single layer of perceptrons in a fullforward neural network"""

    def __init__(self, height, nlf):
        self.height = height
        self.perceptrons = [Perceptron(1, 1, nlf) for _ in range(self.height)]

    def get_height(self):
        """returns the height of the layer"""
        return self.height


class FFNN:
    """defines a fullforward neural network"""

    def __init__(self, layers, heights, nlf):
        self.layers = layers
        layers = [FFNNLayer(heights[i], nlf) for i in range(layers)]

    def get_layers(self):
        """returns the number of layers of the neural network"""
        return self.layers

    def get_height(self, layer):
        """returns the height of the specified layer"""
        assert layer < self.layers
        return self.layers[layer].get_height()
