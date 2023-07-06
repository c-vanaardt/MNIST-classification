"""Docstring to come"""
import numpy as np


def softmax(vector, base=np.e):
    """returns softmax transformation of given vector and base"""
    e_x = np.exp((vector - np.max(vector)) * np.log(base))
    return e_x / e_x.sum()


def ReLU(x):
    """Returns the maximum between 0 and x"""
    return max(x, 0)

