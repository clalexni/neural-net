#!/usr/bin/env python3.6
import sys
import numpy as np
from itertools import islice
import math

class DataSet:
  """
  dataset fields:
  d.ts          An example matrix (training set). It is basically a list of examples, 
                each example is a list of attribute values + class value
  d.col_names   A list of name (label) corresponding to the matrix's column
  """
  def __init__(self, col_names=None, ts=None):
    self.col_names = col_names
    self.ts = ts

def parse_data(f):
  """
  parse data from file
  return a list of column names and an exmaple matrix
  """
  with open(f, 'r') as file:
    data = file.readlines()
    col_names = data[0].split()
    examples = [line.split() for line in islice(data, 1, len(data))]
    return col_names, examples

def LogisticLinearLearner(dataset, alpha=0.01, iteration=1000):
  """
  [Section 18.6.4]
  print learning process for each iteration and return tuned weights
  """
  def sigmoid(x):
    """sigmoid activation function"""
    return 1/(1+math.exp(-x))

  x = [ex[0:len(ex)-1] for ex in dataset.ts]  
  w = [0 for _ in range(len(dataset.col_names)-1)]
  
  for i in range(iteration):
    k = i % len(dataset.ts)
    wx = sum([float(x[k][index]) * w[index] for index in range(len(w))])
    s = sigmoid(wx)
    ds = s * (1 - s)
    delta = float(dataset.ts[k][-1]) - s
    
    print('After iteration ', i + 1, ': ', end='')
    
    # gradient descent
    for wi in range(len(w)):
      w[wi] = w[wi] + alpha * delta * float(x[k][wi]) * ds
      print('w(', dataset.col_names[wi], ') = ', '{:.4f}'.format(w[wi]), ', ', sep='', end='')

    wx = sum([float(x[k][index]) * w[index] for index in range(len(w))])
    print('output = ', '{:.4f}'.format(sigmoid(wx)), sep='')

  return w

def accuracy_test(w, ds):
  """
  return the accuracy based on the test on the dataset ds
  parameters:
  w: list of tuned weights
  ds: test set
  """
  def sigmoid(x):
    """sigmoid activation function"""
    return 1/(1+math.exp(-x))

  x = [ex[0:len(ex)-1] for ex in ds.ts]
  count = 0

  for k in range(len(x)):
    wx = sum([float(x[k][index]) * w[index] for index in range(len(w))])
    output = 1 if sigmoid(wx) >= 0.5 else 0
    if output == int(ds.ts[k][-1]):
      count += 1

  return '{:.1%}'.format(count/len(x))


if __name__ == '__main__':
  train = sys.argv[1]
  test = sys.argv[2]
  alpha = float(sys.argv[3])
  iteration = int(sys.argv[4])

  train_ds = DataSet(*parse_data(train))
  test_ds = DataSet(*parse_data(test))

  w = LogisticLinearLearner(train_ds, alpha, iteration)

  print('\n\nAccuracy on training set (', len(train_ds.ts), ' instances): ', accuracy_test(w, train_ds), sep='')
  print('Accuracy on test set (', len(test_ds.ts), ' instances): ', accuracy_test(w, test_ds), sep='')



