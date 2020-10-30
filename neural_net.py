#!/usr/bin/env python3.6
import sys
from itertools import islice

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


if __name__ == '__main__':
  train = sys.argv[1]
  test = sys.argv[2]
  alpha = sys.argv[3]
  iteration = sys.argv[4]

  train_ds = DataSet(*parse_data(train))
  #print(train_ds.col_names)







