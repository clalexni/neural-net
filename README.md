# Neural Net

## Goal:
- Linear Classification with Sigmoid Regression

## Requirements:
- Python 3.6
- take arguments: 
  1. training_file 
  2. test_file 
  3. learning_rate 
  4. number of iterations


## Notes:
### Description:
- fake attribute is explicitly provided in dataset

### Update Rule:
- Gradient Descent with logistic regression:<br/>
~~~
w_i = w_i + alpha * (y_k - sigmoid(wx^T)) * x_i * sigmoid'(wx^T)
~~~
