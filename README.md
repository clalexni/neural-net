# Neural Net

## Goal:
- Linear Classification with Sigmoid Regression

## Requirements:
- Python 3.6
- the program takes 4 arguments: 
  1. training_file 
  2. test_file 
  3. learning_rate 
  4. number of iterations
- run these two command lines
~~~
chmod u+x main.py
~~~
~~~
./main.py [input1] [input2] [input3] [input4]
~~~
- for example:
~~~
./main.py data/train2.dat data/test2.dat 0.3 400
~~~




## Notes:
### Description:
- fake attribute is explicitly provided in dataset

### Update Rule:
- Gradient Descent with logistic regression:<br/>
~~~
w_i = w_i + alpha * (y_k - sigmoid(wx^T)) * x_i * sigmoid'(wx^T)
~~~
