# Convolution Function


## Description

This Python function implements a 2D convolution operation with support for stride and padding. It takes input data, a kernel matrix, stride values, and padding values as arguments and produces the convolution result.

## Usage

To use the convolution function in your projects, follow these steps:

1. Clone the parent folder and set the folder as your working directory. Run Convolution2D.py 

2. Provide the required input parameters:
   
    input_data = [...]  # 2D input matrix
    kernel = [...]  # 2D kernel matrix
    stride = (2, 2)  # Stride values (height, width)
    padding = (1, 1)  # Padding values (height, width)
    
    
    **Note:** `stride` and `padding` are optional parameters with default values `(1, 1)` and `(0, 0)` respectively.


## Examples

Here are some examples demonstrating how to use the convolution function:

input_data = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]]

kernel = [[1, 0],
          [0, -1]]

stride = (2, 2)

padding = (1, 1)



## Testing

The repository also includes tests using the pytest framework and parameter validation using Pydantic. The test files are as follows:

- `test_conv2d.py`: Contains pytest test cases for the convolution function.
- `validation.py`: Contains example test cases for Pydantic parameter validation.

To run the test file run  pytest and to run validation file type python validation.py in your terminal


