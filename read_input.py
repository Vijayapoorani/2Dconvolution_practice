''' This program is to read input data, kernel, padding and stride values for the 2D convolution function'''
import numpy as np
from Convolution2D import udf_conv2d

# Getting input data - # entries greater than or less than required elements are not handled

R = int(input("Enter the number of rows for data:"))
C = int(input("Enter the number of columns for data :"))
 
 
print("Enter the entries in a single line (separated by space): ")


# User input of entries in a 
# single line separated by space


entries = list(map(int, input().split()))
 
# For printing the matrix
input_data = np.array(entries).reshape(R, C)

## Getting kernel data

R = int(input("Enter the number of rows for kernel:"))
C = int(input("Enter the number of columns for kernel :"))
 
 
print("Enter the entries in a single line (separated by space): ")


# User input of entries in a 
# single line separated by space

# entries greater than or less than required elements are not handled
entries = list(map(int, input().split()))
 
# For printing the matrix
kernel = np.array(entries).reshape(R, C)

print('\n',"input:",'\n',input_data)

print("kernel:",'\n',kernel)

# getting padding   - dimensions are not checked  - user must enter padding values
print("Enter padding values in a single line (separated by space): ")
entries = list(map(int, input().split()))
padding =tuple(entries)

# getting stride - dimensions are not checked   - user must enter stride values
print("Enter stride values in a single line (separated by space): ")
entries = list(map(int, input().split()))
stride =tuple(entries)


# performing convolution
output = udf_conv2d(input_data, kernel, stride, padding)
print("Output:")
print(output)
