import numpy as np
from pydantic import BaseModel, validator
from typing import List,Tuple


class ConvolutionParams(BaseModel):
    ''' Defining input types for validation'''
    input_data: List[List[int]]
    kernel: List[List[int]]
    stride: Tuple[int, int] = (1,1)
    padding: Tuple[int, int] =(0,0)

  

def udf_conv2d(input_data, kernel, stride=(1, 1), padding=(0, 0)):
    ''' 
      This function performs 2D convolution of arrays.
      Cases handled :  1. Kernel size can't be bigger than data size
                       2. Stride length should be a positive non-zero value and padding must contain non-negative  integer
                       3. The stride length and padding can be different along rows and columns
                       4. The input and kernel need not be sqaure matrices
      Output size : ((input size - kernel size + 2* padding)/Stride length) + 1

    '''
    
    input_data = np.array(input_data)
    kernel = np.array(kernel)
    stride = tuple(stride)
    padding = tuple(padding)
    
    
    hd, wd = input_data.shape
    
    hk, wk = kernel.shape
   
    if (hd<hk) or (wd<wk) :                # check if kernel size is bigger than data size
        return ("Kernel size is bigger than data size")
    
    elif (stride <= (0,0)) or (padding <(0,0)):    # Check if the input padding and stride data are valid
        return ("Stride and Padding must be positive")
    
  
               
    else:
        # Apply padding to the input data 
        if padding != (0, 0):
            pad_height = padding[0]
            pad_width = padding[1]
            input_data = np.pad(input_data, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant') 
        
        output_height = (hd - hk + (2 * padding[0])) // stride[1] + 1
       
        output_width = (wd - wk + (2 * padding[1])) // stride[1] + 1
        output = np.zeros((output_height, output_width))
      
        

        for i in range(0, output_height):
            for j in range(0, output_width):
                start_row = i * stride[0]
                end_row = start_row + hk
                start_col = j * stride[1]
                end_col = start_col + wk
                output[i, j] = np.sum(input_data[start_row:end_row, start_col:end_col] * kernel)
        

        return output
    



if __name__ == '__main__':
    try: 
        print("Enter input_data (as a list of lists):")
        input_data = eval(input())

        print("Enter kernel (as a list of lists):")
        kernel = eval(input())

        print("Do you want to enter stride? (y/n)")
        enter_stride = input()
        if enter_stride.lower() == 'y':
            print("Enter stride (as a tuple of two integers):")
            stride = eval(input())
        else:
            stride = (1, 1)  # Default value

        print("Do you want to enter padding? (y/n)")
        enter_padding = input()
        if enter_padding.lower() == 'y':
            print("Enter padding (as a tuple of two integers):")
            padding = eval(input())
        else:
            padding = (0, 0)  # Default value

        params = ConvolutionParams(
            input_data=input_data,
            kernel=kernel,
            stride=stride,
            padding=padding
        )
        result = udf_conv2d(params.input_data, params.kernel, params.stride, params.padding)
        print("Convolution result:" ,'\n', result)
    except Exception as e:
        print(f"Validation error: {e}")