import numpy as np

def udf_conv2d(input_data, kernel, stride=(1, 1), padding=(0, 0)):
    ''' 
      This function performs 2D convolution of arrays.
      Cases handled :  1. Kernel size can't be bigger than data size
                       2. Stride length should be a positive non-zero value and padding must contain non-negative  integer
                       3. The stride length and padding can be different along row and columns
                       4. The input and kernel need not be sqaure matrices
      Output size : ((input size - kernel size + 2* padding)/Stride length) + 1

    '''
    
    
    
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
       # print(output.shape)
        

        for i in range(0, output_height):
            for j in range(0, output_width):
                start_row = i * stride[0]
                end_row = start_row + hk
                start_col = j * stride[1]
                end_col = start_col + wk
                output[i, j] = np.sum(input_data[start_row:end_row, start_col:end_col] * kernel)
        #print(output)

        return output
    
