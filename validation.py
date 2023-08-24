from pydantic import BaseModel, validator
from typing import List,Tuple
from Convolution2D import udf_conv2d
import numpy as np



class ConvolutionParams(BaseModel):
    input_data: List[List[int]]
    kernel: List[List[int]]
    stride: Tuple[int, int] = (1,1)
    padding: Tuple[int, int] =(0,0)

  

def perform_convolution(params: ConvolutionParams):
    input_data = np.array(params.input_data)
    kernel = np.array(params.kernel)
    stride = params.stride
    padding = params.padding

    result = udf_conv2d(input_data, kernel, stride, padding)
    return result

if __name__ == '__main__':
    try:
        params = ConvolutionParams(
            input_data=[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]],
            kernel=[[1, 0],
                    [0, -1]],
            stride=(2,2),
            padding=(1, 1)
        )

        result = perform_convolution(params)
        print(result)
    except Exception as e:
        print(f"Validation error: {e}")
