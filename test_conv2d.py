import numpy as np
import pytest
from Convolution2D import udf_conv2d



# def test_assert_equality():
    # input_data = np.ones((3,3),int)
                            

    # kernel = np.ones((2,2),int)
                       
    # stride = (1, 1)
    # padding = (1, 1)

    # output = udf_conv2d(input_data, kernel, stride, padding)

    # expected_output = np.array([[1, 2, 2, 1], 
                                # [2, 4, 4, 2],
                                # [2, 4, 4, 2],
                                # [1, 2, 2, 1]])

    # np.testing.assert_allclose(output, expected_output, rtol=1e-6)

# def test_assert_inequality():
    # input_data = np.array([[1, 2, 3, 4, 5],
                           # [6, 7, 8, 9, 10],
                           # [11, 12, 13, 14, 15],
                           # [16, 17, 18, 19, 20]])
    # kernel = np.array([[0.5, 1],
                       # [1, 0.5]])
    # stride = (2, 2)
    # padding = (1, 1)

    # output = udf_conv2d(input_data, kernel, stride, padding)

    # expected_output = np.array([[33.6, 40.5],  # Intentional difference
                                # [73.5, 80.5]])

    # with pytest.raises(AssertionError):
        # np.testing.assert_allclose(output, expected_output, rtol=1e-6)

@pytest.fixture
def expected_output_shape():
    input_data = np.ones((3,3),int)
                          

    kernel = np.ones((2,2),int)
                       
    stride = (1, 1)
    padding = (1, 1)


    expected_output_height = ((input_data.shape[0] + 2 * padding[0] - kernel.shape[0]) // stride[0]) + 1
    expected_output_width = ((input_data.shape[1] + 2 * padding[1] - kernel.shape[1]) // stride[1]) + 1

    return expected_output_height, expected_output_width

def test_conv2d_output_shape(expected_output_shape):
    input_data = np.ones((3,3),int)
    kernel = np.ones((2,2),int)
                       
    stride = (1, 1)
    padding = (1, 1)

    result = udf_conv2d(input_data, kernel, stride, padding)
    expected_output_height, expected_output_width = expected_output_shape

    assert result.shape == (expected_output_height, expected_output_width),\
        f"Expected shape: ({expected_output_height}, {expected_output_width}), but got: {result.shape}"


def test_conv2d_result_equality(expected_output_shape):
    input_data = np.ones((3,3),int)
    kernel = np.array([[1, 0],
                       [0, -1]])

    kernel = np.ones((2,2),int)
                       
    stride = (1, 1)
    padding = (1, 1)

    expected_output_height, expected_output_width = expected_output_shape

    result = udf_conv2d(input_data, kernel, stride, padding)
    expected_result = np.array([[1, 2, 2, 1], 
                                [2, 4, 4, 2],
                                [2, 4, 4, 2],
                                [1, 2, 2, 1]])

    assert np.allclose(result, expected_result), "Convolution result does not match the expected values."

if __name__ == '__main__':
    pytest.main()
