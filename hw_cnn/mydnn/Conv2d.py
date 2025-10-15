import numpy as np
from resampling import *

################### Conv2d_stride1 and Conv2d Class Components ###################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input for convolution
# Z:    type: Matrix of N x C_out x H_out x W_out;  features after conv2d with stride 1
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;              bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K X K;       how changes in weights affect loss
# dLdb: type: Matrix of C_out x 1;                  how changes in bias affect loss
######################################################################################

class Conv2d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        self.output_height = input_height - self.kernel_size + 1
        self.output_width = input_width - self.kernel_size + 1   

        Z = np.zeros((batch_size, self.out_channels, self.output_height, self.output_width), dtype=A.dtype)  
        for n in range(batch_size):
            for co in range(self.out_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        Z[n, co, i, j] = (
                            np.sum(
                                self.W[co, :, :, :]
                                * A[n, :, i : i + self.kernel_size, j : j + self.kernel_size]
                            )
                            + self.b[co]
                        )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, input_height, input_width = self.A.shape
        self.dLdW = np.zeros_like(self.W)                               
        for co in range(self.out_channels):
            for ci in range(self.in_channels):
                for ki in range(self.kernel_size):
                    for kj in range(self.kernel_size):
                        self.dLdW[co, ci, ki, kj] = np.sum(
                            dLdZ[:, co, :, :] *
                            self.A[:, ci, ki : ki + self.output_height, kj : kj + self.output_width]
                        )
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))                        

        dLdA = np.zeros_like(self.A)                                    
        for n in range(batch_size):
            for co in range(self.out_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        grad = dLdZ[n, co, i, j]
                        dLdA[n, :, i : i + self.kernel_size, j : j + self.kernel_size] += (
                            grad * self.W[co, :, :, :]
                        )

        
        return dLdA


class Conv2d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)  


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="constant")


        # Call Conv2d_stride1
        Z_stride1 = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

       # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ) 

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)  

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad : -self.pad, self.pad : -self.pad] if self.pad > 0 else dLdA  



        return dLdA
