import numpy as np
from resampling import *

################### Class Components #################################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   stride
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input 
# Z:    type: Matrix of N x C_in x H_out x W_out;  features after pooling
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_in x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
######################################################################################

class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1   
        output_height = input_height - self.kernel + 1 

        Z = np.zeros((batch_size, in_channels, output_width, output_height), dtype=A.dtype)  
        for n in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[n, c, i, j] = np.max(
                            A[n, c, i:i+self.kernel, j:j+self.kernel]
                        )
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = self.A.shape[2]   
        input_height = self.A.shape[3]  

        dLdA = np.zeros_like(self.A)    
        K = self.kernel
        for n in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        window = self.A[n, c, i:i+K, j:j+K]
                        m = np.max(window)
                        mask = (window == m).astype(dLdZ.dtype)
                        # split gradient equally across ties
                        denom = np.sum(mask)
                        if denom == 0:
                            continue
                        dLdA[n, c, i:i+K, j:j+K] += (dLdZ[n, c, i, j] / denom) * mask

        return dLdA


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1    
        output_height = input_height - self.kernel + 1  

        Z = np.zeros((batch_size, in_channels, output_width, output_height), dtype=A.dtype)  
        K = self.kernel
        for n in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[n, c, i, j] = np.mean(A[n, c, i:i+K, j:j+K])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = self.A.shape[2]   
        input_height = self.A.shape[3]  

        dLdA = np.zeros_like(self.A)    
        K = self.kernel
        scale = 1.0 / (K * K)
        for n in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        dLdA[n, c, i:i+K, j:j+K] += dLdZ[n, c, i, j] * scale

        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)   
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ) 

        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
