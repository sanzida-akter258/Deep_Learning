# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

################### Conv1d_stride1 and Conv1d Class Components ###################################
# kernel size:  K;      type scalar;        kernel size
# stride:               type: scalar;       equivalent to downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x W_in;    data input for convolution
# Z:    type: Matrix of N x C_out x W_out;  features after conv1d with stride
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;          bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;    how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K;   gradient of Loss w.r.t. weights
# dLdb: type: Matrix of C_out x 1;          gradient of Loss w.r.t. bias
###################################################################################
class Conv1d_stride1:
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
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size), dtype=A.dtype)  
        for n in range(batch_size):
            for co in range(self.out_channels):
                for t in range(output_size):
                    # sum over in_channels and kernel positions
                    Z[n, co, t] = np.sum(
                        self.W[co, :, :] * A[n, :, t : t + self.kernel_size]
                    ) + self.b[co]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        self.dLdb = self.dLdb = np.sum(dLdZ, axis=(0, 2))
        self.dldW = self.dLdW = np.zeros_like(self.W)
        for co in range(self.out_channels):
            for ci in range(self.in_channels):
                for k in range(self.kernel_size):
                    self.dLdW[co, ci, k] = np.sum(
                        dLdZ[:, co, :] * self.A[:, ci, k : k + output_size]
                    )

        dLdA = np.zeros_like(self.A)  # TODO
        for n in range(batch_size):
            for co in range(self.out_channels):
                for t in range(output_size):
                    d = dLdZ[n, co, t]
                    # add kernel contribution back to input window
                    dLdA[n, :, t : t + self.kernel_size] += d * self.W[co, :, :]

        return dLdA


class Conv1d:
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

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Calculate Z
        # Line 1: Pad with zeros
        # Line 2: Conv1d forward
        # Line 3: Downsample1d forward
        Z = self.downsample1d.forward(  
            self.conv1d_stride1.forward(
                np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode="constant")
            )
        )
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Calculate dLdA
        # Line 1: Downsample1d backward
        # Line 2: Conv1d backward
        # Line 3: Unpad
        dLdA = self.conv1d_stride1.backward(  
            self.downsample1d.backward(dLdZ)
        )
        dLdA = dLdA[:, :, self.pad : -self.pad] if self.pad > 0 else dLdA
        
        return dLdA
