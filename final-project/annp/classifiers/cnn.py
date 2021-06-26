from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C, H, W = input_dim
        after_pooling_dim = int(H * W / 4.0)
        self.params["W1"] = np.random.normal(0.0, weight_scale, (num_filters,C,filter_size,filter_size))
        self.params["b1"] = np.zeros(num_filters)
        self.params["W2"] = np.random.normal(0.0, weight_scale, (after_pooling_dim*num_filters, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim, dtype=float)
        self.params["W3"] = np.random.normal(0.0, weight_scale, (hidden_dim,num_classes))
        self.params["b3"] = np.zeros(num_classes, dtype=float)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in annp/fast_layers.py and  #
        # annp/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out1flat = out1.reshape(out1.shape[0], -1)
        out2, cache2 = affine_relu_forward(out1flat, W2, b2)
        out3, cache3 = affine_forward(out2, W3, b3)
        scores = out3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dout3 = softmax_loss(scores, y)
        dout2, grads["W3"], grads["b3"] = affine_backward(dout3, cache3)
        dout1flat, grads["W2"], grads["b2"] = affine_relu_backward(dout2, cache2)
        dout1 = dout1flat.reshape(out1.shape)
        _, grads["W1"], grads["b1"] = conv_relu_pool_backward(dout1, cache1)
        for i in range(1,4):
            loss += 0.5 * self.reg * (self.params["W%d"%i] ** 2).sum()
            grads["W%d"%i] += self.reg * self.params["W%d"%i]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class MyNet(object):
    """
    conv - bn - relu - 2x2 max pool - affine - bn - relu - dropout - affine - bn - relu - dropout - affine - softmax
    """
    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=5,
        hidden_dim=500,
        hidden_dim1=200,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        after_pooling_dim = int(H * W / 4.0)
        self.params["W1"] = np.random.normal(0.0, weight_scale, (num_filters,C,filter_size,filter_size))
        self.params["b1"] = np.zeros(num_filters)
        self.params["W2"] = np.random.normal(0.0, weight_scale, (after_pooling_dim*num_filters, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim, dtype=float)
        self.params["W3"] = np.random.normal(0.0, weight_scale, (hidden_dim, hidden_dim1))
        self.params["b3"] = np.zeros(hidden_dim1, dtype=float)
        self.params["W4"] = np.random.normal(0.0, weight_scale, (hidden_dim1,num_classes))
        self.params["b4"] = np.zeros(num_classes, dtype=float)
        self.params["gamma1"] = np.ones(num_filters)
        self.params["beta1"] = np.zeros(num_filters)
        self.params["gamma2"] = np.ones(hidden_dim)
        self.params["beta2"] = np.zeros(hidden_dim)
        self.params["gamma3"] = np.ones(hidden_dim1)
        self.params["beta3"] = np.zeros(hidden_dim1)


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        beta1 = self.params["beta1"]
        gamma1 = self.params["gamma1"]
        beta2 = self.params["beta2"]
        gamma2 = self.params["gamma2"]
        beta3 = self.params["beta3"]
        gamma3 = self.params["gamma3"]

        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        self.dropout_param = {'mode': 'train', 'p': 0.4}

        bn_param1 = {'mode': 'train', 'running_mean': np.zeros(beta1.shape[0]), 'running_var': np.zeros(beta1.shape[0])}
        out1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param1, pool_param)
        out1flat = out1.reshape(out1.shape[0], -1)
        bn_param2 = {'mode': 'train', 'running_mean': np.zeros(beta2.shape[0]), 'running_var': np.zeros(beta2.shape[0])}
        out2, cache2 = affine_bn_relu_forward(out1flat, W2, b2, gamma2, beta2, bn_param2)
        out2, cache_do = dropout_forward(out2, self.dropout_param)
        cache2 = (cache2, cache_do)
        bn_param3 = {'mode': 'train', 'running_mean': np.zeros(beta3.shape[0]), 'running_var': np.zeros(beta3.shape[0])}
        out3, cache3 = affine_bn_relu_forward(out2, W3, b3, gamma3, beta3, bn_param3)
        out3, cache_do = dropout_forward(out3, self.dropout_param)
        cache3 = (cache3, cache_do)
        out4, cache4 = affine_forward(out3, W4, b4)
        scores = out4

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout4 = softmax_loss(scores, y)
        dout3, grads["W4"], grads["b4"] = affine_backward(dout4, cache4)
        cache3, cache_do = cache3
        dout3 = dropout_backward(dout3, cache_do)
        dout2, grads["W3"], grads["b3"], grads["gamma3"], grads["beta3"] = affine_bn_relu_backward(dout3, cache3)
        cache2, cache_do = cache2
        dout2 = dropout_backward(dout2, cache_do)
        dout1flat, grads["W2"], grads["b2"], grads["gamma2"], grads["beta2"] = affine_bn_relu_backward(dout2, cache2)
        dout1 = dout1flat.reshape(out1.shape)
        dx, grads["W1"], grads["b1"], grads["gamma1"], grads["beta1"] = conv_bn_relu_pool_backward(dout1, cache1)

        for i in range(1,5):
            loss += 0.5 * self.reg * (self.params["W%d"%i] ** 2).sum()
            grads["W%d"%i] += self.reg * self.params["W%d"%i]
        return loss, grads