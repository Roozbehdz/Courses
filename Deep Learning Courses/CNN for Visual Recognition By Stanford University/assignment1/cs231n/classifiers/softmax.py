from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in range(num_train):
        p = []
        score = X[i].dot(W)
        p = stable_softmax(score)
        loss -= np.log(p[y[i]])
        
        for j in range(num_classes):
          dW[:, j] += (p[j] - (j == y[i])) * X[i]
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.

    # Add regularization
    loss += reg * np.sum(W * W)
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    f = X.dot(W)
    f -= np.matrix(np.max(f, axis=1)).T
      
    term1 = -f[np.arange(num_train), y]
    sum_j = np.sum(np.exp(f), axis=1)
    term2 = np.log(sum_j)
    loss = term1 + term2
    loss = np.sum(loss)
    loss /= num_train 
    loss += 0.5 * reg * np.sum(W * W)
    
    coef = np.exp(f) / np.matrix(sum_j).T
    coef[np.arange(num_train),y] -= 1
    dW = X.T.dot(coef)
    dW /= num_train
    dW += reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
