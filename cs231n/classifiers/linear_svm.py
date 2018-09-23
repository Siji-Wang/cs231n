import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero，初始化导数矩阵

  # compute the loss and the gradient
  num_classes = W.shape[1]#类别的数目
  num_train = X.shape[0]#训练集的数目
  loss = 0.0#初始化损失
  for i in range(num_train):
    scores = X[i].dot(W)#W:(D, C),X:(N, D),so X[i]=(1,D)
    correct_class_score = scores[y[i]]#此处y[i]是一个int，指代class的序号
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1#support vector meachine的margin
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :].T#根据导数更新的公式
        dW[:, j] += X[i, :].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train#对损失取平均
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)#损失函数加上正则项
  dW += reg*W#更新导数矩阵dW

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = X.shape[0]# (N,C)
  num_classes = W.shape[1]# C
  #这是一种特殊的索引方法，有点牛逼，意思就是np.arange(num_train)的序列每一个位置对应原始scores的第一维度，而y则代表第二个维度始终取[1,2,3,...,y]中
  #对应位置上的值，最终的结果则是一个和np.arange(num_train)或者y序列一样规模的numpy数组，(1,N)
  scores_correct = scores[np.arange(num_train), y]   
  scores_correct = np.reshape(scores_correct, (num_train, 1))  #reshape一下(N,1)
  margins = scores - scores_correct + 1.0     # (N,C),广播法则
  margins[np.arange(num_train), y] = 0.0 #(1,N)，就是y类的那一项直接变成0
  margins[margins <= 0] = 0.0 #bool索引，根据规则，小于0的不影响svm的loss函数
  loss += np.sum(margins) / num_train#平均
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1.0
  row_sum = np.sum(margins, axis=1)#(1,N),第二个维度，就是N个加起来
  margins[np.arange(num_train), y] = -row_sum #分情况求导，具体参考公式       
  dW += np.dot(X.T, margins)/num_train + reg * W     # 更新梯度

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
