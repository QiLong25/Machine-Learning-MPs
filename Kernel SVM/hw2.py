import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, lr: float, epochs: int):
        self.epochs = epochs
        self.lr = lr
        self.alpha = None

    def dual_loss(self, alpha: torch.tensor, x_train: torch.tensor, y_train: torch.tensor, kernel):

        N = x_train.shape[0]
        one = torch.ones((N, 1))

        ## generate K
        K = torch.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                K[i][j] = K[j][i] = kernel(x_train[i], x_train[j])

        return 1 / 2 * (alpha.view((1, N)) * y_train.view((1, N))) @ K @ (alpha.view(N, 1) * y_train.view(N, 1)).view(N, 1) - one.view((1, N)) @ alpha

    def train(self, x_train: torch.tensor, y_train: torch.tensor, kernel, c=None):
        ## Initialize alpha
        N = x_train.shape[0]
        self.alpha = torch.zeros((N, 1), requires_grad=True)

        ## Optimizer
        optimizer = optim.SGD([self.alpha], lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.dual_loss(self.alpha, x_train, y_train, kernel)
            loss.backward()

            ## update parameters
            optimizer.step()

            # self.alpha = self.alpha - self.lr * self.alpha.grad         # update alpha
            self.alpha.requires_grad = False
            self.alpha[self.alpha < 0] = 0                  # project alpha
            if type(c) != type(None):
                self.alpha[self.alpha > c] = c              # project alpha
            self.alpha.requires_grad = True

            # print("Epoch : {}, Loss : {}".format(epoch, loss))

def svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''

    model_svm = SVM(lr, num_iters)
    model_svm.train(x_train, y_train, kernel, c)
    return model_svm.alpha

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''

    with torch.no_grad():

        N = x_train.shape[0]
        M = x_test.shape[0]
        result = torch.zeros(M)

        for i in range(M):
            alpha.view((N, 1)) * y_train.view((N, 1))

            ## generate K
            K = torch.zeros((N, 1))
            for j in range(N):
                K[j] = kernel(x_test[i], x_train[j])
                result[i] = (alpha.view((N, 1)) * y_train.view((N, 1)) * K).sum()

        return result

### Main 5.3 ###

# kernel = hw2_utils.poly(degree=2)
kernel = hw2_utils.rbf(4)

def pred_fxn():

    ## Prepare data
    x_train, y_train = hw2_utils.xor_data()

    ## Hyperparameters
    lr = 0.1
    epoches = 10000

    ## Model Training
    alpha = svm_solver(x_train, y_train, lr, epoches, kernel=kernel)

    return lambda x_test: svm_predictor(alpha, x_train, y_train, x_test, kernel=kernel)

## Model Prediction
hw2_utils.svm_contour(pred_fxn=pred_fxn(), xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33)