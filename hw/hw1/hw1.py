import torch
import hw1_utils as utils
import matplotlib.pyplot as plt

"""
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
"""


# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    n = X.size(0)
    X = torch.cat((torch.ones(n, 1), X), dim=1)
    w = torch.zeros(X.size(1))
    for _ in range(num_iter):
        w -= lrate / n * X.T @ (X @ w - Y)

    return w


def linear_normal(X, Y):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    n = X.size(0)
    X = torch.cat((torch.ones(n, 1), X), dim=1)
    w = torch.pinverse(X) @ Y
    return w


def plot_linear():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    rX, Y = utils.load_reg_data()
    w = linear_normal(rX, Y)
    bX = torch.cat((torch.ones(rX.size(0), 1), rX), dim=1)
    pY = bX @ w
    
    
    plt.scatter(bX.cpu()[:,1],Y.cpu())
    plt.plot(bX.cpu()[:,1],pY.cpu(), "r-")
   
    return plt.show()

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    """
    n = X.size(0)
    d = X.size(1)
    
    X = torch.cat((torch.ones(n, 1), X), dim=1) 
    
   
    features = []
    for i in range(1, d + 1):
        for j in range(i, d + 1):
            product = (X[:, i] * X[:, j]).unsqueeze(1)
            features.append(product)
    featureTensor = torch.cat(features, dim=1)
    
    X_poly = torch.cat([X, featureTensor], dim=1)
    
    w = torch.zeros(X_poly.size(1), 1)
    for _ in range(num_iter):
        grad = X_poly.T @ (X_poly @ w - Y) / n
        w -= lrate * grad

    return w


def poly_normal(X, Y):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    """
    n = X.size(0)
    d = X.size(1)
    
    X = torch.cat((torch.ones(n, 1), X), dim=1) 
    
   
    features = []
    for i in range(1, d + 1):
        for j in range(i, d + 1):
            product = (X[:, i] * X[:, j]).unsqueeze(1)
            features.append(product)
    featureTensor = torch.cat(features, dim=1)
    
    X_poly = torch.cat([X, featureTensor], dim=1)
    w = torch.pinverse(X_poly) @ Y
    return w


def plot_poly():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    X,Y = utils.load_reg_data()
    w = poly_normal(X,Y)
    
    n = X.size(0)
    d = X.size(1)
    bX = torch.cat((torch.ones(n, 1), X), dim=1) 
    X = torch.cat((torch.ones(n, 1), X), dim=1) 
    
   
    features = []
    for i in range(1, d + 1):
        for j in range(i, d + 1):
            product = (X[:, i] * X[:, j]).unsqueeze(1)
            features.append(product)
    featureTensor = torch.cat(features, dim=1)
    
    X_poly = torch.cat([X, featureTensor], dim=1)
    pY = X_poly @ w
    plt.scatter(bX.cpu()[:,1],Y.cpu())
    plt.plot(X_poly.cpu()[:,1],pY.cpu(), "r-")
   
    return plt.show()


def poly_xor():
    """
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    """
    pass


# Problem 4
def logistic(X, Y, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    """
    pass


def logistic_vs_ols():
    """
    Returns:
        Figure: the figure plotted with matplotlib
    """
    pass


# Problem 5
def cross_entropy(X, Y, k, lrate=0.01, num_iter=1000):
    """
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        k (int): the number of classes

    Returns:
        d x k FloatTensor: the parameters w
    """
    pass


def get_ntp_weights(n, embedding_dim=10):
    """
    Arguments:
        n (int): the context size
        embedding_dim (int): the size of the random embeddings

    Returns:
        d x k FloatTensor: the parameters w
    """
    pass


def generate_text(w, n, num_tokens, embedding_dim=10, context="once upon a time"):
    """
    Arguments:
        w (d x k FloatTensor): the parameters
        n (int): the context size
        num_tokens (int): the number of additional tokens to generate
        embedding_dim (int): the size of the random embeddings
        context (str): the initial string provided to the model

    Returns:
        String: a string containing the initial context and all generated words, with each word separated by a space
    """
    pass
