import numpy as np
from tqdm import tqdm

class Optimizer:
    """
    Base class for optimization algorithms.

    Parameters
    ----------
    n_iterations : int
        Number of iterations to run the optimization algorithm.
    batch_size : int
        Size of the batch used for the mini-batch gradient descent.

    Methods
    -------
    _gradient(func, x0, X, y):
        Computes the gradient of the function 'func' at the point 'x0'.
    _step(func, x0, X, y):
        Abstract method to compute the optimization step. Should be implemented by subclasses.
    optimize(func, x0, X, y, progress_bar=True):
        Runs the optimization algorithm on the function 'func'.
    """
    def __init__(self, n_iterations, batch_size):
        """
        Constructor for Optimizer.
        """
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        
    def _gradient(self, func, x0, X, y): 
        """
        Computes the gradient of the function 'func' at the point 'x0'.

        Parameters
        ----------
        func : function
            The function for which the gradient is to be computed.
        x0 : ndarray
            The point at which the gradient is computed.
        X, y : ndarray
            Data points and target values used for the gradient computation.

        Returns
        -------
        ndarray
            Gradient of 'func' at 'x0'.
        """

        h = np.cbrt(np.finfo(float).eps)
        grad = np.zeros_like(x0)
        for i in range(len(x0)):
            x_forward, x_backward = x0.copy(), x0.copy()
            x_forward[i] += h
            x_backward[i] -= h
            grad[i] = (func(x_forward, X, y)  - func(x_backward, X, y)) / (2 * h)
        return grad
        
    def _step(self, func, x0, X, y):
        """
        Abstract method to compute the optimization step.

        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")
    
    def optimize(self, func, x0, X, y, progress_bar=True):
        """
        Runs the optimization algorithm.

        Parameters
        ----------
        func : function
            The function to be optimized.
        x0 : ndarray
            Initial guess for the parameters.
        X, y : ndarray
            Data points and target values for the function 'func'.
        progress_bar : bool, default=True
            Whether to display a progress bar for iterations.

        Returns
        -------
        ndarray
            Optimized parameters.
        """
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        x0 = np.asarray(x0).copy()
        rng = tqdm(range(self.n_iterations)) if progress_bar else range(self.n_iterations)
        for _ in rng:
            for i in range(0, X.shape[0], self.batch_size):
                self.st = self._step(func, x0, X[i:i+self.batch_size], y[i:i+self.batch_size])
                x0 += self.st
        return x0

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Inherits from 'Optimizer'.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for the optimizer.
    n_iterations : int, default=10000
        Number of iterations for the optimizer.
    batch_size : int, optional
        Batch size for mini-batch gradient descent.

    Methods
    -------
    _step(func, x0, X, y):
        Computes the SGD step for optimization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=10000, batch_size=None):
        """
        Constructor for SGD.
        """
        super().__init__(n_iterations=n_iterations, batch_size=batch_size)
        self.learning_rate = learning_rate
        
    def _step(self, func, x0, X, y):
        """
        Computes the SGD step for optimization.

        Parameters
        ----------
        func : function
            The function to be optimized.
        x0 : ndarray
            Current guess for the parameters.
        X, y : ndarray
            Data points and target values for the function 'func'.

        Returns
        -------
        ndarray
            The step to be taken in the parameter space.
        """
        self.grad = self._gradient(func, x0, X, y)
        return - self.learning_rate * self.grad
    
class AdaGradOpt(Optimizer):
    """
    AdaGrad optimizer, an adaptive learning rate optimization algorithm.

    Inherits from 'Optimizer'.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The initial learning rate for the optimizer.
    epsilon : float, default=1e-8
        A small constant for numerical stability.
    n_iterations : int, default=10000
        Number of iterations for the optimizer.
    batch_size : int, optional
        Batch size for mini-batch gradient descent.

    Methods
    -------
    _step(func, x0, X, y):
        Computes the AdaGrad step for optimization.
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8, n_iterations=10000, batch_size=None):
        """
        Constructor for AdaGradOpt.
        """
        super().__init__(n_iterations=n_iterations, batch_size=batch_size)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G = None
        
    def _step(self, func, x0, X, y):
        """
        Computes the AdaGrad step for optimization.

        Parameters
        ----------
        func : function
            The function to be optimized.
        x0 : ndarray
            Current guess for the parameters.
        X, y : ndarray
            Data points and target values for the function 'func'.

        Returns
        -------
        ndarray
            The step to be taken in the parameter space.
        """
        if self.G is None:
            self.G = np.zeros_like(x0)
        self.grad = self._gradient(func, x0, X, y)
        self.G += self.grad**2
        return -(self.learning_rate * self.grad / (self.G**.5 + self.epsilon))
    
class AdamOpt(Optimizer):
    """
    Adam optimizer, an algorithm for first-order gradient-based optimization.

    Inherits from 'Optimizer'.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for the optimizer.
    beta_1 : float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta_2 : float, default=0.999
        The exponential decay rate for the second moment estimates.
    epsilon : float, default=1e-8
        A small constant for numerical stability.
    n_iterations : int, default=10000
        Number of iterations for the optimizer.
    batch_size : int, optional
        Batch size for mini-batch gradient descent.

    Methods
    -------
    _step(func, x0, X, y):
        Computes the Adam step for optimization.
    """
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iterations=10000, batch_size=None):
        """
        Constructor for AdamOpt.
        """
        super().__init__(n_iterations=n_iterations, batch_size=batch_size)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = None
        self.v = None
        
    def _step(self, func, x0, X, y):
        """
        Computes the Adam step for optimization.

        Parameters
        ----------
        func : function
            The function to be optimized.
        x0 : ndarray
            Current guess for the parameters.
        X, y : ndarray
            Data points and target values for the function 'func'.

        Returns
        -------
        ndarray
            The step to be taken in the parameter space.
        """
        if self.m is None or self.v is None:
            self.m = np.zeros_like(x0)
            self.v = np.zeros_like(x0)
        self.grad = self._gradient(func, x0, X, y)
        self.m = self.beta_1*self.m + (1-self.beta_1)*self.grad
        self.v = self.beta_2*self.v + (1-self.beta_2)*self.grad**2
        return - (self.learning_rate * self.m / (self.v**.5 + self.epsilon))
    
def integrate(func, start, end, n=10000, simpson=True):
"""
    Performs numerical integration of a given function over a specified interval.

    Parameters
    ----------
    func : callable
        The function to be integrated.
    start : float
        The starting point of the integration interval.
    end : float
        The ending point of the integration interval.
    args : tuple, optional
        Additional arguments to pass to the function.
    n : int, default=10000
        Number of sub-intervals to divide the interval into.
    simpson : bool, default=True
        If True, uses Simpson's rule for integration; otherwise, uses the trapezoidal rule.

    Returns
    -------
    float
        The approximate value of the integral.
    """
    h = (end-start) / n
    if simpson:
        area = 0.0
        for i in range(1, n):
            a = start + i*h
            b = start + (i+1)*h
            area += h/6 * (func(a) + 4*func((a+b)/2)+func(b))
    else:
        area = 0.5 * (func(start) + func(end)) * h
        for i in range(1, n):
            area += func(start+i*h) * h
    return area

def bisect(func, a, b, epsilon=1e-3):
    """
    Finds the root of a function in a given interval using the bisection method.

    Parameters
    ----------
    func : callable
        The function for which the root is to be found.
    a, b : float
        The starting and ending points of the interval. The function must have different signs at these points.
    epsilon : float, default=1e-3
        The precision of the root finding process.

    Returns
    -------
    float
        The approximate value of the root.

    Raises
    ------
    ValueError
        If func(a) and func(b) have the same sign.
    """
    if np.sign(func(a)) == np.sign(func(b)):
        raise ValueError('func(a) and func(b) must have different signs')
    while b - a > epsilon:
        i = (a + b) / 2
        if np.sign(func(a)) != np.sign(func(i)):
            b = i
        else:
            a = i
    return a
