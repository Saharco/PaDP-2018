import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input x

     """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    return sigmoid(x) * (1 - sigmoid(x))


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of xavier initialized np arrays weight matrices

    """

    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of zero np arrays weight matrices

    """
    return [np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of zero np arrays bias matrices

    """
    return [np.zeros(i) for i in sizes]


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    result = []
    for i in range(int(np.ceil(len(data) / batch_size))):
        data_batch = np.copy(data[i * batch_size : (i+1) * batch_size])
        label_batch = np.copy(labels[i * batch_size: (i + 1) * batch_size])
        result.append((data_batch, label_batch))
    return result


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    return [list1[i] + list2[i] for i in range(len(list1))]


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


