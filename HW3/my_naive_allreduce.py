import numpy as np
from mpi4py import MPI


def allreduce(send, recv, comm):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    for i in range(size):
        if i == rank:
            np.copyto(recv, send)
            for j in range(size - 1):
                temp = np.empty_like(recv)
                comm.Recv(temp, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                recv += temp
        else:
            comm.Send(send, dest=i, tag=0)
