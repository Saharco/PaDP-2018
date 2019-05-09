import numpy as np

""" Implementation of a ring-reduce with addition. """
def ringallreduce(send, recv, comm):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    np.copyto(recv, send)
    array_size = len(send)
    curr_seg = rank
    prev_rank = rank - 1 if rank > 0 else size - 1
    next_rank = rank + 1 if rank < size - 1 else 0
    assert array_size >= size
    for i in range(size):
        curr_batch_start = (array_size // size) * curr_seg
        curr_batch_end = (array_size // size) * (curr_seg + 1) if curr_seg != size - 1 else array_size

        prev_batch_start = (array_size // size) * (curr_seg - 1) if curr_seg != 0 else (size - 1) * (array_size // size)
        prev_batch_end = (array_size // size) * curr_seg if curr_seg != 0 else array_size

        prev_batch_size = (prev_batch_end - prev_batch_start)
        temp = np.zeros(prev_batch_size, dtype=recv.dtype)
        req_receive = comm.Irecv(temp, source=prev_rank, tag=0)
        proc_array = recv[curr_batch_start: curr_batch_end]
        req_send = comm.Isend(proc_array, dest=next_rank, tag=0)
        req_receive.Wait()
        req_send.Wait()
        recv[prev_batch_start:prev_batch_end] += temp
        curr_seg = (curr_seg - 1) % size

    curr_seg = next_rank
    for i in range(size):
        curr_batch_start = (array_size // size) * curr_seg
        curr_batch_end = (array_size // size) * (curr_seg + 1) if curr_seg != size - 1 else array_size

        prev_batch_start = (array_size // size) * (curr_seg - 1) if curr_seg != 0 else (size - 1) * (array_size // size)
        prev_batch_end = (array_size // size) * curr_seg if curr_seg != 0 else array_size
        req_recv = comm.Irecv(recv[prev_batch_start:prev_batch_end], source=prev_rank, tag=0)
        req_send = comm.Isend(recv[curr_batch_start:curr_batch_end], dest=next_rank, tag=0)
        req_send.Wait()
        req_recv.Wait()
        curr_seg = (curr_seg - 1) % size
