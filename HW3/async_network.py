from mpi4py import MPI
from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time
mpi4py.rc(initialize=False, finalize=False)


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # setting number of workers
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def wait_for(self, req_array):
        while req_array:
            for i, req in enumerate(req_array):
                if req.Test():
                    req.Wait()
                    req_array = req_array[:i] + req_array[i+1:]
                    break

    def do_worker(self, training_data):
        """ worker functionality

        Parameters
            ----------
        training_data : a tuple of data and labels to train the NN with
        """
        if self.rank - self.num_masters < self.number_of_batches % self.num_workers:
            self.number_of_batches = (self.number_of_batches // self.num_workers) + 1
        else:
            self.number_of_batches = self.number_of_batches // self.num_workers
        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)

            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                req_array = []
                for layer in range(self.num_layers):
                    curr = layer % self.num_masters
                    req_array.append(self.comm.Isend(nabla_b[layer], dest=curr, tag=layer))
                    req_array.append(self.comm.Isend(nabla_w[layer], dest=curr, tag=layer + 0.5))
                for layer in range(self.num_layers):
                    curr = layer % self.num_masters
                    req_array.append(self.comm.Irecv(self.biases[layer], source=curr, tag=layer))
                    req_array.append(self.comm.Irecv(self.weights[layer], source=curr, tag=layer + 0.5))
                # wait for all pending requests to finish.
                self.wait_for(req_array)

    def worker_done(self, worker):
        for layer in range(self.rank, self.num_layers, self.num_masters):
            status_1 = self.comm.Iprobe(source=worker, tag=layer)
            status_2 = self.comm.Iprobe(source=worker, tag=layer+0.5)
            if not (status_1 and status_2):
                return False
        return True

    def get_worker(self):
        while True:
            for worker in range(self.num_masters, self.size):
                if self.worker_done(worker):
                    return worker

    def do_master(self, validation_data):
        """ master functionality

        Parameters
            ----------
        validation_data : a tuple of data and labels to train the NN with
        """

        # setting up the layers this master does.
        nabla_w = []
        nabla_b = []

        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.empty_like(self.weights[i]))
            nabla_b.append(np.empty_like(self.biases[i]))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):
                chosen_worker = self.get_worker()

                req_array = []
                for layer, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    req_array.append(self.comm.Irecv(db, source=chosen_worker, tag=layer))
                    req_array.append(self.comm.Irecv(dw, source=chosen_worker, tag=layer + 0.5))
                self.wait_for(req_array)

                # calculate new weights and biases (of layers in charge).
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send data to all workers and wait for all pending requests
                req_array = []
                # send new values (of layers in charge)
                for layer in range(self.rank, self.num_layers, self.num_masters):
                    req_array.append(self.comm.Isend(self.biases[layer], dest=chosen_worker, tag=layer))
                    req_array.append(self.comm.Isend(self.weights[layer], dest=chosen_worker, tag=layer + 0.5))
                self.wait_for(req_array)

            self.print_progress(validation_data, epoch)

        if self.rank == 0:
            req_array = []

            for layer in range(self.num_layers):
                source = layer % self.num_masters
                if source == 0:
                    continue
                req_array.append(self.comm.Irecv(self.biases[layer], source=source, tag=layer))
                req_array.append(self.comm.Irecv(self.weights[layer], source=source, tag=layer + 0.5))
            self.wait_for(req_array)
        else:
            req_array = []
            for layer in range(self.rank, self.num_layers, self.num_masters):
                req_array.append(self.comm.Isend(self.biases[layer], dest=0, tag=layer))
                req_array.append(self.comm.Isend(self.weights[layer], dest=0, tag=layer + 0.5))
            self.wait_for(req_array)
