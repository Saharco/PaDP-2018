from network import *
from preprocessor import Worker
from multiprocessing import Queue, JoinableQueue
import re
from my_queue import MyQueue

class IPNeuralNetwork(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        self.proc_images = JoinableQueue()
        self.jobs = Queue()
        super(IPNeuralNetwork,self).__init__(*args, **kwargs)

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        self.proc_images = JoinableQueue()
        self.jobs = MyQueue()

        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', open('/proc/self/status').read())
        num_cpu = bin(int(m.group(1).replace(',', ''), 16)).count('1')

        workers = [Worker(self.jobs, self.proc_images, {"rotate": 0.7, "shift": 0.7, "step": 0.6, "skew": 0}) for _ in
                   range(num_cpu)]
        for worker in workers:
            worker.start()
        num_tasks = self.number_of_batches * self.mini_batch_size * self.epochs
        for i in range(num_tasks):
            data_set_size = len(training_data[0])
            self.jobs.put((training_data[0][i % data_set_size], training_data[1][i % data_set_size]))
        for _ in range(num_cpu):
            self.jobs.put(None)
        super().fit(training_data, validation_data)
        for worker in workers:
            worker.join()

    def create_batches(self, data, labels, batch_size):
        '''
     Override this function to return batches created by workers
     '''
        batches = []
        for k in range(self.number_of_batches):
            images = []
            labels = []
            for _ in range(self.mini_batch_size):
                image, label = self.proc_images.get()
                images.append(image)
                labels.append(label)
            batches.append((np.array(images), np.array(labels)))
        return batches
