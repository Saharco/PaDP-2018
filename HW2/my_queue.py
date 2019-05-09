from multiprocessing import Lock, Pipe


class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.lock = Lock()
        self.pusher, self.reader = Pipe()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.pusher.send(msg)

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        self.lock.acquire()
        result = self.reader.recv()
        self.lock.release()
        return result