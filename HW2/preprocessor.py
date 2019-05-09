import multiprocessing
import scipy as sp
from scipy.ndimage import rotate, shift
import numpy as np
import network as nt
import utils


class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, probs):
        super().__init__()
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: Queue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        self.probs = probs

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        original_shape = image.shape
        reshaped_image = image.reshape((28, 28))
        rotated = rotate(input=reshaped_image, angle=angle, reshape=False)
        return rotated.reshape(original_shape)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        image_shifted_left = np.zeros(784)
        for x in range(784):
            if 0 <= (x % 28) + dx < 28:
                image_shifted_left[x] = image[x + dx]
            else:
                image_shifted_left[x] = 0
        image_shifted_top = np.zeros(784)
        for y in range(784):
            if 0 <= (y // 28) + dy < 28:
                image_shifted_top[y] = image_shifted_left[y + 28 * dy]
            else:
                image_shifted_top[y] = 0
        return image_shifted_top

    @staticmethod
    def step_func(image, steps):
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''
        assert steps > 1
        return np.array([np.floor(x * steps) / (steps - 1) for x in image])

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        original_shape = image.shape
        reshaped_image = image.reshape((28, 28))
        result_image = np.copy(reshaped_image)
        for x, y in zip(range(28), range(28)):
            if 0 <= round(x + y * tilt) < 28:
                result_image[y][x] = reshaped_image[y][round(x + y * tilt)]
            else:
                result_image[y][x] = 0
        return np.reshape(result_image, original_shape)

    def process_image(self, image):
        '''Apply the image process functions

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        ROTATION_LIMIT = 15
        SHIFT_LIMIT = 2
        STEP_LIMIT_LOW = 4
        STEP_LIMIT_HIGH = 16
        SKEW_LIMIT = 0.1
        random_numbers = np.random.uniform(size=4)

        if random_numbers[0] < self.probs["rotate"]:
            rotation_amount = round(np.random.randint(-ROTATION_LIMIT, ROTATION_LIMIT))
            image = self.rotate(image,  rotation_amount)

        if random_numbers[1] < self.probs["shift"]:
            shift_x_amount, shift_y_amount = round(np.random.uniform(-SHIFT_LIMIT, SHIFT_LIMIT)), \
                                             round(np.random.uniform(-SHIFT_LIMIT, SHIFT_LIMIT))
            image = self.shift(image, shift_x_amount, shift_y_amount)

        if random_numbers[2] < self.probs["step"]:
            image = self.step_func(image, np.random.randint(low=STEP_LIMIT_LOW, high=STEP_LIMIT_HIGH))

        if random_numbers[3] < self.probs["skew"]:
            skew_amount = round(np.random.uniform(-SKEW_LIMIT, SKEW_LIMIT))
            skew_amount = Worker.limit_in_range(skew_amount, -SKEW_LIMIT, SKEW_LIMIT)
            image = self.skew(image, skew_amount)

        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
        '''
        while True:
            next_job = self.jobs.get()
            if next_job is None:
                break
            self.result.put((self.process_image(next_job[0]), next_job[1]))

    @staticmethod
    def limit_in_range(num, min_limit, max_limit):
        return min(max(num, min_limit), max_limit)