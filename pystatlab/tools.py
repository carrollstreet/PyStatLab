import time
import datetime
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

class ParallelResampler:
    """
    A utility class for performing parallel resampling tasks with optional progress bar support.

    The `ParallelResampler` class provides methods to execute a resampling function or map a function
    over an iterable in parallel using multiple cores, while optionally displaying a progress bar.
    It also keeps track of the elapsed time since its instantiation.

    Parameters:
        n_resamples (int): The number of resamples or iterations to perform.
        n_jobs (int): The number of jobs (processes) to run in parallel. -1 means using all processors.
        random_state (int or None): Seed for the random number generator for reproducibility.
        progress_bar (bool): If True, displays a progress bar for the resampling operations.

    Attributes:
        n_resamples (int): The number of resamples to perform.
        n_jobs (int): The number of parallel jobs.
        progress_bar (bool): Flag to indicate if progress bar should be displayed.
        random_state (int or None): Seed for the random number generator.
        time_start (float): The timestamp when the instance was created.
    """

    def __init__(self, n_resamples, n_jobs, random_state, progress_bar):
        """
        Initializes the ParallelResampler instance with the given parameters.

        Args:
            n_resamples (int): The number of resamples or iterations to perform.
            n_jobs (int): The number of jobs (processes) to run in parallel. -1 uses all processors.
            random_state (int or None): Seed for the random number generator for reproducibility.
            progress_bar (bool): If True, displays a progress bar for the resampling operations.
        """
        self.n_resamples = n_resamples
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.time_start = time.time()

    def _generate_seed_sequences(self):
        """
        Generates a list of random number generators seeded from the initial random state.

        Uses NumPy's SeedSequence to spawn child seeds for each resample, ensuring reproducibility
        and independence between random number generators.

        Returns:
            list of numpy.random.Generator: A list of random number generator instances.
        """
        sq = np.random.SeedSequence(self.random_state)
        child_seeds = sq.spawn(self.n_resamples)
        return [np.random.default_rng(s) for s in child_seeds]

    def _apply_progress_bar(self, collection):
        """
        Wraps the given collection with a progress bar if required.

        If `progress_bar` is True and `n_jobs` is 1 (no parallelism), wraps the collection
        with `tqdm` to display a progress bar during iteration.

        Args:
            collection (iterable): The collection to potentially wrap with a progress bar.

        Returns:
            iterable: The original collection, possibly wrapped with a progress bar.
        """
        return tqdm(collection) if self.progress_bar and self.n_jobs == 1 else collection

    def resample(self, func, *args, **kwargs):
        """
        Executes the given function across multiple resamples in parallel.

        Generates a list of random number generators (one for each resample) and applies the
        function `func` to each, passing additional positional and keyword arguments.

        Args:
            func (callable): The function to apply to each resample. It should accept a random number generator
                as its first argument, followed by any additional arguments.
            *args: Variable length argument list to pass to `func`.
            **kwargs: Arbitrary keyword arguments to pass to `func`.

        Returns:
            numpy.ndarray: An array containing the results from each resample.
        """
        seeds = self._generate_seed_sequences()
        rng_seeds = self._apply_progress_bar(seeds)
        data = Parallel(n_jobs=self.n_jobs)(
            delayed(func)(seed, *args, **kwargs) for seed in rng_seeds
        )
        return np.asarray(data)

    def map(self, func, iterable, *args, **kwargs):
        """
        Applies the given function to each item in the iterable in parallel.

        Wraps the iterable with a progress bar if required and then uses joblib's Parallel to
        apply `func` to each item.

        Args:
            func (callable): The function to apply to each item in `iterable`.
            iterable (iterable): An iterable of items to process.
            *args: Variable length argument list to pass to `func`.
            **kwargs: Arbitrary keyword arguments to pass to `func`.

        Returns:
            numpy.ndarray: An array containing the results from applying `func` to each item.
        """
        rng = self._apply_progress_bar(iterable)
        data = Parallel(n_jobs=self.n_jobs)(
            delayed(func)(i, *args, **kwargs) for i in rng
        )
        return np.asarray(data)

    def elapsed_time(self, return_dt=False):
        """
        Returns or prints the elapsed time since the instance was created.

        Calculates the time difference between the current time and when the instance was
        initialized. Can either return the `timedelta` object or print a formatted string.

        Args:
            return_dt (bool, optional): If True, returns the `datetime.timedelta` object.
                If False, prints the elapsed time. Default is False.

        Returns:
            datetime.timedelta or None: The elapsed time as a `timedelta` object if `return_dt` is True,
                otherwise None.
        """
        delta = datetime.timedelta(seconds=time.time() - self.time_start)
        if return_dt:
            return delta
        else:
            print(f'Elapsed time: {str(delta)}')
