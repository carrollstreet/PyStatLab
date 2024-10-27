import time
import datetime
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

class ParallelResampler:
    """
    A class for parallel resampling and data processing using multiprocessing.
    Allows resampling and mapping of functions over iterations with parallel execution.

    Attributes:
    ----------
    n_resamples : int
        Number of resamples to generate.
    n_jobs : int
        Number of parallel jobs to run. If `n_jobs=1`, tasks run sequentially.
    random_state : int
        Seed for random number generation to ensure reproducibility.
    progress_bar : bool
        Indicates whether to display a progress bar (`tqdm`) during processing.
    time_start : float
        Timestamp when the instance is created for tracking elapsed time.
    """

    def __init__(self, n_resamples, n_jobs, random_state, progress_bar):
        """
        Initializes the ParallelResampler with specified parameters.

        Parameters:
        ----------
        n_resamples : int
            Number of resamples to generate.
        n_jobs : int
            Number of parallel jobs to run.
        random_state : int
            Seed for random number generation.
        progress_bar : bool
            Whether to show a progress bar during processing.
        """
        self.n_resamples = n_resamples
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.time_start = time.time()

    def _generate_seed_sequences(self):
        """
        Generates random seed sequences for each resample based on `random_state`.

        Returns:
        -------
        list of np.random.Generator
            A list of random number generators for each resample.
        """
        sq = np.random.SeedSequence(self.random_state)
        child_seeds = sq.spawn(self.n_resamples)
        return [np.random.default_rng(s) for s in child_seeds]

    def _apply_progress_bar(self, collection):
        """
        Applies a progress bar to a collection if enabled.

        Parameters:
        ----------
        collection : iterable
            The collection to which a progress bar should be applied.

        Returns:
        -------
        iterable
            The collection, possibly wrapped with a progress bar.
        """
        return tqdm(collection) if self.progress_bar and self.n_jobs == 1 else collection

    def resample(self, func, *args, **kwargs):
        """
        Applies `func` to each generated seed in parallel.

        Parameters:
        ----------
        func : callable
            Function to apply to each generated seed.
        *args, **kwargs:
            Additional arguments passed to `func`.

        Returns:
        -------
        np.ndarray
            Array of results from applying `func` for each resample.
        """
        seeds = self._generate_seed_sequences()
        rng_seeds = self._apply_progress_bar(seeds)
        data = Parallel(n_jobs=self.n_jobs)(delayed(func)(seed, *args, **kwargs) for seed in rng_seeds)
        return np.asarray(data)

    def map(self, func, iterable, *args, **kwargs):
        """
        Applies `func` to each element of `iterable` in parallel.

        Parameters:
        ----------
        func : callable
            Function to apply to each element in `iterable`.
        iterable : iterable
            The collection of items to process.
        *args, **kwargs:
            Additional arguments passed to `func`.

        Returns:
        -------
        np.ndarray
            Array of results from applying `func` to each element of `iterable`.
        """
        rng = self._apply_progress_bar(iterable)
        data = Parallel(n_jobs=self.n_jobs)(delayed(func)(i, *args, **kwargs) for i in rng)
        return np.asarray(data)

    def elapsed_time(self, return_dt=False):
        """
        Calculates the time elapsed since the instance was created.

        Parameters:
        ----------
        return_dt : bool, default False
            If `True`, returns `datetime.timedelta`, otherwise prints the elapsed time.

        Returns:
        -------
        None or datetime.timedelta
            The time as `datetime.timedelta` if `return_dt=True`, otherwise `None`.
        """
        delta = datetime.timedelta(seconds=time.time() - self.time_start)
        if return_dt:
            return delta
        else:
            print(f'Elapsed time: {str(delta)}')