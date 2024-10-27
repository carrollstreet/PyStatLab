import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import tt_ind_solve_power
from tools import ParallelResampler
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from collections.abc import Iterable

class DurationEstimator:
    """
    Estimates the sample size and duration needed to achieve a desired statistical power 
    for A/B tests, handling both proportion and continuous metrics.

    Methods
    -------
    __init__(self, baseline_value, uplift, daily_nobs1, ...)
        Initializes the estimator and validates input values.
    __setattr__(self, key, value)
        Adjusts the `uplift` attribute to a multiplier.
    _compute_pvalue(self, seed, sample_size)
        Computes p-value using resampled data.
    compute_size(self, max_days=50)
        Estimates sample size and days needed to reach target power.
        Returns a dictionary with 'total_size', 'sample_size', 'days', 'power', and 'uplift'.
    """    
    def __init__(self, 
                 baseline_value,
                 uplift, 
                 daily_nobs1, 
                 is_proportion=True,
                 alpha=0.05, 
                 power=0.8, 
                 n_resamples=10000, 
                 random_state=None, 
                 n_jobs=-1,
                 progress_bar=False):
        """
        Initializes the DurationEstimator and validates input parameters.

        Parameters
        ----------
        baseline_value : float or array-like
            Baseline metric; float for proportions, iterable for continuous metrics.
        uplift : float
            Expected relative increase (e.g., 0.1 for 10%).
        daily_nobs1 : int
            Daily observations in the control group.
        is_proportion : bool, default True
            Whether `baseline_value` is a proportion.
        alpha : float, default 0.05
            Significance level.
        power : float, default 0.8
            Desired power level.
        n_resamples : int, default 10000
            Number of bootstrap resamples.
        random_state : int or None, default None
            Random seed for reproducibility.
        n_jobs : int, default -1
            Number of parallel jobs.
        progress_bar : bool, default False
            Show progress bar during resampling. Effective only when n_jobs=1
        """        
        self.alpha = alpha
        self.desired_power = power
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.daily_nobs1 = daily_nobs1
        self.progress_bar = progress_bar
        self.uplift = uplift
        self.n_jobs = n_jobs
        self.is_proportion = is_proportion
        
        if self.is_proportion:
            if not isinstance(baseline_value, float):
                raise ValueError('You must pass float value in baseline_arg if is_proportion=True')
            elif (baseline_value <= 0 or baseline_value >= 1):
                raise ValueError('Conversion Rate must take value between 0 and 1')
            else:
                self.p_control = baseline_value
                self.p_test = baseline_value * self.uplift
        if not self.is_proportion:
            if not isinstance(baseline_value, Iterable):
                    raise TypeError('You must pass iterable value if is_proportion=False')
            self.target_sample = baseline_value
            
        
    def __setattr__(self, key, value):
        """
        Custom attribute setting, adjusting `uplift` to a multiplier.
        """
        if key == 'uplift':
            self.__dict__[key] = value + 1
        else:
            super().__setattr__(key, value)

    def _compute_pvalue(self, seed, sample_size):
        """
        Calculates the p-value from resampled data for a given sample size.
        
        Parameters
        ----------
        seed : numpy.random.Generator
            Random number generator for resampling.
        sample_size : int
            Number of observations per group.

        Returns
        -------
        float
            The computed p-value.
        """
        if self.is_proportion:
            a_control = seed.binomial(n=sample_size, p=self.p_control)
            a_test = seed.binomial(n=sample_size, p=self.p_test)
            return proportions_ztest([a_control, a_test],[sample_size]*2)[1]
        else:
            sample_data = seed.choice(self.target_sample, size=sample_size*2, replace=True)
            a,b = sample_data[:sample_size], sample_data[sample_size:] * self.uplift
            return st.ttest_ind(a,b).pvalue
    
    def compute_size(self, max_days=50):
        """
        Estimates required sample size and duration to reach desired power.

        Parameters
        ----------
        max_days : int, default 50
            Maximum days to attempt for reaching power.

        Returns
        -------
        dict
            Contains 'total_size', 'sample_size', 'days', 'power', 'uplift'.
        """
        pr = ParallelResampler(n_resamples=self.n_resamples, random_state=self.random_state, n_jobs=self.n_jobs, progress_bar=self.progress_bar)
        power = 0
        days = 0
        cumulative_sample_size = 0
        while power < self.desired_power:
            cumulative_sample_size += self.daily_nobs1
            pvalues = pr.resample(self._compute_pvalue, cumulative_sample_size)
            days += 1
            power = (pvalues < self.alpha).mean()
            if days == max_days:
                print('Desired power cannot be achieved with the given arguments')
                break
        if self.n_jobs != 1:
            pr.elapsed_time()
        return {'total_size':cumulative_sample_size*2,
                'sample_size':cumulative_sample_size, 
                'days':days, 
                'power': power, 
                'uplift':round(self.uplift-1,3)} 

    
def cohens_d(*args, from_samples=True):
    """
    Calculates Cohen's d, a measure of effect size used to indicate the standardized difference between two means.

    Parameters
    ----------
    *args : variable
        Depending on the value of `from_samples`, this can be either two sample arrays or four parameters 
        (two means and two standard deviations).
    from_samples : bool, default=True
        If True, the function expects two sample arrays from which it calculates the means and pooled standard deviation.
        If False, the function expects four values: two means and two standard deviations.

    Returns
    -------
    float
        The calculated Cohen's d value.

    Raises
    ------
    ValueError
        If `from_samples` is True and not exactly two samples are passed, or if `from_samples` is False and not exactly 
        four parameters are passed.

    Notes
    -----
    Cohen's d is used to express the size of an effect in terms of the number of standard deviations. When `from_samples`
    is True, the function calculates the pooled standard deviation of two samples. When `from_samples` is False, 
    it uses the provided standard deviations to calculate the effect size.
    """
    if from_samples:
        if len(args) != 2:
            raise ValueError('You must pass only two samples')
        else:
            abs_diff = abs(np.mean(args[0]) - np.mean(args[1]))
            sd_pooled = ((np.var(args[0],ddof=1) + np.var(args[1],ddof=1)) / 2)**.5
            return abs_diff / sd_pooled
    else:
        if len(args) != 4:
            raise ValueError('You must pass four params: mu_1, mu_2, std_1, std_2')
        else:
            mu_1, mu_2, std_1, std_2 = args
            return abs(mu_1-mu_2) / ((std_1**2 + std_2**2) / 2)**.5
        
def proportion_size(p, uplift, n_comparison=1, alpha=0.05, power=0.8, groups=2):
    """
    Calculates the required sample size for detecting a given uplift in proportion, with specified significance level 
    (alpha), power, and number of comparisons.

    This function is useful for sample size determination in A/B testing or similar experiments where the goal is to 
    detect a change in proportions with a given level of statistical significance and power.

    Parameters
    ----------
    p : float
        Baseline proportion (e.g., conversion rate) in the control group.
    uplift : float
        Expected relative increase in proportion in the test group compared to the control group.
    n_comparison : int, default=1
        Number of pairwise comparisons. For multiple comparisons, the alpha error is adjusted using the Šidák correction.
    alpha : float, default=0.05
        Significance level for the hypothesis test.
    power : float, default=0.8
        Desired power of the test.
    groups : int, default=2
        Number of groups in the experiment (e.g., 2 for a standard A/B test).

    Returns
    -------
    float
        Calculated sample size per group to achieve the desired power and significance level.

    Notes
    -----
    The Šidák correction is applied for adjusting the alpha error in the case of multiple comparisons. The function 
    allows specifying the desired significance level (alpha) and power, providing flexibility for different research 
    designs. The effect size is calculated based on the expected uplift and the baseline proportion.
    """
    e = proportion_effectsize(p,p*(uplift+1))
    return tt_ind_solve_power(effect_size=e,alpha=1-(1-alpha)**(1/n_comparison),power=power)*groups

def ttest_size(avg, std, uplift, n_comparison=1, alpha=0.05, power=0.8, groups=2):
    """
    Calculates the required sample size for detecting a given uplift in means using an independent two-sample t-test, 
    with specified significance level (alpha), power, and number of comparisons.

    This function is designed for sample size determination in experiments like A/B testing where the goal is to detect 
    a change in means with a given level of statistical significance and power.

    Parameters
    ----------
    avg : float
        Average (mean) of the baseline group.
    std : float
        Standard deviation of the baseline group.
    uplift : float
        Expected relative increase (uplift) in the mean for the test group compared to the control group.
    n_comparison : int, default=1
        Number of pairwise comparisons. For multiple comparisons, the alpha error is adjusted using the Šidák correction.
    alpha : float, default=0.05
        Significance level for the hypothesis test.
    power : float, default=0.8
        Desired power of the test.
    groups : int, default=2
        Number of groups in the experiment (e.g., 2 for a standard A/B test).

    Returns
    -------
    float
        Calculated sample size per group to achieve the desired power and significance level.

    Notes
    -----
    The function uses the Šidák correction to adjust the alpha error for multiple comparisons. It calculates the 
    effect size based on the expected uplift and the standard deviation of the baseline group. This approach is suitable 
    for studies where the primary outcome is a continuous variable, and the objective is to compare means between two groups.
    """
    e = avg * uplift / std
    return tt_ind_solve_power(effect_size=e, alpha=1-(1-alpha)**(1/n_comparison), power=power)*groups

def expected_proportion(effect_size, proportion_1):
    """
    Calculates the expected proportion for a second group based on the effect size and the proportion of a first group.

    This function is useful in scenarios such as A/B testing where you want to estimate the expected proportion in a test 
    group given a known proportion in a control group and an anticipated effect size.

    Parameters
    ----------
    effect_size : float
        The anticipated effect size. This is a measure of the magnitude of the difference between groups.
    proportion_1 : float
        The proportion in the first group (e.g., conversion rate in the control group).

    Returns
    -------
    dict
        A dictionary containing:
        - 'proportion_2': The calculated proportion in the second group (e.g., expected conversion rate in the test group).
        - 'uplift': The relative change (uplift) in proportion from the first group to the second.

    Notes
    -----
    The function uses a transformation method to estimate the expected proportion in the second group. It first converts 
    the proportion to an angle using the arcsine square root transformation, then adjusts it by the effect size, and 
    finally transforms it back. This approach is particularly useful in cases where the proportions are subject to 
    constraints (e.g., between 0 and 1) and the effect size is expected to lead to proportion changes within these bounds.
    """
    proportion_2 = np.sin(np.arcsin(proportion_1**.5) + effect_size/2)**2
    uplift = (proportion_2 - proportion_1) / proportion_1
    return {'proportion_2':proportion_2, 'uplift':uplift}

def fixed_power(args=(), nobs1=None, alpha=0.05, ratio=1, proportion=False):
    """
    Computes the statistical power of a two-sample t-test or a test of proportions.

    Parameters:
    -----------
    args : tuple
        A tuple of two arrays or two single values.
        - If `proportion` is set to `True`, `args` should be two proportion values representing
          the proportions of two groups.
        - If `proportion` is set to `False`, `args` should be a tuple of two arrays:
          - The first array should contain the means of the two groups.
          - The second array should contain the standard deviations of the two groups.

    nobs1 : int
        The number of observations (sample size) in the first group.
        This parameter is required and cannot be `None`.

    alpha : float, optional, default=0.05
        The significance level of the test. It represents the probability of a Type I error,
        i.e., rejecting the null hypothesis when it is true. Common values are 0.05, 0.01, etc.

    ratio : float, optional, default=1
        The ratio of the sample sizes of the second group relative to the first group.
        For example, if `ratio = 2`, the second group will have twice as many observations as the first group.

    proportion : bool, optional, default=False
        Indicates whether the input values in `args` represent proportions.
        - If set to `True`, the function computes the power for a two-proportion z-test.
        - If set to `False`, the function computes the power for a two-sample t-test
          based on the means and standard deviations provided in `args`.

    Returns:
    --------
    power : float
        The computed power of the statistical test.

    Raises:
    -------
    TypeError
        If `nobs1` is not provided or set to `None`.
    ValueError
        - If `args` does not contain exactly two elements.
        - If `proportion` is `True` and the provided `args` are not valid proportion values.
        - If `proportion` is `False` and the provided `args` are not valid mean and standard deviation arrays.

    Examples:
    ---------
    # Example 1: Compute power for two means
    means = [0.5, 0.8]  # mean values of two groups
    stds = [0.1, 0.1]  # standard deviations of two groups
    power = fixed_power(args=(means, stds), nobs1=30)

    # Example 2: Compute power for two proportions
    prop1, prop2 = 0.4, 0.6  # proportions of two groups
    power = fixed_power(args=(prop1, prop2), nobs1=100, proportion=True)

    This function is useful for estimating the power of a test before data collection
    or analyzing the sensitivity of an existing test design.
    """
    if not nobs1:
        raise TypeError("fixed_power() missing 1 required positional argument: 'nobs1'")
    if len(args) != 2:
        raise ValueError('You must pass two args: two proportion values if proportion is True or mean array and std array with two values in each')
    if proportion:
        effect_size = proportion_effectsize(args[0], args[1])
    else:
        effect_size = cohens_d(*args[0], *args[1], from_samples=False)
    return tt_ind_solve_power(effect_size=effect_size, alpha=alpha, nobs1=nobs1, ratio=ratio)

def normal_1samp_size(sigma, d, confidence_level=0.95):
    """
    Calculates the required sample size to estimate the mean of a normally distributed population within a desired 
    precision and confidence level.

    This function determines the sample size necessary to construct a confidence interval around the population mean 
    that is within ±d of the mean with a specified level of confidence. It is useful in scenarios where you need 
    to estimate the mean of a population with a certain precision.

    Parameters
    ----------
    sigma : float
        The standard deviation of the population.
    d : float
        The desired precision level (the margin of error from the population mean).
    confidence_level : float, optional
        The desired confidence level for the interval (default is 0.95).

    Returns
    -------
    int
        The calculated sample size required to achieve the specified precision and confidence level.

    Notes
    -----
    The function uses the standard normal distribution to determine the z-score corresponding to the given confidence level. 
    The calculated sample size will allow constructing a confidence interval around the population mean with a width of 
    2d (±d from the mean) at the specified confidence level. This approach is common in preliminary research and study 
    planning where the goal is to determine an adequate sample size for accurate estimation of the population mean.
    """
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)
    n = (z_score * sigma / d)** 2
    return int(np.ceil(n))

def proportion_1samp_size(p, d, confidence_level=0.95):
    """
    Calculates the required sample size to estimate the proportion of a binary outcome in a population within a desired 
    precision and confidence level.

    This function is useful for determining the sample size needed to estimate a population proportion (such as a 
    conversion rate) with a specified level of precision (margin of error) and confidence.

    Parameters
    ----------
    p : float
        The estimated proportion in the population (e.g., 0.5 if no prior estimate is available).
    d : float
        The desired precision level (the margin of error from the population proportion).
    confidence_level : float, optional
        The desired confidence level for the interval (default is 0.95).

    Returns
    -------
    int
        The calculated sample size required to achieve the specified precision and confidence level.

    Notes
    -----
    The function calculates the sample size needed to construct a confidence interval for a population proportion that 
    is within ±d of the estimated proportion with a specified level of confidence. It uses the standard normal distribution 
    to determine the z-score corresponding to the given confidence level. The formula assumes a simple random sample and 
    is most accurate when the sample size is small relative to the population size.
    """
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)
    n = (z_score / d)**2 * p * (1-p)
    return int(np.ceil(n))

class TestAnalyzer:
    """
    A class for evaluating the applicability of a statistical method to a specific distribution.

    This class is designed to assess whether a given statistical test is appropriate for a particular distribution, 
    especially in scenarios where the test assumptions (such as normality) may not hold. 
    """

    def __init__(self, func, alpha=0.05, n_resamples=10000, n_jobs=-1, random_state=None, progress_bar=False):
        """
        Initializes the TestAnalyzer with the given parameters.

        Parameters
        ----------
        func : callable
            The statistical test function to be applied to each resample.
        alpha : float, default=0.05
            The significance level for hypothesis testing.
        n_resamples : int, default=10000
            The number of resampling iterations to perform.
        n_jobs : int, default=-1
            The number of parallel jobs to use for resampling. -1 means using all available processors.
        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducibility.
        progress_bar : bool, default=False
            Whether to display a progress bar during resampling.
            Note: The progress bar is only displayed if `n_jobs` is set to 1.
        """
        self.func = func
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar
        
    def resample(self, sample):
        """
        Performs resampling on the provided sample for suitability analysis of the test function.

        Parameters
        ----------
        sample : array-like
            The data sample representing the distribution for the analysis.

        Notes
        -----
        The method conducts resampling with replacement and applies the test function to each resample. 
        It is used to assess whether the distribution of p-values is uniform, indicating the test's suitability 
        for the given distribution.
        """
        pr = ParallelResampler(
            n_resamples=self.n_resamples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            progress_bar=self.progress_bar
        )
        
        sample = np.asarray(sample)
        size = sample.shape[0]
        size_per_sample = int(size / 2)
        
        def _resample_func(seed):
            resampled_data = seed.choice(sample, size=size, replace=True)
            a, b = resampled_data[:size_per_sample], resampled_data[size_per_sample:]
            return self.func(a, b)

        self.pvalues = pr.resample(_resample_func)
        if self.n_jobs != 1:
            pr.elapsed_time()

    def compute_fpr(self, weighted=False):
        """
        Computes the false positive rate (FPR) based on the stored p-values.

        Parameters
        ----------
        weighted : bool, default=False
            If True, weights the FPR by the maximum p-value.

        Returns
        -------
        float
            The calculated false positive rate, indicating the test's suitability.
        """
        fpr = np.mean(self.pvalues < self.alpha)
        return fpr * max(self.pvalues) if weighted else fpr

    def perform_chisquare(self, bins=None):
        """
        Performs a chi-square test on the distribution of p-values.

        Parameters
        ----------
        bins : int, optional
            The number of bins to use in the chi-square test. If not provided, 
            the number of bins is automatically determined based on the range of p-values.

        Returns
        -------
        tuple
            The chi-square test statistic and the p-value.

        Notes
        -----
        This method evaluates the uniformity of the p-values distribution as an indicator of the test's suitability 
        for the given distribution.
        """
        if not bins:
            len_ = len(np.arange(0, max(self.pvalues), self.alpha))
            bins = 20 if len_ > 20 else len_
        return st.chisquare(np.histogram(self.pvalues, bins=bins)[0])
    
    def get_charts(self, figsize=(8, 6)):
        """
        Generates and displays a chart for the distribution of test p-values.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Size of the figure to display.

        Notes
        -----
        The chart includes a plot of sorted p-values against a uniform distribution line, 
        and a vertical line at the alpha threshold to help visualize the distribution of p-values 
        and the rate of significant results.
        """
        with sns.axes_style("whitegrid"): 
            plt.figure(figsize=figsize)
            plt.plot([0, 1], [0, 1], linestyle='dashed', color='black', linewidth=2)
            plt.vlines(x=0.05, ymin=0, ymax=1, linestyle='dotted', color='black', linewidth=2) 
            plt.plot(np.array(sorted(self.pvalues)), np.array(sorted(np.linspace(0, 1, self.n_resamples))))
            plt.title('P-values Distribution Estimate')
            plt.ylabel('p-value')
            plt.show()

def fwer(n_comparison, alpha=0.05):
    """
    Calculates the family-wise error rate (FWER) for multiple hypothesis testing.

    The function computes the probability of making one or more false discoveries (Type I errors) among all 
    the hypotheses when performing multiple comparisons.

    Parameters
    ----------
    n_comparison : int
        The number of independent hypothesis tests being performed.
    alpha : float, optional
        The significance level for a single comparison (default is 0.05).

    Returns
    -------
    float
        The calculated FWER, representing the probability of at least one false discovery among all the tests.

    Notes
    -----
    FWER is an important measure in multiple testing to control the overall error rate. This function uses the 
    simple Bonferroni correction method, which assumes independent or positively dependent tests. The Bonferroni 
    method is conservative, meaning it might reduce the power of tests (increasing Type II errors) in an effort 
    to control for Type I errors.
    """
    return 1 - (1 - alpha) ** n_comparison
