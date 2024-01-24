import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import tt_ind_solve_power
from tqdm import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

class DurationEstimatorInterface:
    """
    A base class for estimating the duration and sample size required for statistical tests to achieve a desired power.

    Attributes
    ----------  
    uplift : float
        The expected effect size or uplift.
    daily_size_per_sample : int
        The number of observations per sample on a daily basis.  
    alpha : float
        The significance level used in the hypothesis test.
    power_threshold : float
        The desired power of the test.
    n_resamples : int
        The number of resampling iterations.
    random_state : int
        Seed for the random number generator.
    """
    
    def __init__(self, uplift, daily_size_per_sample, alpha, power, n_resamples, random_state):
        """
        Constructor for DurationEstimatorInterface.
        """
        self.alpha = alpha
        self.power_threshold = power
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.daily_size_per_sample = daily_size_per_sample
        self.uplift = uplift
        
    def __setattr__(self, key, value):
        """
        Custom attribute setter for uplift adjusting.
    
        Parameters
        ----------
        key : str
            The name of the attribute to set.
        value : various
            The value to be assigned to the attribute.
        """
        if key == 'uplift':
            self.__dict__[key] = value + 1
        else:
            super().__setattr__(key, value)
            
    def compute_size(self, progress_bar=True):
        """
        Computes the total sample size and number of days required to achieve the specified power.

        Parameters
        ----------
        progress_bar : bool, default=True
            Whether to display a progress bar during computation.

        Returns
        -------
        dict
            Dictionary containing total sample size ('total_size'), individual sample size ('sample_size'), 
            number of days ('days'), and achieved power ('power').
        """
        np.random.seed(self.random_state)
        power = 0
        days = 0
        cumulative_sample_size = 0
        while power < self.power_threshold:
            cumulative_sample_size += self.daily_size_per_sample
            pvalues = self._compute_pvalues(sample_size=cumulative_sample_size, progress_bar=progress_bar)
            days += 1
            power = (pvalues < self.alpha).mean()
        return {'total_size':cumulative_sample_size*2,'sample_size':cumulative_sample_size, 'days':days, 'power': power} 
    
    def _compute_pvalues():
        """
        Abstract method to be implemented in subclasses. Used for computing simulations.
        """
        raise NotImplementedError("Subclasses should implement this!")
        
class ProportionSizeEstim(DurationEstimatorInterface):
    """
    A class for estimating sample size and duration based on proportion metrics, extending the DurationEstimatorInterface.

    Attributes
    ----------
    cr_baseline : float
        The baseline conversion rate.
    uplift : float
        The expected effect size or uplift.
    daily_size_per_sample : int
        The number of observations per sample on a daily basis.  
    alpha : float
        The significance level used in the hypothesis test.
    power_threshold : float
        The desired power of the test.
    n_resamples : int
        The number of resampling iterations.
    random_state : int
        Seed for the random number generator.
    """
    
    def __init__(self, cr_baseline, uplift, daily_size_per_sample, alpha=0.05, power=0.8, n_resamples=10_000, random_state=None):
        """
        Constructor for DurationEstimatorInterface.
        """
        super().__init__(uplift=uplift, daily_size_per_sample=daily_size_per_sample, alpha=alpha, 
                         power=power, n_resamples=n_resamples, random_state=random_state)
        if (cr_baseline < 0 or cr_baseline > 1):
            raise ValueError('Conversion Rate must take value between 0 and 1')
        self.cr_baseline = cr_baseline
        
    def _compute_pvalues(self, sample_size, progress_bar):
        """
        Computes p-values for the proportion test based on the difference in proportions.
    
        Parameters
        ----------
        sample_size : int
            The sample size for each group.
        progress_bar : bool
            Whether to display a progress bar during computation.
    
        Returns
        -------
        np.array
            Array of p-values from the proportion test.
        """
        p_control = self.cr_baseline
        p_test = p_control * self.uplift
        pvalues = []
        rng = tqdm(range(self.n_resamples)) if progress_bar else range(self.n_resamples)
        for i in rng:
            a_control = np.random.binomial(n=sample_size, p=p_control)
            a_test = np.random.binomial(n=sample_size, p=p_test)
            pvalues.append(proportions_ztest([a_control, a_test],[sample_size]*2)[1])
        return np.array(pvalues)
    
class TtestSizeEstim(DurationEstimatorInterface):
    """
    A class for estimating sample size and duration based on mean metrics (T-test), extending the DurationEstimatorInterface.
    
    Parameters
    ----------
    target_sample : array-like
        The target sample data for T-test analysis.
    uplift : float
        Expected effect size or uplift.
    daily_size_per_sample : int
        Number of observations per sample on a daily basis.
    alpha : float, default=0.05
        Significance level for the hypothesis test.
    power : float, default=0.8
        Desired power of the test.
    n_resamples : int, default=10000
        Number of resampling iterations.
    random_state : int, optional
        Seed for the random number generator.
    """
    
    def __init__(self, target_sample, uplift, daily_size_per_sample, alpha=0.05, power=0.8, n_resamples=10_000, random_state=None):
        """
        Constructor for TtestSizeEstim.
        """
        super().__init__(uplift=uplift, daily_size_per_sample=daily_size_per_sample, alpha=alpha, 
                         power=power, n_resamples=n_resamples, random_state=random_state)
        self.target_sample = target_sample
        
    def _compute_pvalues(self, sample_size, progress_bar):
        """
        Computes p-values for the T-test based on the difference in means.
    
        Parameters
        ----------
        sample_size : int
            The sample size for each group.
        progress_bar : bool
            Whether to display a progress bar during computation.
    
        Returns
        -------
        np.array
            Array of p-values from the T-test.
        """
        pvalues = []
        rng = tqdm(range(self.n_resamples)) if progress_bar else range(self.n_resamples)
        for i in rng:
            sample_data = np.random.choice(self.target_sample, size=sample_size*2, replace=True)
            a,b = sample_data[:sample_size], sample_data[sample_size:] * self.uplift
            pvalues.append(st.ttest_ind(a,b).pvalue)
        return np.array(pvalues)
    
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
    return tt_ind_solve_power(effect_size=e,alpha=1-(1-alpha)**(1/n_comparsion),power=power)*groups

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
    return tt_ind_solve_power(effect_size=e, alpha=1-(1-alpha)**(1/n_comparsion), power=power)*groups

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
    upper_bound = (1 - confidence_level) / 2 
    n = (st.norm.ppf(upper_bound) * sigma / d)** 2
    return int(n)

def proportion_1samp_size(p, d, confidence_level=0.05):
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
    upper_bound = (1 - confidence_level) / 2 
    n = (st.norm.ppf(upper_bound) / d)**2 * p * (1-p)
    return int(n)

class TestAnalyzer:
    """
    A class for evaluating the applicability of a statistical method to a specific distribution.

    This class is designed to assess whether a given statistical test is appropriate for a particular distribution, 
    especially in scenarios where the test assumptions (such as normality for instance) may not hold. 

    Attributes
    ----------
    n_resamples : int
        The number of resampling iterations to perform.
    random_state : int, optional
        Seed for the random number generator.
    alpha : float
        Significance level for hypothesis testing.
    func : callable
        The statistical test function to apply on each resample.

    Methods
    -------
    resample(sample, progress_bar=True)
        Performs resampling on the provided sample and applies the test function to assess its suitability.
    compute_fpr(weighted=False)
        Computes the false positive rate (FPR) from the stored p-values, indicating test suitability.
    perform_chisquare(bins=None)
        Performs a chi-square test to evaluate the uniformity of the p-values distribution.
    get_charts(figsize=(8,6))
        Generates and displays chart for the test statistics and p-values, providing visual assessment of test suitability.
    """
    
    def __init__(self, func, alpha=0.05, n_resamples=10_000, random_state=None):
        """
        Constructor for DurationEstimatorInterface.
        """
        self.func = func
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.alpha = alpha
        
    def resample(self, sample, progress_bar=True):
        """
        Performs resampling on the provided sample for suitability analysis of the test function.

        Parameters
        ----------
        sample : array-like
            The data sample representing the distribution for the analysis.
        progress_bar : bool, default=True
            Whether to display a progress bar during resampling.

        Notes
        -----
        The method conducts resampling with replacement and applies the test function to each resample. 
        It is used to assess whether the distribution of p-values is uniform, indicating the test's suitability 
        for the given distribution.
        """
        
        np.random.seed(self.random_state)
        sample = np.asarray(sample)
        pvalues = []
        size = sample.shape[0]
        size_per_sample = int(size / 2)
        rng = tqdm(range(self.n_resamples)) if progress_bar else range(self.n_resamples)
        for i in rng:
                resampled_data = np.random.choice(sample, size=size, replace=True)
                a,b = resampled_data[:size_per_sample], resampled_data[size_per_sample:] 
                stat_result = self.func(a,b)
                pvalues.append(stat_result)
        self.pvalues = np.array(pvalues)
        
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
            The number of bins to use in the chi-square test.

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
    
    def get_charts(self, figsize=(8,6)):
        """
        Generates and displays chart for the distribution of test p-values.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Size of the figure to display.
        bins : int, default=20
            The number of bins for the histogram.
        """
        with sns.axes_style("whitegrid"): 
            plt.figure(figsize=figsize)
            plt.plot([0, 1], [0, 1], linestyle='dashed', color='black', linewidth=2)
            plt.vlines(x=0.05,ymin=0,ymax=1,linestyle='dotted', color='black', linewidth=2) 
            plt.plot(np.array(sorted(self.pvalues)), np.array(sorted(np.linspace(0,1,self.n_resamples))))
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
