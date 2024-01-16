import numpy as np
import scipy.stats as st

def correlation_ratio(values, categories):
    """
    Calculates the correlation ratio, a measure of the relationship between a categorical variable and a continuous variable.

    The correlation ratio (η) measures the extent to which a continuous variable changes within categories defined by a 
    categorical variable. It's a way to quantify the association between a categorical independent variable and a continuous 
    dependent variable.

    Parameters
    ----------
    values : array-like
        An array of continuous data values (dependent variable).
    categories : array-like
        An array of categories corresponding to the values (independent variable).

    Returns
    -------
    float
        The calculated correlation ratio, a value between 0 and 1, where 0 indicates no association and 1 indicates a perfect association.

    Notes
    -----
    The formula for the correlation ratio involves dividing the sum of squares between groups (SSB) by the total sum of squares (SSB + SSW), 
    where SSW is the sum of squares within groups. A higher value indicates a stronger relationship between the categorical and continuous variables.

    Reference
    ---------
    More information about the correlation ratio can be found on Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    """
    
    values = np.array(values)
    categories = np.array(categories)
    
    ssw = 0 
    ssb = 0 
    for category in set(categories):
        subgroup = values[np.where(categories == category)[0]]
        ssw += sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5


def cramers_v(rc_table, observations='raise'):
    
    """
    Calculates Cramér's V statistic, a measure of association between two categorical variables.

    Cramér's V is based on a nominal variation of the chi-squared test, and it provides a measure of the strength 
    of association between two categorical variables with values ranging from 0 (no association) to 1 (complete association).

    Parameters
    ----------
    rc_table : array-like
        A two-dimensional array (contingency table) representing the frequencies or counts of the categorical variables.
    observations : str, optional
        Handling method for small sample sizes: 'raise' to raise an error if any expected frequency is less than 5, 
        'ignore' to compute the statistic without raising an error (default is 'raise').

    Returns
    -------
    dict
        A dictionary containing 'correlation' (Cramér's V value), 'pvalue' (p-value of the chi-squared test), 
        and 'chi2' (chi-squared statistic).

    Notes
    -----
    The function optionally applies Yates' correction for continuity in case of small sample sizes. 
    Cramér's V is an appropriate measure for nominal (categorical) data and is symmetric; it does not depend 
    on the order of variables.

    Reference
    ---------
    More information about Cramér's V can be found on Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    
    rc_table = np.array(rc_table)
    
    def get_corr(rc_table):
        
        if rc_table.min() < 10:
            correction = True
        else:
            correction = False
            
        n = rc_table.sum()
        chi2_stats = st.chi2_contingency(rc_table, correction=correction)
        cramers_v = (chi2_stats[0]/(n*min(rc_table.shape[0]-1, rc_table.shape[1]-1)))**.5        
        return {'correlation':cramers_v, 'pvalue': chi2_stats[1], 'chi2': chi2_stats[0]}
    
    if observations == 'raise':
        if rc_table.min() < 5:
            raise ValueError('Not enough observations')
        else:
            return get_corr(rc_table)
    elif observations == 'ignore':
        return get_corr(rc_table)
    
def robust_mean(data, trunc_level=.2, type_='truncated'):
    """
    Calculates a robust mean of the data using either the truncated mean or Winsorized mean method.

    This function aims to provide a mean value that is less sensitive to outliers or extreme values by either
    truncating or Winsorizing the data based on a specified truncation level.

    Parameters
    ----------
    data : array-like
        The data from which the robust mean is to be calculated.
    trunc_level : float, optional
        The level at which data should be truncated or Winsorized, given as a proportion (default is 0.2).
    type_ : str, optional
        The type of robust mean to calculate: 'truncated' for truncated mean and 'winsorized' for Winsorized mean 
        (default is 'truncated').

    Returns
    -------
    float
        The calculated robust mean of the data.

    References
    ----------
    More information about the truncated mean can be found at: https://en.wikipedia.org/wiki/Truncated_mean
    More information about the Winsorized mean can be found at: https://en.wikipedia.org/wiki/Winsorized_mean
    """
    data = np.array(data)
    q = np.quantile(data, q=[trunc_level / 2, 1 - trunc_level / 2])
    trunc_data = data[(data > q[0]) & (data < q[1])]
    if type_ == 'truncated':
        return trunc_data.mean()
    elif type_ == 'winsorized':
        return np.clip(data, trunc_data.min(), trunc_data.max()).mean()

    
def binom_wilson_confidence_interval(p, n, confidence_level=0.95):
    """
    Calculates the Wilson confidence interval for a binomial proportion.

    This method provides a more accurate confidence interval for a binomial proportion, especially useful when the sample size is small or the proportion is close to 0 or 1.

    Parameters
    ----------
    p : float
        Observed proportion (successes / total).
    n : int
        Total number of observations (sample size).
    confidence_level : float, optional
        The desired confidence level for the interval (default is 0.95).

    Returns
    -------
    tuple
        A tuple containing the lower and upper bounds of the Wilson confidence interval.

    Notes
    -----
    The Wilson score interval is an improvement over the standard normal approximation, particularly for small sample sizes or extreme proportion values. 
    It adjusts the standard error to account for the uncertainty inherent in the estimation of a proportion.
    """
    z = st.norm.ppf(1 - (1 - confidence_level) / 2)
    a = p + (z**2 / (2 * n))
    b = z * (p * (1 - p) / n + z**2 / (4 * n**2))**0.5
    c = 1 + z**2 / n
    lower, upper = (a - b) / c, (a + b) / c
    return lower, upper

def get_lognormal_params(mean, std):
    """
    Function to estimate the parameters of a lognormal distribution.

    Methodology: Data is generated from a normal distribution with the given mean and 
    standard deviation. Then, the natural logarithm is taken from each data point.
    The mean and standard deviation of these log-transformed points are calculated 
    and returned as estimates of the lognormal distribution parameters.
    This approach corresponds to the Monte Carlo method for estimating statistical parameters.

    Parameters:
    mean (float): Mean value for generating the normal distribution.
    std (float): Standard deviation for generating the normal distribution.

    Returns:
    tuple: Estimates of the mean and standard deviation of the lognormal distribution.
    """
    dist = np.log(np.abs(np.random.normal(mean, std, size=1_000_000)))
    return dist.mean(), dist.std(ddof=1)
