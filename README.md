# PyStatLab

## Install
```bash
git clone https://github.com/carrollstreet/PyStatLab
cd PyStatLab
pip3 install .
```
## Update
```bash
cd PyStatLab
git pull origin main
pip3 install . --upgrade
```

# Docs 
# ab_tesing
The `ab_testing` module is essential for:
- Conducting A/B tests with robust statistical backing.
- Analyzing tests where traditional assumptions (e.g., normal distribution, equal variances) do not apply.
- Incorporating Bayesian approaches into A/B testing.
- Comparing quantiles in large datasets.
- Visualizing the results of statistical tests for better interpretation and decision-making.

## ParentTestInterface

### Description
Base interface for conducting statistical tests, particularly for A/B testing. Provides foundational methods and attributes for subclass implementations.

### Parameters
- `confidence_level`: The confidence level for calculating confidence intervals.
- `n_resamples`: The number of simulations or units to be considered in the test.
- `random_state`: The seed for the random number generator to ensure reproducibility.

### Methods
- `__init__`: Initializes the class.
- `__setattr__`: Custom attribute setter.
- `_compute_confidence_bounds`: Computes the confidence bounds.
- `_compute_ci`: Computes confidence intervals for given data.
- `_compute_uplift`: Calculates uplift.
- `get_test_parameters`: Retrieves initial test parameters.
- `_get_alternative_value`: Calculates p-value for two-sided tests.
- `_get_readable_format`: Prints results in a readable format.
- `_metric_distributions_chart`: Draws a histogram chart.
- `_uplift_distribtuion_chart`: Draws a cumulative distribution chart.
- `resample`: Abstract method for resampling.
- `compute`: Abstract method for computing test results.
- `get_charts`: Abstract method for generating charts.

---

## BayesBeta

### Description
Implements Bayesian approach to A/B testing using beta distributions. Extends ParentTestInterface with specific methods for Bayesian analysis.

### Parameters
- Inherits parameters from `ParentTestInterface`.

### Methods
- `__init__`: Constructor for the class.
- `resample`: Generates beta distributions for analysis.
- `compute`: Calculates statistical significance and other metrics.
- `get_charts`: Generates and displays charts.

---

## Bootstrap

### Description
Bootstrap class for conducting statistical tests using bootstrapping. Provides resampling method to estimate the distribution of a statistic.

### Parameters
- Inherits parameters from `ParentTestInterface`.
- `func`: Statistical function to apply to the samples.

### Methods
- `__init__`: Constructor for the Bootstrap class.
- `resample`: Performs the resampling process.
- `compute`: Computes statistical significance and metrics.
- `get_charts`: Generates and displays charts.

---

## QuantileBootstrap

### Description
Efficient implementation of the quantile comparison method using bootstrap resampling for large-scale A/B testing scenarios.

### Parameters
- Inherits parameters from `ParentTestInterface`.
- `q`: Target quantile for comparison.

### Methods
- `__init__`: Constructor for the QuantileBootstrap class.
- `resample`: Performs resampling for quantile distribution.
- `compute`: Computes statistical significance of quantile comparison.
- `get_charts`: Generates and displays charts.

---

## ResamplingTtest

### Description
Conducts t-distribution-based resampling tests in A/B testing scenarios, suitable for unequal variances and smaller sample sizes.

### Parameters
- Inherits parameters from `ParentTestInterface`.

### Methods
- `__init__`: Constructor for the class.
- `resample`: Performs resampling using a t-distribution approach.
- `compute`: Computes statistical significance using an analytical approach.
- `get_charts`: Generates and displays charts.

---

## Additional Functions

### permutation_ind
Performs an independent two-sample permutation test.

### g_squared
Calculates the G-squared statistic for a given contingency table.

### ttest_confidence_interval
Calculates the confidence interval for the difference between means using a t-test.






