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
  
## Classes

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

## Functions

### permutation_ind
Performs an independent two-sample permutation test.

### g_squared
Calculates the G-squared statistic for a given contingency table.

### ttest_confidence_interval
Calculates the confidence interval for the difference between means using a t-test.

---
# test_design
The `test_design` module is instrumental for:
- Planning and conducting A/B tests with appropriate sample sizes to detect meaningful effects.
- Evaluating the effectiveness of different statistical tests for specific distributions.
- Calculating effect sizes and understanding their implications in experimental design.
- Controlling error rates in multiple testing situations, ensuring the reliability of results.

## Classes

### DurationEstimatorInterface
- **Description**: Base class for estimating test duration and sample size.
- **Attributes**:
  - `uplift`: Expected effect size or uplift.
  - `daily_size_per_sample`: Daily observations per sample.
  - `alpha`: Significance level for hypothesis testing.
  - `power_threshold`: Desired test power.
  - `n_resamples`: Number of resampling iterations.
  - `random_state`: Seed for random number generator.
- **Methods**:
  - `__init__`: Initializes the class with specified parameters.
  - `__setattr__`: Customizes attribute setting, especially for uplift.
  - `compute_size`: Computes required sample size and duration.
  - `_compute_pvalues`: Abstract method for p-value computation.

### ProportionSizeEstim
- **Description**: Estimates sample size and duration for proportion metrics.
- **Inherits**: `DurationEstimatorInterface`
- **Attributes**:
  - `cr_baseline`: Baseline conversion rate.
  - Additional attributes inherited from `DurationEstimatorInterface`.
- **Methods**:
  - `__init__`: Initializes the class with specific parameters for proportions.
  - `_compute_pvalues`: Computes p-values for proportion metrics.

### TtestSizeEstim
- **Description**: Estimates sample size and duration for mean metrics using T-tests.
- **Inherits**: `DurationEstimatorInterface`
- **Attributes**:
  - `target_sample`: Target sample data for analysis.
  - Additional attributes inherited from `DurationEstimatorInterface`.
- **Methods**:
  - `__init__`: Initializes the class with specific parameters for T-tests.
  - `_compute_pvalues`: Computes p-values for mean metrics.

## Functions

### cohens_d
- **Description**: Calculates Cohen's d, a measure of effect size for the difference between two means.
- **Parameters**:
  - `*args`: Two sample arrays or four specific values (two means and two standard deviations).
  - `from_samples`: Flag to indicate calculation method.
- **Returns**: Cohen's d value.

### proportion_size
- **Description**: Calculates required sample size for proportion testing.
- **Parameters**:
  - `p`: Baseline proportion.
  - `uplift`: Expected uplift.
  - `n_comparison`: Number of comparisons.
  - `alpha`: Significance level.
  - `power`: Desired test power.
  - `groups`: Number of groups in the experiment.
- **Returns**: Total sample size.

### ttest_size
- **Description**: Calculates required sample size for T-tests on mean metrics.
- **Parameters**: Similar to `proportion_size`, but focused on mean metrics.
- **Returns**: Total sample size.

### expected_proportion
- **Description**: Estimates expected proportion based on effect size.
- **Parameters**:
  - `effect_size`: Anticipated effect size.
  - `proportion_1`: Proportion in the first group.
- **Returns**: Expected proportion in the second group and uplift.

### normal_1samp_size
- **Description**: Determines sample size for estimating population mean.
- **Parameters**:
  - `sigma`: Population standard deviation.
  - `d`: Desired precision level.
  - `confidence_level`: Desired confidence level.
- **Returns**: Required sample size.

### proportion_1samp_size
- **Description**: Determines sample size for estimating population proportion.
- **Parameters**: Similar to `normal_1samp_size`, but for proportions.
- **Returns**: Required sample size.

### fwer
- **Description**: Calculates the family-wise error rate for multiple hypothesis testing.
- **Parameters**:
  - `n_comparison`: Number of hypothesis tests.
  - `alpha`: Significance level for a single comparison.
- **Returns**: Calculated FWER.








