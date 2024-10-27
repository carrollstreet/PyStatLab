# PyStatLab
PyStatLab is a comprehensive toolkit designed for statistical analysis, hypothesis testing, and data-driven decision-making. It provides a wide range of functions and classes to support researchers, data analysts, and data scientists in conducting rigorous statistical investigations across various domains. Its design emphasizes flexibility, ease of use, and high computational performance, making it suitable for both small-scale analyses and large, data-intensive projects.

Key Features

	•	Resampling and Bootstrap Analysis: The library includes robust support for bootstrap methods, enabling users to compute confidence intervals, perform resampling-based tests, and estimate distributions of sample statistics. This allows for more accurate inference in cases where assumptions of traditional parametric tests may not hold.
	•	Hypothesis Testing: With a variety of built-in tests, including permutation tests, Bootstrap, Quantile Poisson Bootstrap, Resampling Ttest for continuous data and Bayes test for proportions, the library offers tools for assessing statistical significance in diverse scenarios. Some of these tests are designed to handle both independent and paired samples, as well as ratio-based comparisons.
	•	Power Analysis and Sample Size Estimation: DurationEstimator, ttest_size, proportion_size, normal_1samp_size, proportion_1samp_size, fixed_power assist users in planning experiments and surveys by estimating the required sample size and duration to achieve a desired power level. This feature is particularly valuable in experimental design, helping to ensure that studies are adequately powered to detect meaningful effects.
	•	Parallel Processing for Speed: Many of the library’s methods, including bootstrapping and resampling functions, are optimized for parallel processing. This significantly speeds up computations for large datasets and iterative analyses, making the library suitable for use in high-performance environments.
	•	Customizable Statistical Functions: Users can define their own statistical functions to apply in resampling procedures, allowing for tailored analyses that suit specific research needs. The library’s flexibility supports a wide range of metrics, from simple averages to complex ratios.
	•	Data Visualization Support: While the core focus is on analysis, the library provides tools to generate plots and visualizations for resampling results, distribution comparisons, and more. These features help users interpret and communicate their findings more effectively.

Typical Use Cases

	•	Experimental Analysis: Ideal for A/B testing and other experimental setups where understanding the effect of changes is crucial, such as marketing campaigns, product testing, and clinical trials.
	•	Survey Data Analysis: Useful for analyzing survey data, including estimation of population parameters and calculation of confidence intervals.
	•	Business Analytics: Helps businesses make data-driven decisions through robust statistical tests and power analysis, ensuring that conclusions drawn from data are reliable and actionable.
	•	Research and Academic Use: Provides the depth and flexibility needed for rigorous statistical research in fields like psychology, medicine, and social sciences.

Flexibility and Extensibility

The library’s modular design allows users to extend its functionality with custom methods and integrate it with other analytical workflows. It works seamlessly with popular data manipulation libraries like numpy, scipy, statsmodels, pandas, making it easy to incorporate into existing data pipelines. With support for various statistical paradigms and methods, the library is suitable for both beginners looking to perform basic analyses and advanced users needing custom-tailored statistical solutions.

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


