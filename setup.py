from setuptools import setup, find_packages

setup(
    name='pystatlab',
    version='0.1',
    packages=find_packages(),
    description='''PyStatLab is a powerful and flexible Python library designed to streamline common and advanced analytical tasks in data analysis, statistics, and machine learning. It offers a broad range of tools for statistical testing, including A/B test analysis, hypothesis testing, and distribution fitting, as well as optimization algorithms and mathematical computations. Ideal for data scientists/analysts and researchers, PyStatLab enables efficient and in-depth data exploration, model evaluation, and algorithm development. Whether you're conducting robust statistical tests, optimizing models, or performing complex integrations and transformations, PyStatLab provides the necessary functionality in an intuitive and user-friendly package.''',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kiryl Sirotsin',
    author_email='musomania@protonmail.com',
    url='https://github.com/carrollstreet/PyStatLab/',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'plotly',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
