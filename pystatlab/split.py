import pandas as pd
from sklearn.model_selection import train_test_split

class RankedSplit:
    """
    A class for performing stratified dataset splits based on ranking within a specified column.

    This class allows for creating splits of a dataset where the splits are based on stratified 
    ranking, particularly useful for handling imbalanced datasets or ensuring proportional representation across 
    strata.

    Parameters
    ----------
    sample_size_ratio : float
        The proportion of the dataset to include in the test set.
    strat_size : int, optional
        The number of strata to divide the dataset into based on the ranking (default is 10).
    same_size : bool, optional
        Whether the datasets should have the same size (default is False).
    random_state : int, optional
        Seed for the random number generator used in the split (default is None).

    Methods
    -------
    get_split(dataframe, target_column)
        Splits the dataframe into training and test sets based on stratified ranking within the target column.

    Raises
    ------
    TypeError
        If 'strat_size' is not an integer.
    ValueError
        If 'sample_size_ratio' is not within the range (0, 1) or if 'sample_size_ratio * strat_size' is less than 1.

    Notes
    -----
    The class utilizes Pandas and scikit-learn's train_test_split function for data manipulation and splitting. 
    It is designed to handle scenarios where a simple random split could result in unrepresentative data
    due to imbalances or specific distributions in the data.
    """
    
    def __init__(self, sample_size_ratio, strat_size=10, same_size=False, random_state=None):
        """
        Constructor for RankedSplit.
        """
        self.strat_size = strat_size
        self.random_state = random_state
        self.same_size = same_size

        if not isinstance(strat_size, int):
            raise TypeError('strat_size must be int')
        
        if sample_size_ratio <= 0 or sample_size_ratio >= 1:
            raise ValueError('The total ratio of sample_size must be greater than 0 and must not exceed 1')
        else:
            self.test_size = sample_size_ratio
        if sample_size_ratio * strat_size < 1:
            raise ValueError(f'With {size_per_sample=} your strat_size must be >= {int(1/size_per_sample)}')
            
    @staticmethod
    def _get_rank(data):
        """
        Computes the ranks for the values in the provided column.
    
        Parameters
        ----------
        data : array-like
            A array of data for which ranks are to be computed.
    
        Returns
        -------
        DataFrame
            A Pandas DataFrame containing the original data and their corresponding ranks.
    
        Notes
        -----
        The method uses the 'first' method to handle ties in ranking, meaning that ranks are assigned based on the 
        order in which the values appear in the column. This is particularly useful for preserving the original 
        order of data when multiple values have the same rank.
        """
        rank = pd.Series(data).rank(ascending=False, method='first')
        output = pd.DataFrame({'data':data,'rank':rank})
        return output
        
    def get_split(self, dataframe, target_column):
        """
        Splits the given dataframe into training and test sets based on stratified ranking.

        Parameters
        ----------
        dataframe : DataFrame
            The Pandas DataFrame to split.
        target_column : str
            The Pandas Series from a 'dataframe' based on which the ranking and stratification are performed.

        Returns
        -------
        tuple
            A tuple of DataFrames representing the training or control and test sets.

        Notes
        -----
        The method first ranks the values in the 'target_column', then divides these ranks into 'strat_size' strata. 
        It uses these strata to create a stratified split of the dataset, ensuring each stratum is represented in 
        both training and test sets.
        """
        self.dataset = self._get_rank(target_column)
        self.ranked_dataset = (pd.concat([self._get_rank(self.dataset[self.dataset['rank'] % self.strat_size == i]['data']) for i in range(0,self.strat_size)])
                               .sort_values(by=['rank','data'], ascending=(True,False)))
        self.ranked_dataset = self.ranked_dataset.drop('data', axis=1).join(dataframe)
        shortrank = self.ranked_dataset['rank'].value_counts().to_frame('count').query('count < @self.strat_size').index
        if len(shortrank) > 0:
            self.ranked_dataset.loc[self.ranked_dataset['rank'].eq(shortrank[0]), 'rank'] = shortrank[0] - 1

        if not self.same_size:
            train_size = None
        else:
            train_size = self.test_size
        second, first = train_test_split(self.ranked_dataset, 
                                        train_size=train_size,
                                        test_size=self.test_size,
                                        shuffle=True, 
                                        stratify=self.ranked_dataset['rank'],
                                        random_state=self.random_state)

        return first, second
