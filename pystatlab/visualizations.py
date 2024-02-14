import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import numpy as np

def mosaic_plot(rc_table, residuals=None, title=None):
    """
    Generates a mosaic plot from a given contingency table. The plot can optionally display 
    standardized residuals or percentage errors to provide additional insights into the data.

    Parameters
    ----------
    rc_table : pd.DataFrame
        A pandas DataFrame representing the contingency table. Each cell should contain the count 
        of observations for the combination of categories represented by the row and column.
    residuals : {'standardized', 'percentage', None}, optional
        The type of residuals to display. 'standardized' will show standardized residuals, 
        'percentage' will show percentage errors, and None will display no additional information.
    title : str, optional
        The title of the plot. If None, no title is displayed.

    Raises
    ------
    ValueError
        If an invalid option is provided for the 'residuals' parameter.

    Returns
    -------
    None
        Displays the mosaic plot.

    Notes
    -----
    - The plot shows the distribution of counts across different categories.
    - Heights of bars represent the proportion of counts in each row category, while widths 
      represent the proportion of counts in each column category.
    - Standardized residuals can help identify categories with counts significantly different 
      from what would be expected under independence.
    - Percentage error provides a relative measure of the deviation from expected counts.
    - This function uses Plotly's graphing library for visualization.

    Example
    -------
    >>> df = pd.DataFrame({'A': [30, 20], 'B': [35, 15]}, index=['Category 1', 'Category 2'])
    >>> mosaic_plot(df, title='My Mosaic Plot', residuals='standardized')
    """
    labels = rc_table.columns
    heights = rc_table / rc_table.sum() 
    widths = rc_table.sum(axis=0) / rc_table.sum().sum()
    expected = st.chi2_contingency(rc_table)[3]
    standartized_residuals = ((rc_table - expected) / expected ** .5)
    expected_relative = expected / expected.sum(axis=0)
    percentage_error = (heights - expected_relative) / expected_relative

    
    chart = []
    for i in heights.index:
        if residuals == 'standardized':
            marker = dict(cmin=-4, cmax=4, color=standartized_residuals.loc[i], colorscale='RdBu',
                          colorbar={'title': ''})
            customdata = standartized_residuals.loc[i]
            texttemplate = ': '.join([f'{i}',"%{text:,}"])
            error_info = 'standardized_error: %{customdata:.1f}'
            showlegend = False
            
        elif residuals == 'percentage':
            marker = dict(cmin=-100, cmax=100, color=percentage_error.loc[i]*100, colorscale='edge_r',
                          colorbar={'ticktext':list(range(-100,101,20)), 
                                    'tickvals':list(range(-100,101,20)), 'title': ''})
            customdata = percentage_error.loc[i]
            texttemplate = ': '.join([f'{i}',"%{text:,}"])
            error_info = 'percentage_error: %{customdata:.1%}'
            showlegend = False
            
        elif residuals is None:
            marker = None
            customdata = None
            texttemplate = None
            showlegend = True
            error_info = ''
        
        else:
            raise ValueError(f"Invalid property name.\
            \nRecieved value: '{residuals}' \n\nUse ['standardized', 'percentage', None].")

        h = heights.loc[i]
        chart += [go.Bar(y=h, x=(np.cumsum(widths) - widths)*100, width=widths*100, offset=0, 
                         text=rc_table.loc[i], textposition='inside', marker=marker, name=i, textangle=0,
                         customdata=customdata, texttemplate=texttemplate,
                         hovertemplate="<br>".join(['height: %{y:.1%}',
                                                    'width: %{width:.1f}%',
                                                    'value: %{text:,}',
                                                     error_info])
                         )]

    fig = go.Figure(chart)
    fig.update_layout(template='simple_white', barmode="stack", uniformtext={'mode': "hide", 'minsize': 12},
                      yaxis={'tickformat':',.0%'},
                      xaxis={'range': [0, 100], 'tickvals': (np.cumsum(widths) - widths / 2) * 100,
                'ticktext': ["{}<br>{:,}".format(l, w) for l, w in zip(labels, rc_table.sum(axis=0).tolist())]},
                      title={'text': title, 'x': .5}, showlegend=showlegend,
                      legend_title_text=rc_table.index.name)
    fig.show()

def pareto_chart(collection, title=None, limit=None, layout_params={}):
    """
    Generates a Pareto chart from a given collection of data. The Pareto chart combines a bar chart and 
    a line graph, where individual values are represented in descending order by bars, and the cumulative 
    total is represented by the line.

    Parameters
    ----------
    collection : array-like, Series, or list
        The data collection from which to generate the Pareto chart. The collection should be categorical 
        or discrete data where frequency of occurrence matters.
    title : str, optional
        The title of the chart. If None, no title is displayed.
    limit : int
        How many values to display

    Returns
    -------
    None
        Displays the Pareto chart.

    Notes
    -----
    - The bars in the chart represent the frequency or count of each category in descending order.
    - The line graph represents the cumulative percentage of these counts.
    - The Pareto chart is useful for identifying the 'vital few' categories that account for the most 
      occurrences.
    - This function utilizes Plotly's graphing library for visualization.
    """
    if limit and not isinstance(limit, int):
        raise TypeError('Limit must be int data type')
        
    collection = pd.Series(collection)
    counts = (collection.value_counts().to_frame('counts')
              .join(collection
                    .value_counts(normalize=True)
                    .cumsum()
                    .to_frame('ratio')))[:limit]

    fig = go.Figure([go.Bar(x=counts.index, 
                            y=counts['counts'], 
                            yaxis='y1', 
                            name='count'),
                     go.Scatter(x=counts.index, 
                                y=counts['ratio'], 
                                yaxis='y2', 
                                name='cumulative ratio', 
                                hovertemplate='%{y:.1%}', 
                                marker={'color': '#000000'})])

    fig.update_layout(template='plotly_white', 
                      showlegend=False, 
                      hovermode='x', 
                      bargap=.3,
                      title={'text': title, 'x': .5}, 
                      yaxis={'title': 'count'},
                      yaxis2={'rangemode': "tozero", 
                              'overlaying': 'y',
                              'position': 1, 
                              'side': 'right',
                              'tickformat':'.1%'},
                     **layout_params)

    fig.show()

def ci_chart(metrics,
             central_values, 
             lower_bounds, 
             upper_bounds, 
             title=None, 
             xaxis_params={'tickformat':'.1%'}, 
             layout_params={}):
    """
    Generates a chart to visualize confidence intervals for a set of metrics. Each metric is represented 
    by a point, with horizontal error bars indicating the confidence interval.

    Parameters
    ----------
    metrics : array-like
        The names or identifiers of the metrics being visualized. These will appear on the y-axis.
    central_values : array-like
        The central or 'best estimate' values of the metrics. These values are plotted on the x-axis.
    lower_bounds : array-like
        The lower bounds of the confidence intervals for each metric.
    upper_bounds : array-like
        The upper bounds of the confidence intervals for each metric.
    title : str, optional
        The title of the chart. If None, no title is displayed.
    xaxis_params : dict, optional
        Additional parameters for customizing the x-axis, such as tick format. Defaults to a percentage format.
    layout_params : dict, optional
        Additional layout parameters for customizing the overall appearance of the chart.

    Returns
    -------
    None
        Displays the chart with confidence intervals.

    Notes
    -----
    - The function uses Plotly's graphing library for visualization.
    - Confidence intervals are visualized as horizontal error bars extending from the central values.
    - This type of chart is useful for comparing the precision and relative positions of different metrics.

    Example
    -------
    >>> metrics = ['Metric A', 'Metric B', 'Metric C']
    >>> central_values = [0.1, 0.15, 0.2]
    >>> lower_bounds = [0.08, 0.13, 0.18]
    >>> upper_bounds = [0.12, 0.17, 0.22]
    >>> ci_chart(metrics, central_values, lower_bounds, upper_bounds, title='Confidence Interval Chart')
    """
    central_values, lower_bounds, upper_bounds = np.asarray(central_values), np.asarray(lower_bounds), np.asarray(upper_bounds)
    
    fig = go.Figure(data=go.Scatter(
            mode='markers',
                x=central_values,
                y=metrics,
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=(upper_bounds-central_values).round(3),
                    arrayminus=(central_values-lower_bounds).round(3))
                ))
    fig.add_vline(x=0, 
                  line_dash="dash", 
                  line_color="black")
        
    fig.update_layout(template='plotly_white',
                     title={'text': title,'x':.5}, 
                     xaxis={**xaxis_params},
                     **layout_params)
    fig.show()
