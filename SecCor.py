
"""
Sector rotation based on cross_industry corelation
Version 3: train 30 sectors rolling 12-month returns and annual returns (based on monthly returns) minus monthly return of Treasury bill return
By Arnold LIU
Created on June 14, 2017
"""

import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import Lasso
import os
import holoviews as hv
from bkcharts import Histogram

def Quantmodel1(path):
    # setup path
    file_name = "sector_return_30.csv"

    # assumptions
    train_start_year = 1960
    train_end_year = 1999
    test_start_year = 2000
    test_end_year = 2017
    signal_return_months = 12
    # end_month = 12
    sector_num = 30
    index_column = "Date"

    # reading data into DataFrame
    data_raw = pd.read_csv(path+file_name)
    data_raw[index_column] =  pd.to_datetime(data_raw[index_column])
    data_raw_indexed = data_raw.set_index(data_raw[index_column], drop=True).iloc[:,1:]

    # compute data
    sector_name = data_raw_indexed.columns
    data_sector_log = np.log(data_raw_indexed/100.0 + 1)

    # compute rolling compounded returns
    data_sector_rolling = pd.DataFrame(index = data_sector_log.index)
    for col in sector_name:
        data_sector_rolling[col] = \
        (np.exp( data_sector_log[col].rolling(window=signal_return_months,center=False).sum() )-1)
    for col in sector_name:
        data_sector_rolling[col+"_forward_"+ str(signal_return_months)] = data_sector_rolling[col].shift(-signal_return_months)
    data_sector_rolling_dropna = data_sector_rolling.dropna()

    # model A: monthly rolling 12 months
    dict_coef_result, dict_return_forecast, dict_predict_rank, dict_top_ranks = {}, {}, {}, {}
    portfolio_returns = pd.DataFrame(index=range(test_start_year+1, test_end_year), columns=range(1,13))
    rank = 3
    portfolio_weight = [0.34, 0.33, 0.33]
    for end_month in range(1,13):
        test_data_A = data_sector_rolling_dropna.loc[data_sector_rolling_dropna.index.month == end_month]
        trian_X = test_data_A.loc[(test_data_A.index.year>=train_start_year) & (test_data_A.index.year<=train_end_year),sector_name]
        test_X = test_data_A.loc[(test_data_A.index.year>=test_start_year) & (test_data_A.index.year<=test_end_year),sector_name]
        coef_result = pd.DataFrame(index=sector_name, columns=sector_name)
        coef_result['intercept'] = 0
        return_forecast = pd.DataFrame(index=test_X.index.year)
        model = Lasso(alpha=0.007, fit_intercept=False)
        for sec in sector_name:
            train_y = test_data_A.loc[(test_data_A.index.year>=train_start_year) & (test_data_A.index.year<=train_end_year),sec+"_forward_"+ str(signal_return_months)]
            model.fit(trian_X, train_y)
            coef_result.loc[sec] = list(model.coef_) + [model.intercept_]
            return_forecast[sec] = model.predict(test_X)
        predict_rank = return_forecast.rank(axis=1,ascending=False)
        top_ranks = pd.DataFrame(index=range(test_start_year, test_end_year-1), columns=range(1,rank+1))
        top_ranks_returns = pd.DataFrame(index=range(test_start_year+1, test_end_year), columns=range(1,rank+1))
        return_table = data_sector_rolling.loc[data_sector_rolling.index.month == end_month]
        test_data_year = return_table.set_index(return_table.index.year)
        for year in range(test_start_year, test_end_year-1):
            for r in top_ranks.columns:
                top_ranks.loc[year,r] = predict_rank.loc[:,predict_rank.loc[year] == r].columns[0]
                top_ranks_returns.loc[year+1,r] = test_data_year.loc[year+1, top_ranks.loc[year,r]]
        portfolio_returns[end_month] = (top_ranks_returns*portfolio_weight).sum(axis=1)
        dict_coef_result[end_month] = coef_result
        dict_return_forecast[end_month] = return_forecast
        dict_predict_rank[end_month] = predict_rank
        dict_top_ranks[end_month] = top_ranks

    portfolio_returns.loc["Average"] =  portfolio_returns.loc[range(test_start_year+1, test_end_year)].mean()
    # portfolio_returns.to_csv(path+"model_3_result.csv")

    return portfolio_returns[1]


def create_figure(df, current_feature_name, bins):

	p = Histogram(df, current_feature_name, title=current_feature_name, color='Portfolios',
 	bins=bins, legend='top_right', width=600, height=400)

	# Set the x axis label
	p.xaxis.axis_label = 'Returns: percentage'

	# Set the y axis label
	p.yaxis.axis_label = 'Account Number'
	return p
