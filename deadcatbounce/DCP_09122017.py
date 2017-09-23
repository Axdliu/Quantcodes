""" Final Project """
""" Topic2 Multi_factor Portfolio Trading Strategy """

""" By Xiaodong LIU """

import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random as rd
from sklearn.linear_model import LinearRegression
import pickle

""" Key Assumptions """
# data path
path_ticker = "D:\\QCF\\DCP\\"          # address for ticker list
path_data = "D:\\QCF\\DCP\\database\\"   # address for data download
start_date = dt.datetime(2010, 1, 1)
end_date =  dt.datetime(2017, 9, 11)

""" read stock names ticker """       
tickers = list(pd.read_csv(path_ticker+"russell_3000_20110627.csv", header=None)[0])
avail_tickers = []

for i in tickers:
    try:
        readdata = web.DataReader(i, 'yahoo',  start_date, end_date)
        readdata.to_csv(path_data  + i + ".csv")
        avail_tickers.append(i)
    except IOError:
        print "The data of ", i, " cannot be downloaded!"
        continue

fulldataset = web.DataReader("^RUA", 'yahoo',  start_date, end_date)
fulldataset = fulldataset.rename(columns = {"Adj Close":"RUA"})["RUA"] 
for ticker in avail_tickers: 
    print(ticker)
    data_file = path_data + ticker + ".csv" 
    data = pd.read_csv(data_file, header=0, index_col=0)
    data = data.set_index(pd.to_datetime(data.index))
    fulldataset = pd.concat([fulldataset, data["Adj Close"]], axis=1, join_axes=[fulldataset.index])
    fulldataset = fulldataset.rename(columns = {"Adj Close":ticker})                    

""" store data """           
pickle.dump(fulldataset, open(path_ticker+"data.p", "wb" ))
fulldata_1 = pickle.load( open(path_ticker+"data.p", "rb" )) 

""" computation and split data  """
fullrtn = fulldata_1.pct_change().iloc[1:]
stats_nan = np.sum(pd.isnull(fullrtn))
fullrtn_dn = fullrtn.dropna(axis=1)
data_train = fullrtn_dn[fullrtn_dn.index <= dt.datetime(2016, 12, 31)]
data_test = fullrtn_dn[fullrtn_dn.index > dt.datetime(2016, 12, 31)]

""" analysis """
# define common day
dailystdindex = data_train['RUA'].std()
dailymeanindex = data_train['RUA'].mean()

return_summary = {}
window_list = range(3,8)
result = pd.DataFrame()
    
def deadcat(dataset, window=7, indexname='RUA', sharpfall=-0.1, num_std=1):
    
    range_common = (dailymeanindex-num_std*dailystdindex, dailymeanindex+num_std*dailystdindex)
    common_date = data_train[indexname][ (data_train[indexname]>=range_common[0]) & (data_train[indexname]<=range_common[1]) ]
    uncommon_date = data_train[indexname][ (data_train[indexname]<range_common[0]) | (data_train[indexname]>range_common[1]) ]
    # data_train['RUA'].plot()
    # common_date.plot()
    common_day = pd.DataFrame(0, index=data_train.index, columns=data_train.columns)
    common_day.loc[common_date.index] = 1
    # find sharp fall
    sharp_drop = pd.DataFrame(0, index=data_train.index, columns=data_train.columns)
    sharp_drop[data_train<=sharpfall] = 1 # sharp drop day
    # find sharp fall track
    rtn_neg = (data_train < 0)*1
    loss_track = pd.DataFrame(index=data_train.index, columns=data_train.columns)
    loss_track.iloc[0] = rtn_neg.iloc[0]
    for i in range(1,len(loss_track.index)):
        loss_track.iloc[i] = loss_track.iloc[i-1].multiply(rtn_neg.iloc[i]) + rtn_neg.iloc[i]
    # track loss for the sharp loss date
    sharp_drop_track = sharp_drop
    for day in range(1, window):
        sharp_drop_track = sharp_drop_track.add(sharp_drop.shift(day))
    sharp_drop_track =  ((sharp_drop_track > 0)*1).multiply(common_day)
    sharp_drop_track = sharp_drop_track.multiply(rtn_neg)
    sharp_drop_cum = pd.DataFrame(index=data_train.index, columns=data_train.columns)
    sharp_drop_cum.iloc[0] = sharp_drop_track.iloc[0]
    for i in range(1,len(loss_track.index)):
        sharp_drop_cum.iloc[i] = sharp_drop_cum.iloc[i-1].multiply(sharp_drop_track.iloc[i]) + sharp_drop_track.iloc[i]
    trade_signal = (sharp_drop_cum == window)*1
    trade_signal[trade_signal != 1] = np.nan
    return_daily = trade_signal.shift(1).multiply(data_train)
    # remove some suspicious tickers
    return_daily['ART'] = np.nan
    return return_daily

pickle.dump(return_summary, open(path_ticker+"result.p", "wb" ))

window=10
test = deadcat(dataset=data_train, window=window, indexname='RUA', sharpfall=-0.1, num_std=1)
return_summary[str(window)] = test.mean()
result[str(window)] = return_summary[str(window)].describe() 
return_summary[str(window)].hist(bins=50)
result.to_csv(path_ticker+"result.csv")
