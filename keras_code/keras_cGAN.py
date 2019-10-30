#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 25 13:20:52 2019

@author: Yiwei Zhang
"""
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from mpmath import math2
import copy

import datetime as dt
import pandas_datareader.data as web
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib.pylab import rcParams
from fastai.tabular import add_datepart
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from pmdarima.arima import auto_arima

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, LeakyReLU, Input, Flatten, Embedding, multiply, Reshape, BatchNormalization

import keras

# data of AAPL from 4/4/2016---6/15/2016(3 months)
# baseline models:LR,KNN,auto ARIMA and LSTM
# trainset/testset: others/last 5 days
# changed at 16:14; 9/10
class asset_prices:
	
	def __init__(self, symbol='AAPL'):
		# inputs
		self.symbol = symbol
		self.df = pd.DataFrame()
		# outputs
		self.train_df = pd.DataFrame()
		self.test_df = pd.DataFrame()
		
		#parameters 80 DAYS
		self.start = '2016-04-4'
		self.start_y = 2016
		self.start_m = 4
		self.start_d = 4
		self.end = '2016-06-10'
		self.end_y = 2016
		self.end_m = 6
		self.end_d = 10
		self.split_day = 66 - 1 #shift(): delete one row
		self.n_periods = 5
		
		#LSTM
		self.step_size = 5
		self.feature_num = 3
		
		#GAN
		self.feature_numG = 4
		self.num_classes = 3
		self.n_train_days = self.split_day - self.step_size
		self.iterations1 = self.n_train_days
		self.iterations2 = self.n_periods
		self.batch_size = 1
	# set root directory
	# os.chdir("../../")
	# print("root directory = %s " % os.getcwd())
	
	def sentimentScore(self,Tweet):  # Add sentiment to each tweet by using Vader
		analyzer = SentimentIntensityAnalyzer()
		results = []
		for sentence in Tweet:
			vs = analyzer.polarity_scores(sentence)
			#print("Vader score: " + str(vs))
			
			results.append(vs)
		return results
	
	def load_data(self):
		""" Crawl price data from yahoo and combine it with local text to create new variables
		Input:Sentiment information from tweet
		Output:All stock data in dataframe style
		"""
		# xls = pd.ExcelFile('input/2016_AAPL_tweet.xlsx')
		# stock = "AAPL"
		# df = pd.read_excel(xls, header=0, encoding='latin-1')
		# df["date"] = df[
		# 	"created at"].dt.date  # Add column with just the date, remove column with date & time, rearrange columns
		# Tweet = df['text']
		# Tweet.head()
		#
		# df_results = pd.DataFrame(self.sentimentScore(Tweet))  # use sentimentScore function
		# df_tweets = pd.merge(df, df_results, left_index=True, right_index=True)
		# df_tweets['date'] = pd.to_datetime(df_tweets['date'])
		# df_tweets = df_tweets[(df_tweets['date'] >= self.start) & (df_tweets['date'] <= self.end)]
		#
		# df_tweets['datetime'] = pd.to_datetime(df_tweets['date'])
		# df_tweet_SA = df_tweets.set_index('datetime')
		# df_tweet_SA.drop(['date'], axis=1, inplace=True)
		#
		# df_tweet_SA = df_tweets[['datetime', 'text', 'follower count', 'compound', 'neg', 'neu', 'pos']]
		# df_tweet_SA = df_tweet_SA[(df_tweet_SA[['compound']] != 0).all(axis=1)]  # Remove tweets were compound is zero
		# df_tweet_SA['Compound_multiplied_raw'] = df_tweet_SA['compound'] * df_tweet_SA['follower count']
		# # 'compound' multiplied by nr of followers of the Tweeter
		# nan_rows = df_tweet_SA[df_tweet_SA['follower count'].isnull()]
		# # nan_rows
		# # Remove rows where 'follower count' is NaN
		# df_tweet_SA = df_tweet_SA[np.isfinite(df_tweet_SA['follower count'])]
		# df_daily_mean = (df_tweet_SA.groupby(df_tweet_SA.datetime).mean())
		#
		# # daily MEANS of each column
		# start = dt.datetime(self.start_y, self.start_m, self.start_d)
		# end = dt.datetime(self.end_y, self.end_m, self.end_d)
		#
		# df_stock = web.DataReader(stock, 'yahoo', start, end)
		# df_stock['Pct_change_raw'] = (df_stock['Close'] - df_stock['Open']) / df_stock['Open'] * 100.0
		# df_full = pd.concat([df_stock[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Pct_change_raw']], \
		#                      df_daily_mean], axis=1, sort=False)
		# df_full.tail(11)
		# # Combine the tweet sentiment dataframe with the stock data dataframe
		# df_full['follower count'].fillna(df_full['follower count'].mean(), inplace=True)
		# df_full['compound'].fillna(df_full['compound'].mean(), inplace=True)
		# df_full['neg'].fillna(df_full['neg'].mean(), inplace=True)
		# df_full['neu'].fillna(df_full['neu'].mean(), inplace=True)
		# df_full['pos'].fillna(df_full['pos'].mean(), inplace=True)
		# df_full['Compound_multiplied_raw'].fillna(df_full['Compound_multiplied_raw'].mean(), inplace=True)
		#
		# # relaced missing data with their means
		# df_full = df_full[
		# 	['High', 'Low', 'Open', 'Close', 'Adj Close', 'Pct_change_raw', 'follower count', 'compound', 'neg', 'neu',
		# 	 'pos','Compound_multiplied_raw']].interpolate(method='linear', limit_direction='forward', axis=0)
		# # Interpolate for missing weekend stock data
		# pd.DataFrame.describe(df_full)
		#
		# df_final = df_full.drop(['Adj Close', 'follower count', 'compound', 'neg', 'neu', 'pos'], axis=1)
		# order = ['Open', 'High', 'Low', 'Close', 'Pct_change_raw', 'Compound_multiplied_raw']
		# self.df_final = df_final[order]
		# self.df_final.reset_index()
		#
		# self.df_final.to_csv('input/results_AAPL.csv')
		self.df_final = pd.read_csv('input/results_CSCO_new.csv')
		self.df_final.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Pct_change_raw', 'Compound_multiplied_raw']
		
		#avoid the data leakage problem,but nn models don't have to avoid
		self.nn_data1 = self.df_final.iloc[len(self.df_final)-1][5]
		self.nn_data2 = self.df_final.iloc[len(self.df_final)-1][6]
		print(self.nn_data1,self.nn_data2)
		
		self.df_final['Pct_change'] = self.df_final['Pct_change_raw'].shift(1)
		self.df_final.drop(['Pct_change_raw'], axis=1, inplace=True)
		self.df_final['Compound_multiplied'] = self.df_final['Compound_multiplied_raw'].shift(1)
		self.df_final.drop(['Compound_multiplied_raw'], axis=1, inplace=True)
		self.df_final.dropna(axis=0, how='any', inplace=True)
		print(self.df_final.head())

	def join_data(self):
		""" Setting index as date and sort the whole dataset
		INPUT:initial dataset 'df'
		OUTPUT:dataset after indexed and sorted 'data'
		"""
		self.df_final['Date'] = pd.to_datetime(self.df_final.Date, format='%Y-%m-%d')
		self.df_final.index = self.df_final['Date']
		self.data = self.df_final.sort_index(ascending=True, axis=0)
		print(self.data.head())
	
	def transform_data(self):
		""" Remove useless valuables and divide dataset into OHLC datasets (add two more colums the next time)
		INPUT:dataset that has been set 'date' as index
		OUTPUT:'data' divide respectively into O,H,L,C dataframe and then combine into a list
		"""
		self.data_list = []
		#self.data = self.data.drop(['Adj Close', 'Volume'], axis=1)
		print(self.data.head())
		
		# data['HL_PCT'] = (data['High'] - data['Low'])/data['Low'] * 100.0
		# data['PCT_change'] = (data['Close'] - data['Open'])/data['Open'] * 100.0
		
		data_O = pd.DataFrame(index=range(0, len(self.data)), columns=['Date', 'Open','Pct_change','Compound_multiplied'])
		for i in range(0, len(self.data)):
			data_O['Date'][i] = self.data['Date'][i]
			data_O['Open'][i] = self.data['Open'][i]
			data_O['Pct_change'][i] = self.data['Pct_change'][i]
			data_O['Compound_multiplied'][i] = self.data['Compound_multiplied'][i]
		
		data_H = pd.DataFrame(index=range(0, len(self.data)), columns=['Date', 'High','Pct_change','Compound_multiplied','Open'])
		for i in range(0, len(self.data)):
			data_H['Date'][i] = self.data['Date'][i]
			data_H['High'][i] = self.data['High'][i]
			data_H['Pct_change'][i] = self.data['Pct_change'][i]
			data_H['Compound_multiplied'][i] = self.data['Compound_multiplied'][i]
			data_H['Open'][i] = self.data['Open'][i]
		
		data_L = pd.DataFrame(index=range(0, len(self.data)), columns=['Date', 'Low','Pct_change','Compound_multiplied','Open'])
		for i in range(0, len(self.data)):
			data_L['Date'][i] = self.data['Date'][i]
			data_L['Low'][i] = self.data['Low'][i]
			data_L['Pct_change'][i] = self.data['Pct_change'][i]
			data_L['Compound_multiplied'][i] = self.data['Compound_multiplied'][i]
			data_L['Open'][i] = self.data['Open'][i]
		
		data_C = pd.DataFrame(index=range(0, len(self.data)), columns=['Date', 'Close','Pct_change','Compound_multiplied','Open'])
		for i in range(0, len(self.data)):
			data_C['Date'][i] = self.data['Date'][i]
			data_C['Close'][i] = self.data['Close'][i]
			data_C['Pct_change'][i] = self.data['Pct_change'][i]
			data_C['Compound_multiplied'][i] = self.data['Compound_multiplied'][i]
			data_C['Open'][i] = self.data['Open'][i]
		
		self.data_list.append(data_O)
		self.data_list.append(data_H)
		self.data_list.append(data_L)
		self.data_list.append(data_C)
		
		#if you don'twant to add sentiments values;change this part;also change line 266 268 275 and line 755 758 !!!!!!
		# aa = []
		# for i in range(4):
		# 	a = self.data_list[i].drop(['Pct_change','Compound_multiplied'],axis=1)
		# 	aa.append(a)
		# self.data_list = aa
		
	def clean_data(self):
		"""Treat outliers and split dataset:train/test=3/1,then normalization
		INPUT:a list with dataset of O,H,L,C and 'data' for getting attributes
		OUTPUT:O,H,L,C respectively divided into trainset and testset,then combine into 2 lists'train_list','test_list'
		"""
		#get the statistics of trainset
		self.statistic_temp = copy.deepcopy(self.data)
		self.statistic = self.statistic_temp.iloc[:self.split_day]
		statistics = self.statistic.describe()
		
		print(statistics.head())
		
		means = statistics.loc[['mean']]
		stds = statistics.loc[['std']]
		
		self.train_list = []
		self.test_list = []
		for n in range(4):
			data_temp = self.data_list[n]
			train_temp = data_temp[:self.split_day]
			test_temp = data_temp[self.split_day:]
			self.train_list.append(train_temp)
			self.test_list.append(test_temp)
		
		# treat outliers and normalize data
		for j in range(4):
			train_temp = self.train_list[j]
			range_max = means.iloc[0][j] + 2 * stds.iloc[0][j]
			range_min = means.iloc[0][j] - 2 * stds.iloc[0][j]
			for i in range(3):
				if train_temp.iloc[i][1] < range_min or train_temp.iloc[i][1] > range_max:
					train_temp.iloc[i][1] = means.iloc[0][j]
				else:
					continue
			for i in range(3, len(train_temp)):
				if train_temp.iloc[i][1] < range_min or train_temp.iloc[i][1] > range_max:
					three_mean = (train_temp.iloc[i - 1][1] + train_temp.iloc[i - 2][1] + train_temp.iloc[i - 3][1]) / 3
					train_temp.iloc[i][1] = three_mean
				else:
					continue
			
			# Fourier Transform
			Price_temp = train_temp.iloc[:,1]
			open_fft = np.fft.fft(np.asarray(Price_temp.tolist()))
			fft_df = pd.DataFrame({'fft': open_fft})
			fft_list = np.asarray(fft_df['fft'].tolist())
			fft_list_m10 = np.copy(fft_list)
			num_ = 100  # num_ is fourier transform component
			fft_list_m10[num_:-num_] = 0
			fft_list_m10 = np.fft.fftshift(fft_list_m10)
			Fourier_LR = np.abs(np.fft.ifft(fft_list_m10))
			Fourier_LR = np.asarray(Fourier_LR.tolist())
			
			for i in range(0, len(train_temp)):
				train_temp.iloc[i][1] = Fourier_LR[i]
			self.train_list[j] = train_temp
	
	def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg
	
	def nn_preprocess(self):
		""" Build a training model LSTM with train data in self.train_df
		"""
		# setting index; rolling mean of the sentiment variables and wipe out its outlier
		for j in range(4):
			self.train_list[j].index = self.train_list[j].Date
			self.train_list[j].drop(['Date'], axis=1, inplace=True)
			if j > 0 :
				self.train_list[j].drop(['Open'], axis=1, inplace=True)
			data_temp = self.train_list[j]
			for i in range(len(data_temp) - 1, 0, -1):
				data_temp.iloc[i][2] = (data_temp.iloc[i][2] - data_temp.iloc[i - 1][2]) / data_temp.iloc[i - 1][2]
			data_temp.iloc[0][2] = data_temp.iloc[1][2]
			for i in range(3):
				if data_temp.iloc[i][2] < -5:
					data_temp.iloc[i][2] = (data_temp.iloc[i][2] - 5) / data_temp.iloc[i][2]
				elif data_temp.iloc[i][2] > 5:
					data_temp.iloc[i][2] = (data_temp.iloc[i][2] + 5) / data_temp.iloc[i][2]
			for i in range(3, len(data_temp)):
				if data_temp.iloc[i][2] < -5 or data_temp.iloc[i][2] > 5:
					three_mean = (data_temp.iloc[i - 1][2] + data_temp.iloc[i - 2][2] + data_temp.iloc[i - 3][2]) / 3
					data_temp.iloc[i][2] = three_mean
				else:
					continue
			self.train_list[j] = data_temp
		for j in range(4):
			self.test_list[j].index = self.test_list[j].Date
			self.test_list[j].drop(['Date'], axis=1, inplace=True)
			if j > 0 :
				self.test_list[j].drop(['Open'], axis=1, inplace=True)
			data_temp = self.test_list[j]
			for i in range(len(data_temp) - 1, 0, -1):
				data_temp.iloc[i][2] = (data_temp.iloc[i][2] - data_temp.iloc[i - 1][2]) / data_temp.iloc[i - 1][2]
			data_temp.iloc[0][2] = data_temp.iloc[1][2]
			for i in range(3):
				if data_temp.iloc[i][2] < -5:
					data_temp.iloc[i][2] = (data_temp.iloc[i][2] - 5) / data_temp.iloc[i][2]
				elif data_temp.iloc[i][2] > 5:
					data_temp.iloc[i][2] = (data_temp.iloc[i][2] + 5) / data_temp.iloc[i][2]
			for i in range(3, len(data_temp)):
				if data_temp.iloc[i][2] < -5 or data_temp.iloc[i][2] > 5:
					three_mean = (data_temp.iloc[i - 1][2] + data_temp.iloc[i - 2][2] + data_temp.iloc[i - 3][2]) / 3
					data_temp.iloc[i][2] = three_mean
				else:
					continue
			self.test_list[j] = data_temp
		
		#gain OHLC dataset before splited(but after transformation)
		self.values_list = []
		for i in range(4):
			nn_temp1 = self.test_list[i].iloc[0][1]
			nn_temp2 = self.test_list[i].iloc[0][2]
			(self.train_list[i])['Pct_change'] = (self.train_list[i])['Pct_change'].shift(-1)
			(self.train_list[i])['Compound_multiplied'] = (self.train_list[i])['Compound_multiplied'].shift(-1)
			(self.train_list[i])['Pct_change'][len(self.train_list[i])-1] = nn_temp1
			(self.train_list[i])['Compound_multiplied'][len(self.train_list[i]) - 1] = nn_temp2

			(self.test_list[i])['Pct_change'] = (self.test_list[i])['Pct_change'].shift(-1)
			(self.test_list[i])['Compound_multiplied'] = (self.test_list[i])['Compound_multiplied'].shift(-1)
			(self.test_list[i])['Pct_change'][len(self.test_list[i]) - 1] = self.nn_data1
			(self.test_list[i])['Compound_multiplied'][len(self.test_list[i]) - 1] = self.nn_data2
		
		# for GAN
		self.train_list_G = copy.deepcopy(self.train_list)
		self.test_list_G = copy.deepcopy(self.test_list)
		
		for i in range(4):
			self.train_list[i] = self.train_list[i].values
			self.test_list[i] = self.test_list[i].values
			values_temp = np.concatenate([self.train_list[i], self.test_list[i]], 0)
			# ensure all data is float
			values_temp = values_temp.astype('float32')
			self.values_list.append(values_temp)

	def train_cGAN(self):
		# bulid generator
		g_input = keras.Input(shape=(1, self.step_size * self.feature_numG))
		label = keras.Input(shape=(1,), dtype='int32')
		label_embedding1 = Flatten()(Embedding(self.num_classes, self.step_size * self.feature_numG)(label))
		generator_input = multiply([g_input, label_embedding1])
		generator_input = Reshape((self.step_size, self.feature_numG))(generator_input)
		
		x = LSTM(75, return_sequences=True)(generator_input)
		x = Dropout(0.2)(x)
		x = LSTM(25)(x)
		x = Dense(1)(x)
		# x = BatchNormalization(momentum=0.99)(x)
		x = LeakyReLU()(x)
		generator = Model([g_input, label], x)
		generator.summary()
		
		# build discriminator
		d_input = Input(shape=(self.step_size + 1, 1))
		label = Input(shape=(1,), dtype='int32')
		label_embedding2 = Flatten()(Embedding(self.num_classes, self.step_size + 1)(label))
		discriminator_input = multiply([d_input, label_embedding2])
		
		y = Dense(72)(discriminator_input)
		y = LeakyReLU(alpha=0.05)(y)
		y = Dense(100)(y)
		y = LeakyReLU(alpha=0.05)(y)
		y = Dense(10)(y)
		# y = BatchNormalization(momentum=0.99)(y)
		y = LeakyReLU(alpha=0.05)(y)
		y = Dense(1, activation='sigmoid')(y)
		discriminator = Model([d_input,label], y)
		discriminator.summary()
		# 为了训练稳定，在优化器中使用学习率衰减和梯度限幅（按值）。
		discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
		discriminator.compile(optimizer=discriminator_optimizer, loss=['binary_crossentropy'],metrics=['accuracy'])
		
		# build cGAN
		g_input1 = Input(shape=(1, self.step_size * self.feature_numG))
		label1 = Input(shape=(1,), dtype='int32')
		d_input1 = generator([g_input1, label1])
		# 将鉴别器（discrimitor）权重设置为不可训练（仅适用于`gan`模型）作了修改，对于trainable与compile问题
		frozen_D = Model([d_input,label], y)
		frozen_D.trainable = False
		gan_input = [g_input1, label1]
		gan_output = frozen_D([d_input1, label1])
		gan = Model(gan_input, gan_output)
		gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
		gan.compile(optimizer=gan_optimizer, loss=['binary_crossentropy'])
		
		
		# PREPARATION OF TIME SERIES DATASET
		df_list =[]
		for i in range(4):
			df_temp = pd.concat([self.train_list_G[i], self.test_list_G[i]], 0)
			df_list.append(df_temp)
			
		dataset = pd.concat([df_list[0],df_list[1],df_list[2],df_list[3]], 1)
		dataset = dataset.drop_duplicates(keep='last').T.drop_duplicates(keep='last').T
		
		# treat labels! and extract the two variables
		label_data = dataset[['Pct_change', 'Compound_multiplied']]
		# 将情感分析变量转换成标签
		scaler_G1 = MinMaxScaler(feature_range=(-1, 1))
		scaled = scaler_G1.fit_transform(label_data.values)
		labels = []
		for i in range(len(label_data)):
			label = 0.3 * scaled[i][0] + 0.7 * scaled[i][1]
			labels.append(label)
		labels = np.transpose(np.array([labels]))

		scaler_G2 = MinMaxScaler(feature_range=(0, 3))
		scaled = scaler_G2.fit_transform(labels)
		labels = (np.floor(scaled)).astype("int32")
		for i in range(labels.shape[0]):
			if labels[i][0] == 3:
				labels[i][0] = 2
		print(labels)
		
		# treat dataset
		dataset.drop(['Pct_change', 'Compound_multiplied'], axis=1, inplace=True)
		values = dataset.values  # 67*6
		# ensure all data is float
		values = values.astype('float32')
		# normalize features
		scaler = MinMaxScaler(feature_range=(0, 1))
		scaled = scaler.fit_transform(values)
		# frame as supervised learning
		reframed = self.series_to_supervised(scaled, self.step_size, 1)  # 5天预测一天
		print(reframed.shape)  # 一共6个变量，每个变量6列：t-5,t-4,t-3,t-2,t-1,t;shape应为(原长 - step_size)*((step_size+1)*fea_num)
		# split into train and test sets
		values = reframed.values
		
		train = values[:self.n_train_days, :]
		test = values[self.n_train_days:, :]
		train_label = labels[self.step_size - 1:self.n_train_days + self.step_size - 1, :]
		test_label = labels[self.n_train_days - 1:-1, :]
		
		# predict the C price
		# split into input and outputs
		n_obs = self.step_size * self.feature_numG
		train_X, train_Y = train[:, :n_obs], train[:, -(self.feature_numG-3)]  # choose the first feature,namely 'open price'
		test_X, test_Y = test[:, :n_obs], test[:, -(self.feature_numG-3)]  # !!!!!!!!!!!!!!!
		print(train_X.shape, len(train_X), train_Y.shape)  # train_X 应为 280*30,train_Y 应为 280*6
		# reshape input to be 3D [samples, timesteps, features]
		trainX = train_X.reshape((train_X.shape[0], self.step_size, self.feature_numG))
		testX = test_X.reshape((test_X.shape[0], self.step_size, self.feature_numG))
		
		final = []
		# 开始训练迭代
		for step in range(self.iterations1):
			temp_X = copy.deepcopy(trainX[step])
			temp_X = temp_X.reshape(self.batch_size, 1, n_obs)
			temp_Y = copy.deepcopy(train_Y[step])
			temp_Y = temp_Y.reshape(self.batch_size, 1)
			temp_label = copy.deepcopy(train_label[step])
			predictions = generator.predict([temp_X, temp_label])
			# 训练鉴别器（discrimitor）
			for i in range(100):
				aaa = trainX[step]
				input_f = np.concatenate([np.transpose(np.array([aaa[:, 0]])), predictions], 0)
				input_r = np.concatenate([np.transpose(np.array([aaa[:, 0]])), temp_Y], 0)
				d_loss_fake = discriminator.train_on_batch([[input_f], temp_label],np.array([np.ones((self.step_size + 1, 1))]))
				d_loss_real = discriminator.train_on_batch([[input_r], temp_label],np.array([np.zeros((self.step_size + 1, 1))]))
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
			# 训练生成器（generator）（通过gan模型，鉴别器（discrimitor）权值被冻结）
			for i in range(20):
				misleading_targets = np.zeros((self.batch_size, 1))
				g_loss = gan.train_on_batch([temp_X, temp_label], [misleading_targets])
			print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step, d_loss[0], 100 * d_loss[1], g_loss))
			final.append(predictions[0])
		
		final = np.concatenate((train_X[:,-4:-1],np.array(final)), axis=1)
		final2 = np.concatenate((train_X[:,-4:-1],np.transpose(np.array([train_Y]))), axis=1)
		int1 = scaler.inverse_transform(final)
		int2 = scaler.inverse_transform(final2)
		MAPE_O1 = np.mean(np.abs((int2[:, 3] - int1[:, 3]))/ int2[:, 3])
		print(MAPE_O1)
		
		# 得到测试集结果
		final = []
		for step in range(self.iterations2):
			temp_X = copy.deepcopy(testX[step])
			temp_X = temp_X.reshape(self.batch_size, 1, n_obs)
			temp_label = copy.deepcopy(test_label[step])
			predictions = generator.predict([temp_X, temp_label])
			final.append(predictions[0])
		final = np.concatenate((test_X[:,-4:-1], np.array(final)), axis=1)
		final2 = np.concatenate((test_X[:,-4:-1], np.transpose(np.array([test_Y]))), axis=1)
		int1 = scaler.inverse_transform(final)
		int2 = scaler.inverse_transform(final2)
		MAPE_O2 = np.mean(np.abs((int2[:, 3] - int1[:, 3]))/ int2[:, 3])
		print(MAPE_O2)
		
	def main(self):
		""" Driver function that calls the other functions
		"""
		t0_clock = time.process_time()
		rcParams['figure.figsize'] = 20, 10
		
		# load, clean, and enrich data
		self.load_data()
		self.join_data()
		self.transform_data()
		self.clean_data()
		
		#  training and evalutoin and visualization of improving model:LSTM
		self.nn_preprocess()
		self.train_cGAN()
		
		t1_clock = time.process_time()
		print("\nTotal CPU time = %.1f minutes." % round((t1_clock - t0_clock) / 60, 1))

if __name__ == '__main__':
	ap = asset_prices()
	ap.main()
	import doctest
	
	doctest.testmod()
