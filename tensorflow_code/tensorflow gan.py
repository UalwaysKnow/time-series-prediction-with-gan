import pandas as pd
import numpy as np
import os
import time
import pickle

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class GANgraph():
	
	def xavier_init(self, size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		aaa = tf.random_normal(shape=size, stddev=xavier_stddev, dtype=tf.float64)
		return tf.to_float(aaa)
	
	def init_paras_3hidden(self, sess):
		h1_dim = 72
		h2_dim = 100
		h3_dim = 10
		h4_dim = 1
		input_dim = tf.to_int32(self.step_size+1).eval(session=sess)
		h1_dim = tf.to_int32(h1_dim).eval(session=sess)
		h2_dim = tf.to_int32(h2_dim).eval(session=sess)
		h3_dim = tf.to_int32(h3_dim).eval(session=sess)
		h4_dim = tf.to_int32(h4_dim).eval(session=sess)
		W1 = tf.Variable(self.xavier_init([input_dim, h1_dim]))
		W2 = tf.Variable(self.xavier_init([h1_dim, h2_dim]))
		W3 = tf.Variable(self.xavier_init([h2_dim, h3_dim]))
		W4 = tf.Variable(self.xavier_init([h3_dim, h4_dim]))
		b1 = tf.Variable(tf.zeros(shape=[h1_dim]))
		b2 = tf.Variable(tf.zeros(shape=[h2_dim]))
		b3 = tf.Variable(tf.zeros(shape=[h3_dim]))
		b4 = tf.Variable(tf.zeros(shape=[h4_dim]))
		return W1, W2, W3, W4, b1, b2, b3 ,b4
	
	def __init__(self, sess):
		self.step_size = 5
		self.batch_size = self.step_size
		self.feature_num = 6
		self.lambda1 = 0.5
		self.lambda2 = 0.5
		self.learning_rate = 1e-6
		self.filename1 = 'generator_weights.h5'
		self.filename2 = 'discriminator_weights.pkl'
		if not os.path.exists(os.path.dirname(self.filename1)):
			self.param_G = 0
		else:
			self.param_G = 1
		if not os.path.exists(os.path.dirname(self.filename2)):
			self.param_D = None
		else:
			self.param_D = pickle.load(open(self.filename2,'rb'))
		
		self.train(sess)
	
	def discriminator(self, input, D):
		W1 = D[0]
		W2 = D[1]
		W3 = D[2]
		W4 = D[3]
		b1 = D[4]
		b2 = D[5]
		b3 = D[6]
		b4 = D[7]
		D_h1 = tf.nn.leaky_relu(tf.matmul(input, W1) + b1,alpha=0.2)
		D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, W2) + b2,alpha=0.2)
		D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, W3) + b3,alpha=0.2)
		out = tf.nn.sigmoid(tf.matmul(D_h3, W4) + b4)
		return out
	# def discriminator(self,input,sess):
	# 	#discriminator；应该输出d_real,d_fake;输出应为1*6
	# 	y = Dense(72)(input)
	# 	y = LeakyReLU(alpha=0.05)(y)
	# 	y = Dense(100)(y)
	# 	y = LeakyReLU(alpha=0.05)(y)
	# 	y = Dense(10)(y)
	# 	y = LeakyReLU(alpha=0.05)(y)
	# 	y = Dense(1)(y)
	# 	# self.discriminator_pre = Activation('sigmoid')(y)
	# 	return ...
	
	# def generator(self,input,sess):
	# 	#generator;每次feed 6个属性值，由前五天预测第六天;应该输出1*6
	# 	x = LSTM(32)(input)
	# 	x = Dropout(0.2)(x)
	# 	x = LSTM(16)(x)
	# 	x = Dense(6)(x)
	# 	self.generator_pre = LeakyReLU(alpha=0.05)(x)
	def train(self,sess):
		#param_G 初始化前参数，G 初始化后参数，param-gen 保存参数
		#tf.reset_default_graph()
		# INPUT OF TRAINING for generator and discriminator
		self.input = tf.placeholder(tf.float32,shape=(1,self.batch_size,self.feature_num))
		self.labels = tf.placeholder(tf.float32,shape=(1,self.feature_num))
		
		# #generator model;每次feed 6个属性值，由前五天预测第六天;应该输出1*6
		# input = Input(shape=(self.step_size, self.feature_num))
		# x = LSTM(32, return_sequences = True)(input)
		# x = Dropout(0.2)(x)
		# x = LSTM(16)(x)
		# x = Dense(6)(x)
		# generator_pre = LeakyReLU(alpha=0.05)(x)
		# generator = Model(inputs=input, outputs=generator_pre)

		# with tf.variable_scope('generator'):
		# 	if self.param_G == 0:
		# 		self.param_gen = generator.get_weights()
		# 	else:
		# 		generator.load_weights(self.filename1)
		# 		self.param_gen = generator.get_weights()
		#
		# 	self.G_temp = [0,0,0,0,0,0,0,0]
		# 	self.G = []
		# 	for i in range(8):
		# 		self.G_temp[i] = tf.convert_to_tensor(self.param_gen[i])
		# 		self.G.append(tf.Variable(self.G_temp[i]))
		# self.prediction = generator_pre
		
		# with tf.variable_scope('lstm'):
		# 	lstm = tf.contrib.rnn.OutputProjectionWrapper(
		# 		tf.contrib.rnn.LSTMCell(num_units=50, activation=tf.nn.relu), output_size=1)
		# 	Y_hat, states = tf.nn.dynamic_rnn(lstm, self.input, dtype=tf.float32)
		with tf.variable_scope('lstm'):
			rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
			outputs, final_state = tf.nn.dynamic_rnn(
				cell=rnn_cell,  # 选择传入的cell
				inputs=self.input,  # 传入的数据
				initial_state=None,  # 初始状态
				dtype=tf.float32,  # 数据类型
				time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
			)
			self.prediction = tf.layers.dense(inputs=outputs[:, -1, :], units=6)
		
		with tf.variable_scope('discriminator'):
			if self.param_D == None:
				W1,W2,W3,W4,b1,b2,b3,b4 = self.init_paras_3hidden(sess)
				self.D = [W1,W2,W3,W4,b1,b2,b3,b4]
			else:
				self.D = tf.Variable(self.param_D,name='discriminator')
		
		#self.params_gen = self.G
		self.params_dis = self.D
		
		self.input_f = tf.concat([self.input[0],self.prediction],0)
		self.input_r = tf.concat([self.input[0],self.labels],0)
		
		self.pre_f = self.discriminator(tf.transpose(self.input_f),self.D)
		self.pre_r = self.discriminator(tf.transpose(self.input_r),self.D)
		self.g_MSE = tf.reduce_mean(tf.square(self.prediction - self.labels))
		self.g_los = tf.reduce_mean(tf.log(1.0 - self.pre_f))
		self.g_loss = self.lambda1 * self.g_MSE + self.lambda2 * self.g_los
		self.d_loss = tf.reduce_mean(tf.log(self.pre_r)) + tf.reduce_mean(tf.log(1.0 - self.pre_f))

		self.d_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) \
			.minimize(-self.d_loss, var_list=self.D)
		self.g_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) \
			.minimize(self.g_loss)
		self.D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D]
		self.D = self.D_clip
	
	def save_params_dis(self, sess):
		param = sess.run(self.params_dis)
		pickle.dump(param,open(self.filename2, "wb"))
		
	# def save_params_gen(self, sess):
	# 	param = sess.run(self.params_gen)
	# 	pickle.dump(param, open(self.filename1, 'w'))
	
def series_to_supervised(data,n_in=1, n_out=1, dropnan=True):
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
	
def data_preprocessing(dataset,step_size):
	'''load data and transform data into trainset and testset
		features: O,H,L,C ,(H-L)/L,(C-O)/O
		'''
	dataset['Date'] = pd.to_datetime(dataset.Date, format='%Y-%m-%d')
	dataset.index = dataset['Date']
	dataset = dataset.sort_index(ascending=True, axis=0)
	dataset.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
	print(dataset.head())
	# # drop the first 24 hours !!!why drop frist 24 hours' data
	# dataset = dataset[24:]
	# summarize first 5 rows
	
	# add new 2 features
	dataset['HL_PCT'] = (dataset['High'] - dataset['Low']) / dataset['Low'] * 100.0
	dataset['PCT_change'] = (dataset['Close'] - dataset['Open']) / dataset['Open'] * 100.0
	
	dataset.drop(['Date'],axis=1,inplace=True)
	values = dataset.values
	# ensure all data is float
	values = values.astype('float32')
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	
	# frame as supervised learning
	reframed = series_to_supervised(scaled, step_size, 1) #5天预测一天
	print(reframed.shape)  # 一共6个变量，每个变量6列：t-5,t-4,t-3,t-2,t-1,t;shape应为375*36
	
	# split into train and test sets
	values = reframed.values
	n_train_days = 280
	train = values[:n_train_days, :]
	test = values[n_train_days:, :]
	
	return train, test,scaler

def main():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = GANgraph(sess)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	localtime = time.asctime(time.localtime(time.time()))
	print('Begining Time :', localtime)
	
	step_size = 5
	feature_num = 6
	
	# PREPARATION OF TIME SERIES DATASET
	dataset = pd.read_csv('AAPL 1.5year.csv') # header=0, index_col=0 # 将第0行作为列标题；将第0列作为行标题
	train,test,scaler = data_preprocessing(dataset,step_size)
	# split into input and outputs
	n_obs = step_size * feature_num
	train_X, train_Y = train[:, :n_obs], train[:, -feature_num:]
	test_X, test_Y = test[:, :n_obs], test[:, -feature_num:]  # !!!!!!!!!!!!!!!
	print(train_X.shape, len(train_X), train_Y.shape) #train_X 应为 280*30,train_Y 应为 280*6
	# reshape input to be 3D [samples, timesteps, features]
	trainX = train_X.reshape((train_X.shape[0], step_size, feature_num))
	
	predictions = []
	
	for epoch_all in range(280):#len(train_X)-step_size-1
		#取一个batch,即5*6的大小
		trainX_batch = [trainX[epoch_all]]
		trainY_batch = [train_Y[epoch_all]]
		
		feed_dict = {model.input: trainX_batch, model.labels: trainY_batch}
		# discriminator
		_,d_loss = sess.run([model.d_optim,model.d_loss], feed_dict=feed_dict)
		#generator
		for epoch_dis in range(5):
			_,g_loss,g_MSE,prediction = sess.run([model.g_optim,model.g_loss,model.g_MSE,model.prediction],feed_dict=feed_dict)
		predictions.append(prediction[0])
		if epoch_all%10 == 0:
			print(epoch_all,g_loss,d_loss,g_MSE)
		#model.save_params_gen(sess)
		model.save_params_dis(sess)
	
	int1 = scaler.inverse_transform(predictions)
	int2 = scaler.inverse_transform(train_Y)
	MAPE = []
	for i in range(6):
		aaa = int1[:,i]
		bbb = int2[:,i]
		MAPE.append(np.mean(np.abs((bbb - aaa) / bbb)))
	print('training result:',MAPE[0])
	
	localtime = time.asctime(time.localtime(time.time()))
	print('Endding Time :', localtime)

if __name__ == "__main__":
	main()