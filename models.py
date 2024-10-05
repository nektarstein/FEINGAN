import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Reshape,Flatten, Dropout, Lambda, RepeatVector
from tensorflow.keras.layers import Add,Multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling1D, Conv1D
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam,Nadam
from keras.utils import plot_model
#from keras.utils.training_utils import multi_gpu_model
from keras import backend as K

def generator_model_mlp_bn():
  input_noise = Input(shape=(100,))
  model = Dense(128,kernel_initializer='random_uniform')(input_noise)
  model = BatchNormalization()(model)
  #model = LeakyReLU()(model)
  model = Activation('tanh')(model)
  model = Dense(2048,kernel_initializer='random_uniform')(model)
  model = BatchNormalization()(model)
  model = Activation('tanh')(model)
  model = Dense(8192,kernel_initializer='random_uniform')(model)
  #model = BatchNormalization()(model)
  model = Activation('tanh')(model)
  model = Reshape((8192,1))(model)
  from keras.layers import Lambda
  def mean_computation(x):
    return K.mean(x,axis=1)

  def mean_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          
  
  def std_computation(x):
    return K.std(x,axis=1)

  def std_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          

  mean_layer = Lambda(mean_computation,output_shape=mean_computation_output_shape)
  std_layer = Lambda(std_computation,output_shape=std_computation_output_shape)
  mean = mean_layer(model)
  std = std_layer(model)
  model = Model(input_noise,model)
  model_statistics = Model(input_noise,[mean,std])
  return model,model_statistics

def generator_model_mlp():
  input_noise = Input(shape=(100,))
  model = Dense(128,kernel_initializer='random_uniform')(input_noise)
  #model = BatchNormalization()(model)
  #model = LeakyReLU()(model)
  model = Activation('tanh')(model)
  model = Dense(2048,kernel_initializer='random_uniform')(model)
  #model = BatchNormalization()(model)
  model = Activation('tanh')(model)
  model = Dense(8192,kernel_initializer='random_uniform')(model)
  #model = BatchNormalization()(model)
  model = Activation('tanh')(model)
  model = Reshape((8192,1))(model)
  from keras.layers import Lambda
  def mean_computation(x):
    return K.mean(x,axis=1)

  def mean_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          
  
  def std_computation(x):
    return K.std(x,axis=1)

  def std_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          

  mean_layer = Lambda(mean_computation,output_shape=mean_computation_output_shape)
  std_layer = Lambda(std_computation,output_shape=std_computation_output_shape)
  mean = mean_layer(model)
  std = std_layer(model)
  model = Model(input_noise,model)
  model_statistics = Model(input_noise,[mean,std])
  return model,model_statistics

def generator_model_cnn():
  input_noise = Input(shape=(100,))
  model = Dense(128)(input_noise)
  model_1 = Reshape((128,1))(model)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,35,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,25,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1_1 = Conv1D(64,7,padding='same')(model_1)
  model_1_1_1 = Conv1D(64,4,padding='same')(model_1)
  model_1 = Add()([model_1,model_1_1,model_1_1_1])
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = Conv1D(1,1,padding='same')(model_1)
  model_1 = Activation('tanh')(model_1)
  model = Model(input_noise,model_1)
  return model

def generator_model_mlp_cnn():
  input_noise = Input(shape=(100,))
  model = Dense(128)(input_noise)
  model_1 = Reshape((128,1))(model)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,35,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,25,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1_1 = Conv1D(64,7,padding='same')(model_1)
  model_1_1_1 = Conv1D(64,4,padding='same')(model_1)
  model_1 = Add()([model_1,model_1_1,model_1_1_1])
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = Conv1D(1,1,padding='same')(model_1)
  model_1 = Activation('tanh')(model_1)
  model_2 = Dense(8192)(model)
  model_2 = Activation('tanh')(model_2)
  model_2 = Reshape((8192,1))(model_2)
  model = Multiply()([model_1,model_2])
  model = Model(input_noise,model)
  return model

def generator_model_mlp_cnn_plus():
  input_noise = Input(shape=(100,))
  model = Dense(128)(input_noise)
  model_1 = Reshape((128,1))(model)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,35,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,25,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1_1 = Conv1D(64,7,padding='same')(model_1)
  model_1_1_1 = Conv1D(64,4,padding='same')(model_1)
  model_1 = Add()([model_1,model_1_1,model_1_1_1])
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = Conv1D(1,1,padding='same')(model_1)
  model_1 = Activation('tanh')(model_1)
  model_2 = Dense(8192)(model)
  model_2 = Activation('tanh')(model_2)
  model_2 = Reshape((8192,1))(model_2)
  model_3 = Dense(1)(input_noise)
  model_3 = Activation('sigmoid')(model_3)
  from keras.layers import RepeatVector
  model_3 = RepeatVector(8192)(model_3)
  model = Multiply()([model_1,model_2,model_3])
  model = Model(input_noise,model)
  return model

def generator_model_mlp_cnn():
  input_noise = Input(shape=(100,))
  model = Dense(128)(input_noise)
  model_1 = Reshape((128,1))(model)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,35,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,25,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = UpSampling1D(2) (model_1)
  model_1 = Conv1D(64,15,padding='same')(model_1)
  model_1_1 = Conv1D(64,7,padding='same')(model_1)
  model_1_1_1 = Conv1D(64,4,padding='same')(model_1)
  model_1 = Add()([model_1,model_1_1,model_1_1_1])
  model_1 = BatchNormalization()(model_1)
  model_1 = LeakyReLU()(model_1)
  model_1 = Conv1D(1,1,padding='same')(model_1)
  model_1 = Activation('tanh')(model_1)
  model_2 = Dense(8192)(model)
  model_2 = Activation('tanh')(model_2)
  model_2 = Reshape((8192,1))(model_2)
  model = Multiply()([model_1,model_2])
  from keras.layers import Lambda
  def mean_computation(x):
    return K.mean(x,axis=1)

  def mean_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          
  
  def std_computation(x):
    return K.std(x,axis=1)

  def std_computation_output_shape(input_shape):
    new_shape = tuple([input_shape[0],input_shape[-1]])
    return new_shape                                          

  mean_layer = Lambda(mean_computation,output_shape=mean_computation_output_shape)
  std_layer = Lambda(std_computation,output_shape=std_computation_output_shape)
  mean = mean_layer(model)
  std = std_layer(model)
  model = Model(input_noise,model)
  model_statistics = Model(input_noise,[mean,std])
  return model,model_statistics

def discriminator_model():
  model = Sequential()
  model.add(Conv1D(64,10,padding='same',input_shape=(8192,1)))
  model.add(LeakyReLU(0.2))
  model.add(Conv1D(128,10,padding='same'))
  model.add(LeakyReLU(0.2))
  model.add(Conv1D(128,10,padding='same'))
  model.add(LeakyReLU(0.2))
  model.add(Flatten())
  model.add(Dense(32))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  return model
