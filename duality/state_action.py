'''
This file implements the naive q-learning algorithm.
Basically, we use neural network to predict
the reward of each action from different state vectors
and select the action with highest reward.
Then we train the whole model end-to-end.
'''
import keras
from keras.layers import *
import keras.backend as K
state_shape = (3,3)
action_shape = (10)
ACTION_NUM=10
state_action_reward_layers =[
    Dense(128),
    Activation('relu'),
    BatchNormalization(),
    Dense(ACTION_NUM),
    Activation('tanh')
]
s_a_r_model = keras.models.Sequential(state_action_reward_layers)

State = Input(shape=state_shape)
Action_Rewards = s_a_r_model(State)
Reward = Lambda(lambda x: K.max(x,1))
whole_model = keras.models.Model(State,Reward)

def train_model(model,train_X,train_Y,test_X,test_Y):
    model.compile(loss='mae',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(train_X,train_Y,
              batch_size=32,
              nb_epoch=10,
              verbose=1,
              validation_data=(test_X,test_Y))

