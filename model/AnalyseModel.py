from Environment import Environment
import os
import numpy as np
from QTableAgent import QTableAgent
import pickle
from random import randint
from matplotlib import pyplot as plt
from Utils import read_model


LOAD_BATTERY_STATE = 'battery'
LOAD_PRICE_STATE = 'price'
RANDOMIZE_BATTERY = 'rb'#True
LEARNING_RATE = 'lr'#0.03
DISCOUNT_FACTOR = 'df'#0.9
NUM_DUM_LOADS = 'ndl'#9
DAY = 'day'#199999
MODE = 'mode'
STATES = 'state'
MOVING_BUCKETS = 'movingbuckets'


model_params = {
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:1999,
                        MODE:'vanilla',
                        STATES:['b100','p10'],
                        MOVING_BUCKETS: True,
                }

agent = {}
for day in range(1999,39999,2000):
    # MODEL_PATH = os.getcwd()
    # MODEL_PATH += '/basic_qlearning_models'
    # if model_params[STATES] > 0:
    #     MODEL_PATH += '/' + str(model_params[STATES]) + 'states'
    #     if model_params[MOVING_BUCKETS]:
    #         MODEL_PATH += '/moving_buckets'
    #     else:
    #         MODEL_PATH += '/static_buckets'
    # if model_params[RANDOMIZE_BATTERY]:
    #     MODEL_PATH+='/randomize_battery'
    # else:
    #     MODEL_PATH+='/continuous_battery'
    # MODEL_PATH+= '/dumloads'+str(model_params[NUM_DUM_LOADS]) + \
    #              '/df' + str(model_params[DISCOUNT_FACTOR]) + \
    #              '/lr'+str(model_params[LEARNING_RATE])
    #
    # # policy = np.load(MODEL_PATH + '/policy_'+str(DAY) + '.npy')
    # # with open(MODEL_PATH+'/agent_'+str(model_params[DAY])+'.pickle', 'rb') as f:
    # with open(MODEL_PATH + '/' + model_params[MODE] + '_agent_' + str(day) + '.pickle', 'rb') as f:
    #     agent[day] = pickle.load(f)
    model_params[DAY] = day
    agent[day] = read_model(model_params)

averages = []
avg0, avg1, avg2 = [],[],[]


for a in agent.keys():
    averages.append(np.mean(agent[a].qtable))
    avg0.append(np.mean(agent[a].qtable[:,:,0]))
    avg1.append(np.mean(agent[a].qtable[:,:,1]))
    avg2.append(np.mean(agent[a].qtable[:,:,2]))

plt.axes().set(title="Learning Curve - 4 States Static Buckets Vanilla Model",xlabel = "Day", ylabel = "Average Q-value")
plt.plot(range(1999,39999,2000),averages,'k:')
# plt.figure(1)
plt.plot(range(1999,39999,2000),avg0,'b')
# plt.figure(2)
plt.plot(range(1999,39999,2000),avg1,'r')
# plt.figure(3)
plt.plot(range(1999,39999,2000),avg2,'g')
