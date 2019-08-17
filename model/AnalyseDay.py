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
LOAD_MEAN_BATTERY_STATE = 'mb'
LOAD_VARIANCE_BATTERY_STATE = 'vb'

RANDOMIZE_BATTERY = 'rb'#True
LEARNING_RATE = 'lr'#0.03
DISCOUNT_FACTOR = 'df'#0.9
NUM_DUM_LOADS = 'ndl'#9
DAY = 'day'#199999
MODE = 'mode'
STATES = 'state'
MOVING_BUCKETS = 'movingbuckets'




load_agent_params = [
                    {
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:9999,
                        MODE:'vanilla',
                        STATES:['b20', 'p10'],
                        MOVING_BUCKETS: True
                     },
{
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:9999,
                        MODE:'vanilla',
                        STATES:['b101', 'p10'],
                        MOVING_BUCKETS: True
                     },

                     ]


NUM_AGENTS = len(load_agent_params)
load_agent_dict = {}

for i in range(NUM_AGENTS):
    # MODEL_PATH = os.getcwd()
    # MODEL_PATH += '/basic_qlearning_models'
    # if load_agent_params[i][STATES] >0:
    #     MODEL_PATH+='/'+str(load_agent_params[i][STATES])+'states'
    #     if load_agent_params[i][MOVING_BUCKETS]:
    #         MODEL_PATH += '/moving_buckets'
    #     else:
    #         MODEL_PATH += '/static_buckets'
    # if load_agent_params[i][RANDOMIZE_BATTERY]:
    #     MODEL_PATH+='/randomize_battery'
    # else:
    #     MODEL_PATH+='/continuous_battery'
    # MODEL_PATH+= '/dumloads'+str(load_agent_params[i][NUM_DUM_LOADS]) + \
    #              '/df' + str(load_agent_params[i][DISCOUNT_FACTOR]) + \
    #              '/lr'+str(load_agent_params[i][LEARNING_RATE])
    #
    #
    # # with open(MODEL_PATH+'/agent_'+str(load_agent_params[i][DAY])+'.pickle', 'rb') as f:
    # with open(MODEL_PATH + '/' + load_agent_params[i][MODE] + '_agent_' + str(load_agent_params[i][DAY]) + '.pickle',
    #           'rb') as f:
    #     load_agent_dict[i] = pickle.load(f)
    load_agent_dict[i] = read_model(load_agent_params[i])

def setup():
    env = Environment()
    env.add_connections({0:range((NUM_AGENTS+2))})
    env.add_dumb_loads(0,1000)
    env.set_environment_ready(test_mode=True)
    env.reset(0)
    # load_agent_dict = {0:QTableAgent(env.get_load_action_space(),
    #                                  {LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0)},
    #                                  {LOAD_BATTERY_STATE:20, LOAD_PRICE_STATE:10},
    #                                  default_action=1,
    #                                  discount_factor=DISCOUNT_FACTOR
    #                                 )}
    # load_agent_dict[0].set_learning_rate(LEARNING_RATE)
    for i in load_agent_dict.keys():
        load_agent_dict[i].set_explore_rate(0)
    return env

def run_day():
    current_states = dict((i, {}) for i in range(NUM_AGENTS + 2))
    actions = dict((k,v) for k,v in enumerate([0]*(NUM_AGENTS+2)))
    # rewards = [0.0]*(NUM_AGENTS+2)
    prices = []
    demands = dict((k,[]) for k in range(NUM_AGENTS+2))
    actions[NUM_AGENTS+1] = 1                                       # Constant Agent

    response = env.reset()
    for k in range(NUM_AGENTS):
        current_states[k] = {LOAD_BATTERY_STATE: response[1][k][0][0],
                             LOAD_PRICE_STATE: response[1][k][0][1][-1],
                             LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                             LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                             }
    for i in range(NUM_AGENTS + 2):
        demands[i].append(response[1][i][1][0])
    # prices.append(response[1][0][1][0])

    for step in range(env.get_max_timestep() + 1):
        for k in range(NUM_AGENTS):
            # actions[k] = load_agent_dict[k].take_action()
            actions[k] = load_agent_dict[k].get_action(current_states[k])
        actions[NUM_AGENTS] = randint(0, 2)  # Random Agent
        response = env.step(loadActionDict=actions)
        # print(response[1][0][1][0])
        for k in range(NUM_AGENTS+2):
            current_states[k] = {LOAD_BATTERY_STATE: response[1][k][0][0],
                      LOAD_PRICE_STATE: response[1][k][0][1][-1],
                      LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                      LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                      }
        for i in range(NUM_AGENTS + 2):
            demands[i].append(response[1][i][1][1]+response[1][i][1][0])
        prices.append(response[1][0][1][2])

    return demands, prices


def smooth(a, win = 10):
    b = np.zeros_like(a[:-win])
    for i in range(a.shape[0]-win):
        b[i] = a[i:i+win].mean()
    return b


env = setup()

demands, prices = run_day()
colors = ['y','b','g','r','k','r:']
# window_size = 10
smooth_demands = {}
for k in demands.keys():
    demands[k] = np.array(demands[k])
    smooth_demands[k] = smooth(demands[k])
prices = np.array(prices)
smooth_prices = smooth(prices)

labels_dict = {0: 'b20p10', 1:'b101p10', 2:'random', 3:'action 1'}
for i,k in enumerate(demands.keys()):
    plt.plot(demands[k][1:], colors[i], label=labels_dict[k])
    plt.xlabel("Time in minutes")
    plt.ylabel("Performance of agent")
    plt.legend()
    print(k, colors[i], sum(prices*demands[k][1:]))

# for i,k in enumerate(demands.keys()):
#     plt.plot(smooth_demands[k][1:], colors[i], label=i+k)
#     plt.legend(loc='upper left')
#     print(k, colors[i], sum(smooth_prices*smooth_demands[k][1:]))
