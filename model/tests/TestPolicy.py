from Environment import Environment
import os
import numpy as np
from QTableAgent import QTableAgent
import pickle
from random import randint
from matplotlib import pyplot as plt



LOAD_BATTERY_STATE = 'battery'
LOAD_PRICE_STATE = 'price'
RANDOMIZE_BATTERY = 'rb'#True
LEARNING_RATE = 'lr'#0.03
DISCOUNT_FACTOR = 'df'#0.9
NUM_DUM_LOADS = 'ndl'#9
DAY = 'day'#199999
MODE = 'mode'

load_agent_params = [#{
#                         RANDOMIZE_BATTERY:True,
#                         LEARNING_RATE: 0.03,
#                         DISCOUNT_FACTOR: 0.9,
#                         NUM_DUM_LOADS:0,
#                         DAY:9999,
#                         MODE:'sarsa'
#                      },
#                     {
#                         RANDOMIZE_BATTERY:True,
#                         LEARNING_RATE: 0.03,
#                         DISCOUNT_FACTOR: 0.95,
#                         NUM_DUM_LOADS:0,
#                         DAY:99999,
#                         MODE:'sarsa'
#                      },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.95,
                    #     NUM_DUM_LOADS:9,
                    #     DAY:99999,
                    #     MODE:'sarsa'
                    #  },
                    {
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.03,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:9999,
                        MODE:'vanilla'
                     },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.95,
                    #     NUM_DUM_LOADS:999,
                    #     DAY:9999,
                    #     MODE:'vanilla'
                    #  },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.9 ,
                    #     NUM_DUM_LOADS:9,
                    #     DAY:99999,
                    #     MODE:'vanilla'
                    #  },
                     ]
NUM_AGENTS = len(load_agent_params)
load_agent_dict = {}

for i in range(NUM_AGENTS):
    MODEL_PATH = os.getcwd()
    MODEL_PATH += '/sharpe_ratio_models/2states/moving_buckets'
    if load_agent_params[i][RANDOMIZE_BATTERY]:
        MODEL_PATH+='/randomize_battery'
    else:
        MODEL_PATH+='/continuous_battery'
    MODEL_PATH+= '/dumloads'+str(load_agent_params[i][NUM_DUM_LOADS]) + \
                 '/df' + str(load_agent_params[i][DISCOUNT_FACTOR]) + \
                 '/lr'+str(load_agent_params[i][LEARNING_RATE])

    # policy = np.load(MODEL_PATH + '/policy_'+str(DAY) + '.npy')
    # with open(MODEL_PATH+'/agent_'+str(load_agent_params[i][DAY])+'.pickle', 'rb') as f:
    with open(MODEL_PATH + '/' + load_agent_params[i][MODE] + '_agent_' + str(load_agent_params[i][DAY]) + '.pickle', 'rb') as f:
        load_agent_dict[i] = pickle.load(f)

def setup():
    env = Environment()
    env.add_connections({0:range(NUM_AGENTS+2)})
    env.add_dumb_loads(0,1000)
    env.set_environment_ready()
    env.reset(0)
    load_agent_dict = {0:QTableAgent(env.get_load_action_space(),
                                     {LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0)},
                                     {LOAD_BATTERY_STATE:20, LOAD_PRICE_STATE:10},
                                     default_action=1,
                                     discount_factor=DISCOUNT_FACTOR
                                    )}
    load_agent_dict[0].set_learning_rate(LEARNING_RATE)
    for i in load_agent_dict.keys():
        load_agent_dict[i].set_explore_rate(0)
    return env

def calculate_rewards(startday = 0, endday = 100):
    rewards = [0.0]*(NUM_AGENTS+2)
    rewardslist = []
    actions = dict((k,v) for k,v in enumerate([0]*(NUM_AGENTS+2)))
    actions[-2] = 1
    for day in range(startday, endday):
        response = env.reset()
        for k in range(NUM_AGENTS):
            current_state = {LOAD_BATTERY_STATE: response[1][k][0][0], LOAD_PRICE_STATE: response[1][k][0][1][-1]} #loadFeedbackDict, loadID, observation, battery/price
            # load_agent_dict[k].update_state(current_state)
        for step in range(env.get_max_timestep()+1):
            for k in range(NUM_AGENTS):
                actions[k] = load_agent_dict[k].get_action(current_state)
            actions[-1] = randint(0,2)
            response = env.step(loadActionDict=actions)
            for k in range(NUM_AGENTS):
                current_state = {LOAD_BATTERY_STATE: response[1][k][0][0], LOAD_PRICE_STATE: response[1][k][0][1][
                    -1]}  # loadFeedbackDict, loadID, observation, battery/price
                # load_agent_dict[k].update_state(current_state)
            for i in range(NUM_AGENTS+2):
                rewards[i] += response[1][i][1]/1000
        print(day, rewards)
        rewardslist.append(rewards[:])

    return rewardslist



env = setup()
rewards = calculate_rewards()

rewards = np.array(rewards)
colors = ['y','b','g','r','k','r:']
plt.axes().set(title="Agent Performance",xlabel = "Day", ylabel = "Overall Cost of Power")
for i in range(len(rewards[0])):
    # plt.figure(i)
    plt.plot(range(len(rewards)),rewards[:,i],colors[i])
#
