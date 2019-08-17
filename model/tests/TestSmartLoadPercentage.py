# TODO add initial offset to calculate breakeven for different battery capacities
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
PF = '0'

NUM_COPIES = 1

PENALTY_FACTOR = 0

load_agent_params = {
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:9999,
                        MODE:'vanilla',
                        STATES:['b20','p10'],
                        MOVING_BUCKETS: False,
                        PF:0.5
                     }
NUM_AGENTS = 1
# load_agent_dict = {}
#
# MODEL_PATH = os.getcwd()
# MODEL_PATH += '/basic_qlearning_models'
# if load_agent_params[STATES] >0:
#     MODEL_PATH+='/'+str(load_agent_params[STATES])+'states'
#     if load_agent_params[MOVING_BUCKETS]:
#         MODEL_PATH += '/moving_buckets'
#     else:
#         MODEL_PATH += '/static_buckets'
# if load_agent_params[RANDOMIZE_BATTERY]:
#     MODEL_PATH+='/randomize_battery'
# else:
#     MODEL_PATH+='/continuous_battery'
# MODEL_PATH+= '/dumloads'+str(load_agent_params[NUM_DUM_LOADS]) + \
#              '/df' + str(load_agent_params[DISCOUNT_FACTOR]) + \
#              '/lr'+str(load_agent_params[LEARNING_RATE]) #+ \
#              # '/pf_'+str(load_agent_params[i][PF])
#
#     # policy = np.load(MODEL_PATH + '/policy_'+str(DAY) + '.npy')
#     # with open(MODEL_PATH+'/agent_'+str(load_agent_params[i][DAY])+'.pickle', 'rb') as f:
# with open(MODEL_PATH + '/' + load_agent_params[MODE] + '_agent_' + str(load_agent_params[DAY]) + '.pickle', 'rb') as f:
#     load_agent = pickle.load(f)

load_agent = read_model(load_agent_params)


def capacity_function(i):
    return 0.02*i+0.01

def charging_rate_function(i):
    return 0.1*(i+1)

def setup(num_dumb_loads):
    env = Environment()
    env.add_connections({0:range((NUM_AGENTS*NUM_COPIES+3))})
    env.add_dumb_loads(0,num_dumb_loads)
    env.set_environment_ready(test_mode=True)
    env.reset(0)
    # load_agent_dict = {0:QTableAgent(env.get_load_action_space(),
    #                                  {LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0)},
    #                                  {LOAD_BATTERY_STATE:20, LOAD_PRICE_STATE:10},
    #                                  default_action=1,
    #                                  discount_factor=DISCOUNT_FACTOR
    #                                 )}
    # load_agent_dict[0].set_learning_rate(LEARNING_RATE)
    load_agent.set_explore_rate(0)
    return env

def calculate_rewards(startday = 0, endday = 100):
    rewards = [0.0]*(NUM_AGENTS*NUM_COPIES+3)
    rewardslist = []
    actions = dict((k,v) for k,v in enumerate([0]*(NUM_AGENTS*NUM_COPIES+3)))
    current_states = dict((i,{}) for i in range(NUM_AGENTS*NUM_COPIES+3))
    actions[NUM_AGENTS*NUM_COPIES+1] = 1                                       # Constant Agent
    for day in range(startday, endday):
        response = env.reset()
        for k in range(NUM_AGENTS):
            for j in range(NUM_COPIES):
                current_states[k*NUM_COPIES+j] = {LOAD_BATTERY_STATE: response[1][k*NUM_COPIES+j][0][0],
                              LOAD_PRICE_STATE: response[1][k*NUM_COPIES+j][0][1][-1],
                              # LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                              # LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                              } #loadFeedbackDict, loadID, observation, battery/price
        current_states[NUM_AGENTS*NUM_COPIES+2] = {LOAD_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][0],
                          LOAD_PRICE_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][1][-1],
                          # LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                          # LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                          }
            # load_agent_dict[k].update_state(current_state)
            # current_state = load_agent_dict[k].state_to_bucket(current_state)
        for step in range(env.get_max_timestep()+1):
            for k in range(NUM_AGENTS):
                # actions[k] = load_agent_dict[k].take_action()
                for j in range(NUM_COPIES):
                    actions[k*NUM_COPIES+j] = load_agent.get_action(current_states[k*NUM_COPIES+j], policy='onpolicy')
            actions[NUM_AGENTS*NUM_COPIES] = randint(0,2)
            # Random Agent
            actions[NUM_AGENTS*NUM_COPIES+2] = load_agent.get_action(current_states[NUM_AGENTS*NUM_COPIES+2], policy = 'manual')
            response = env.step(loadActionDict=actions)
            for k in range(NUM_AGENTS):
                for j in range(NUM_COPIES):
                    current_states[k*NUM_COPIES+j] = {LOAD_BATTERY_STATE: response[1][k*NUM_COPIES+j][0][0],
                              LOAD_PRICE_STATE: response[1][k*NUM_COPIES+j][0][1][-1],
                              # LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                              # LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                              } # loadFeedbackDict, loadID, observation, battery/price
            current_states[NUM_AGENTS*NUM_COPIES + 2] = {LOAD_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][0],
                                              LOAD_PRICE_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][1][-1],
                                              # LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                                              # LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                                              }
                # load_agent_dict[k].update_state(current_state)
                # current_state = load_agent_dict[k].state_to_bucket(current_state)
            for i in range(NUM_AGENTS):
                for j in range(NUM_COPIES):
                    rewards[i*NUM_COPIES+j]+= ((response[1][i*NUM_COPIES+j][1][1]+response[1][i*NUM_COPIES+j][1][0]) * response[1][i*NUM_COPIES+j][1][2]
                                   # sum(env.get_demand_bounds(0)) * response[1][0][1][1] * PENALTY_FACTOR * 0.5 *
                                   # get_battery_reward_factor(actions[i*NUM_COPIES+j],
                                   #                           current_states[i*NUM_COPIES+j][LOAD_BATTERY_STATE],
                                                             # current_states[i][LOAD_MEAN_BATTERY_STATE],
                                                             # current_states[i][LOAD_VARIANCE_BATTERY_STATE],
                                                             # load_agent_dict[0].bucket_bounds[LOAD_VARIANCE_BATTERY_STATE][1]
                                                             # )
                    )/1000
            for i in range(NUM_AGENTS*NUM_COPIES, NUM_AGENTS*NUM_COPIES+3):
                rewards[i]+= ((response[1][i][1][1]+response[1][i][1][0]) * response[1][i][1][2]# +
                               # sum(env.get_demand_bounds(0)) * response[1][0][1][1] * PENALTY_FACTOR * 0.5 *
                               # get_battery_reward_factor(actions[i],
                               #                           current_states[i][LOAD_BATTERY_STATE],
                                                         # current_states[i][LOAD_MEAN_BATTERY_STATE],
                                                         # current_states[i][LOAD_VARIANCE_BATTERY_STATE],
                                                         # load_agent_dict[0].bucket_bounds[LOAD_VARIANCE_BATTERY_STATE][1]
                                                         # )
                    )/1000
        print(day, rewards)
        rewardslist.append(rewards[:])

    return rewardslist


NUM_DAYS = 10
costs = [[],[],[],[]]
for k in range(10):
    env = setup(2**k)
    rewards = calculate_rewards(0,NUM_DAYS)
    costs[0].append(rewards[-1][0])
    costs[1].append((rewards[-1][-3]))
    costs[2].append((rewards[-1][-2]))
    costs[3].append((rewards[-1][-1]))


# rewards = np.array(rewards)
colors = ['y','b','g','r','k','r:']
labels = ['Trained 2 states agent', 'Random Agent', 'Action 1 agent', 'Handcrafted Agent']
# capacity_rewards = np.array([(capacity_function(i),j) for i,j in enumerate(rewards[-1][:-3])])
# capacity_rewards_sorted = np.array(sorted([(capacity_function(i),j) for i,j in enumerate(rewards[-1][:-3])], key = lambda x: x[1]))
for i in range(1):
    plt.plot([2**i for i in range(10)],costs[i], colors[i], label = labels[i])
plt.legend()
# plt.plot(capacity_rewards[:,0], means+vars, label = 'means + 1 std')
# plt.plot(capacity_rewards[:,0], means, label = 'means')
# plt.plot(capacity_rewards[:,0], means-vars, label = 'means - 1 std')
# plt.plot(capacity_rewards[:,0], vars, label = 'Standard Deviation')
# plt.legend()
plt.axes().set(title="Variation ",xlabel = "Number of loads")
# for i in range(NUM_AGENTS):
#     # plt.figure(i)
#     for j in range(0,5):
#         plt.plot(range(len(rewards)),rewards[:,i*NUM_COPIES+j], colors[i], label=str(2.0*j+2))
# plt.legend()
# for i in range(3):
#     plt.plot(range(len(rewards)), rewards[:, NUM_AGENTS* NUM_COPIES + i], colors[i+NUM_AGENTS], labels = 'benchmark'+str(i))
#
