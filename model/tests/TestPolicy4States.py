from Environment import Environment
import os
import numpy as np
from QTableAgent import QTableAgent
import pickle
from random import randint
from matplotlib import pyplot as plt
from Utils import read_model
from constants import *



NUM_COPIES = 1

PENALTY_FACTOR = 0

load_agent_params = [#{
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.9,
                    #     NUM_DUM_LOADS:999,
                    #     DAY:9999,
                    #     MODE:'sarsa',
                    #     STATES:2
                    #  },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.95,
                    #     NUM_DUM_LOADS:999,
                    #     DAY:99999,
                    #     MODE:'sarsa',
                    #     STATES:2
                    #  },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.95,
                    #     NUM_DUM_LOADS:9,
                    #     DAY:99999,
                    #     MODE:'sarsa'
                    #  },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.9,
                    #     NUM_DUM_LOADS:999,
                    #     DAY:99999,
                    #     MODE:'vanilla',
                    #     STATES:2
                    #  },
                    # {
                    #     RANDOMIZE_BATTERY:True,
                    #     LEARNING_RATE: 0.03,
                    #     DISCOUNT_FACTOR: 0.95,
                    #     NUM_DUM_LOADS:999,
                    #     DAY:99999,
                    #     MODE:'vanilla',
#                     #     STATES:2
#                     #  },
{
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:39999,
                        MODE:'vanilla',
                        STATES:['b20','p10'],
                        MOVING_BUCKETS: False,
                        PF:0.5,
                        SHARPE:False
                     },
{
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:39999,
                        MODE:'vanilla',
                        STATES:['b20','p10'],
                        MOVING_BUCKETS: True,
                        PF:0.5,
                        SHARPE:False
                     },
{
                        RANDOMIZE_BATTERY:True,
                        LEARNING_RATE: 0.1,
                        DISCOUNT_FACTOR: 0.95,
                        NUM_DUM_LOADS:999,
                        DAY:39999,
                        MODE:'vanilla',
                        STATES:['b20','p10', 'd15'],
                        MOVING_BUCKETS: True,
                        PF:0.5,
                        SHARPE: False
                     },
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
    load_agent_dict[i] = read_model(load_agent_params[i])

def setup():
    env = Environment()
    env.add_connections({0:range((NUM_AGENTS*NUM_COPIES+3))})
    env.add_dumb_loads(0,100000)
    env.set_environment_ready()
    env.reset()
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

def calculate_rewards(startday = 0, endday = 500):
    rewards = [0.0]*(NUM_AGENTS*NUM_COPIES+3)
    rewardslist = []
    actions = dict((k,v) for k,v in enumerate([0]*(NUM_AGENTS*NUM_COPIES+3)))
    current_states = dict((i,{}) for i in range(NUM_AGENTS*NUM_COPIES+3))
    actions[NUM_AGENTS*NUM_COPIES+1] = 1                                     # Constant Agent
    for day in range(startday, endday):
        response = env.reset()
        # print(response)
        for k in range(NUM_AGENTS):
            for j in range(NUM_COPIES):
                # print(current_states)
                current_states[k*NUM_COPIES+j] = {LOAD_BATTERY_STATE: response[1][k*NUM_COPIES+j][0][0],
                              LOAD_PRICE_STATE: response[1][k*NUM_COPIES+j][0][1][-1],
                              LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                              LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                              LOAD_DEMAND_STATE: response[1][k][1][0]
                              } #loadFeedbackDict, loadID, observation, battery/price
        current_states[NUM_AGENTS*NUM_COPIES+2] = {LOAD_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][0],
                          LOAD_PRICE_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][1][-1],
                          LOAD_MEAN_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][2],
                          LOAD_VARIANCE_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][0][3],
                          LOAD_DEMAND_STATE: response[1][NUM_AGENTS*NUM_COPIES+2][1][0]
                          }
            # load_agent_dict[k].update_state(current_state)
            # current_state = load_agent_dict[k].state_to_bucket(current_state)
        for step in range(env.get_max_timestep()+1):
            for k in range(NUM_AGENTS):
                # actions[k] = load_agent_dict[k].take_action()
                for j in range(NUM_COPIES):
                    actions[k*NUM_COPIES+j] = load_agent_dict[k].get_action(current_states[k*NUM_COPIES+j], policy = 'onpolicy')
            actions[NUM_AGENTS*NUM_COPIES] = randint(0,2)
            # Random Agent
            actions[NUM_AGENTS*NUM_COPIES+2] = load_agent_dict[0].get_action(current_states[NUM_AGENTS*NUM_COPIES+2], policy = 'manual')
            response = env.step(loadActionDict=actions)
            # print(actions)
            for k in range(NUM_AGENTS):
                for j in range(NUM_COPIES):
                    # print(current_states)
                    current_states[k*NUM_COPIES+j] = {LOAD_BATTERY_STATE: response[1][k*NUM_COPIES+j][0][0],
                              LOAD_PRICE_STATE: response[1][k*NUM_COPIES+j][0][1][-1],
                              LOAD_MEAN_BATTERY_STATE: response[1][k][0][2],
                              LOAD_VARIANCE_BATTERY_STATE: response[1][k][0][3],
                              LOAD_DEMAND_STATE: response[1][k][1][0]
                              } # loadFeedbackDict, loadID, observation, battery/price
            current_states[NUM_AGENTS*NUM_COPIES + 2] = {LOAD_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][0],
                                              LOAD_PRICE_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][1][-1],
                                              LOAD_MEAN_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][2],
                                              LOAD_VARIANCE_BATTERY_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][0][3],
                                              LOAD_DEMAND_STATE: response[1][NUM_AGENTS*NUM_COPIES + 2][1][0]
                                              }
                # load_agent_dict[k].update_state(current_state)
                # current_state = load_agent_dict[k].state_to_bucket(current_state)
            for i in range(NUM_AGENTS):
                for j in range(NUM_COPIES):
                    rewards[i*NUM_COPIES+j]+= ((response[1][i*NUM_COPIES+j][1][1]
                                                + response[1][i*NUM_COPIES+j][1][0]
                                                ) * response[1][i*NUM_COPIES+j][1][2] #+
                                   # sum(env.get_demand_bounds(0)) * response[1][0][1][1] * PENALTY_FACTOR * 0.5 *
                                   # get_battery_reward_factor(actions[i*NUM_COPIES+j],
                                   #                           current_states[i*NUM_COPIES+j][LOAD_BATTERY_STATE],
                                                             # current_states[i][LOAD_MEAN_BATTERY_STATE],
                                                             # current_states[i][LOAD_VARIANCE_BATTERY_STATE],
                                                             # load_agent_dict[0].bucket_bounds[LOAD_VARIANCE_BATTERY_STATE][1]
                                                             # )
                    )/1000
            for i in range(NUM_AGENTS*NUM_COPIES, NUM_AGENTS*NUM_COPIES+3):
                rewards[i]+= ((response[1][i][1][1]
                               + response[1][i][1][0]
                               )* response[1][i][1][2]# +
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



env = setup()
rewards = calculate_rewards(0,200)

rewards = np.array(rewards)
colors = ['y','b','g','r','k','r:','b:']
plt.axes().set(title="Cost Incurred vs Time",xlabel = "Day", ylabel = "Total Costs Incurred ")
labels = ['B20P10', 'B20P10 With Moving Buckets','B20P10D15 With Moving Buckets', 'Random Agent', 'Action 1 Agent', 'Handcrafted Agent']
offset=25
for i in range(NUM_AGENTS):
    # plt.figure(i)
    for j in range(NUM_COPIES):
        plt.plot(range(len(rewards)-offset),rewards[offset:,i*NUM_COPIES+j]-rewards[offset,i*NUM_COPIES+j],colors[i], label = labels[i])
for i in range(3):
    plt.plot(range(len(rewards)-offset), rewards[offset:, NUM_AGENTS* NUM_COPIES + i]-rewards[offset, NUM_AGENTS* NUM_COPIES + i], colors[i+NUM_AGENTS], label = labels[NUM_AGENTS+i])
plt.legend(loc='best')
print('Final percentage gain for 2,3,4 state models: ',100-100*(rewards[-1][:NUM_AGENTS]-rewards[offset][:NUM_AGENTS])/(rewards[-1][-2]-rewards[offset][-2]))
