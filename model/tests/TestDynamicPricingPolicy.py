from Environment import Environment
import os
import numpy as np
from QTableAgent import QTableAgent
import pickle
from random import randint
from matplotlib import pyplot as plt


SOURCE_DEMAND_STATE = 'demand'
SOURCE_SMART_LOADS = 'smart_load'
SOURCE_LEARNING_RATE = 'slr'
SOURCE_DISCOUNT_FACTOR = 'sdf'
SOURCE_NUM_LOADS = 'snl'
SOURCE_DAY = 'day'
SOURCE_MODE = 'vanilla'

LOAD_RANDOMIZE_BATTERY = 'rb'#True
LOAD_MODE = 'mode'
LOAD_DAY = 'day'#199999
LOAD_NUM_LOADS = 'ndl'
LOAD_LEARNING_RATE = 'lr'
LOAD_DISCOUNT_FACTOR = 'df'
LOAD_BATTERY_STATE = 'battery'
LOAD_PRICE_STATE = 'price'


load_agent_params =         {
                                LOAD_RANDOMIZE_BATTERY:True,
                                LOAD_LEARNING_RATE: 0.03,
                                LOAD_DISCOUNT_FACTOR: 0.95,
                                LOAD_NUM_LOADS:9,
                                LOAD_DAY:99999,
                                LOAD_MODE:'vanilla'
                            }
LOAD_MODEL_PATH = os.getcwd()
LOAD_MODEL_PATH += '/basic_qlearning_models'
if load_agent_params[LOAD_RANDOMIZE_BATTERY]:
    LOAD_MODEL_PATH+='/randomize_battery'
else:
    LOAD_MODEL_PATH+='/continuous_battery'
LOAD_MODEL_PATH+= '/dumloads'+str(load_agent_params[LOAD_NUM_LOADS]) +'/df' + str(load_agent_params[LOAD_DISCOUNT_FACTOR]) + '/lr'+str(load_agent_params[LOAD_LEARNING_RATE])


source_agent_params = [
    {
        SOURCE_SMART_LOADS : True,
        SOURCE_LEARNING_RATE :0.03,
        SOURCE_DISCOUNT_FACTOR : 0.95,
        SOURCE_NUM_LOADS : 10,
        SOURCE_MODE : 'vanilla',
        SOURCE_DAY: 9999
    },
]

NUM_AGENTS = len(source_agent_params)
source_agent_dict = {}

source_load_map = {}
for i in range(NUM_AGENTS+2):
    source_load_map[i] = [10*i,10*(i+1)]


for i in range(NUM_AGENTS):
    MODEL_PATH = os.getcwd()
    MODEL_PATH += '/dynamic_pricing_models'
    if source_agent_params[i][SOURCE_SMART_LOADS]:
        MODEL_PATH+='/smart'
    else:
        MODEL_PATH+='/dumb'
    MODEL_PATH+= '/df' + str(source_agent_params[i][SOURCE_DISCOUNT_FACTOR]) + \
                 '/lr'+str(source_agent_params[i][SOURCE_LEARNING_RATE])

    # policy = np.load(MODEL_PATH + '/policy_'+str(DAY) + '.npy')
    # with open(MODEL_PATH+'/agent_'+str(load_agent_params[i][DAY])+'.pickle', 'rb') as f:
    with open(MODEL_PATH + '/' + source_agent_params[i][SOURCE_MODE] + '_agent_' + str(source_agent_params[i][SOURCE_DAY]) + '.pickle', 'rb') as f:
        source_agent_dict[i] = pickle.load(f)

load_agent = None
if SOURCE_SMART_LOADS:
    with open(
            LOAD_MODEL_PATH + '/' + load_agent_params[LOAD_MODE] + '_agent_' + str(
                load_agent_params[LOAD_DAY]) + '.pickle',
            'rb') as f:
        load_agent = pickle.load(f)


def setup():
    env = Environment()

    connect_graph = {}
    for i in range(NUM_AGENTS+2):
        connect_graph[i] = list(range(source_load_map[i][0], source_load_map[i][1]))
    env.add_connections(connect_graph)
    env.set_environment_ready()
    env.reset(0)
    # load_agent_dict = {0:QTableAgent(env.get_load_action_space(),
    #                                  {LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0)},
    #                                  {LOAD_BATTERY_STATE:20, LOAD_PRICE_STATE:10},
    #                                  default_action=1,
    #                                  discount_factor=DISCOUNT_FACTOR
    #                                 )}
    # load_agent_dict[0].set_learning_rate(LEARNING_RATE)
    for i in source_agent_dict.keys():
        source_agent_dict[i].set_explore_rate(0)
    return env

def calculate_rewards(startday = 0, endday = 100):
    rewards = [0.0]*(NUM_AGENTS+2)
    rewardslist = []
    actions = dict((k,v) for k,v in enumerate([0]*(NUM_AGENTS+2)))
    actions[NUM_AGENTS] = 4
    load_actions = {}
    for day in range(startday, endday):
        response = env.reset()
        for k in range(NUM_AGENTS):
            current_state = {SOURCE_DEMAND_STATE: response[0][k][0][0]} #loadFeedbackDict, loadID, observation, battery/price
            source_agent_dict[k].update_state(current_state)
        for step in range(env.get_max_timestep()+1):
            for k in range(NUM_AGENTS):
                actions[k] = source_agent_dict[k].take_action()
                if SOURCE_SMART_LOADS:
                    for i in range(source_load_map[k][0], source_load_map[k][1]):
                        load_actions[i] = load_agent.get_action(
                            {LOAD_BATTERY_STATE: response[1][i][0][0], LOAD_PRICE_STATE: response[1][i][0][1][-1]})
            actions[NUM_AGENTS+1] = randint(0,4)

            response = env.step(sourceActionDict=actions, loadActionDict=load_actions)
            # print(response[0])
            for k in range(NUM_AGENTS):
                current_state = {SOURCE_DEMAND_STATE: response[0][k][0][0]} # loadFeedbackDict, loadID, observation, battery/price
                source_agent_dict[k].update_state(current_state)
            for i in range(NUM_AGENTS+2):
                rewards[i] += response[0][i][1]/1000
        print(day, rewards)
        rewardslist.append(rewards[:])

    return rewardslist



env = setup()
rewards = calculate_rewards()

