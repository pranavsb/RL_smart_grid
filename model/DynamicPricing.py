from Environment import Environment
from QTableAgent import QTableAgent
import time, os
import numpy as np
import pickle
from Utils import get_battery_reward_factor

SOURCE_DEMAND_STATE = 'demand'
SOURCE_SMART_LOADS = True
SOURCE_LEARNING_RATE = 0.03
SOURCE_DISCOUNT_FACTOR = 0.95
SOURCE_NUM_LOADS = 10
SOURCE_MODE = 'vanilla'

LOAD_RANDOMIZE_BATTERY = 'rb'#True
LOAD_MODE = 'mode'
LOAD_DAY = 'day'#199999
LOAD_NUM_LOADS = 'ndl'
LOAD_LEARNING_RATE = 'lr'
LOAD_DISCOUNT_FACTOR = 'df'
LOAD_BATTERY_STATE = 'battery'
LOAD_PRICE_STATE = 'price'





MODEL_PATH = os.getcwd()#+'/basic_qlearning_models/dumloads'+str(NUM_DUM_LOADS)+'/df'+str(DISCOUNT_FACTOR)+'/lr'+str(LEARNING_RATE)
MODEL_PATH+='/dynamic_pricing_models'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if SOURCE_SMART_LOADS:
    MODEL_PATH += '/smart'
else:
    MODEL_PATH+='/dumb'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/df'+str(SOURCE_DISCOUNT_FACTOR)
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/lr'+str(SOURCE_LEARNING_RATE)
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

load_agent_params =         {
                                LOAD_RANDOMIZE_BATTERY:True,
                                LOAD_LEARNING_RATE: 0.03,
                                LOAD_DISCOUNT_FACTOR: 0.9,
                                LOAD_NUM_LOADS:999,
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



def setup():
    env = Environment()
    # env.add_connections({0:[0]})
    load_agent = None
    if SOURCE_SMART_LOADS:
        with open(
                LOAD_MODEL_PATH + '/' + load_agent_params[LOAD_MODE] + '_agent_' + str(load_agent_params[LOAD_DAY]) + '.pickle',
                'rb') as f:
            load_agent = pickle.load(f)
        env.add_connections({0:list(range(SOURCE_NUM_LOADS))})
    else:
        env.add_dumb_loads(0,SOURCE_NUM_LOADS)
    env.set_environment_ready()
    env.reset(True)
    source_agent_dict = {0:QTableAgent(env.get_source_action_space(),
                                     {SOURCE_DEMAND_STATE:env.get_overall_demand_bounds(0)},
                                     {SOURCE_DEMAND_STATE:20},
                                     default_action=1,
                                     discount_factor=SOURCE_DISCOUNT_FACTOR
                                    )}
    source_agent_dict[0].set_learning_rate(SOURCE_LEARNING_RATE)
    return env, source_agent_dict, load_agent

def train(startday=0, endday=200000):
    start=time.time()
    load_actions = {}
    for day in range(startday, endday):
        states = []
        actions = []
        max_change = 0
        max_change_state_action = []
        response = env.reset(True)
        next_state = {SOURCE_DEMAND_STATE: response[0][0][0][0]}
        source_agent_dict[0].update_state(next_state)
        next_action = source_agent_dict[0].take_action()

        for step in range(env.get_max_timestep()+1):
            # print(env.get_current_timestep(),step)
            current_state = next_state
            current_action = next_action
            actions.append(current_action)
            if SOURCE_SMART_LOADS:
                for i in range(SOURCE_NUM_LOADS):
                    load_actions[i] = load_agent.get_action(
                        {LOAD_BATTERY_STATE: response[1][i][0][0], LOAD_PRICE_STATE: response[1][i][0][1][-1]})

            response = env.step(sourceActionDict={0:current_action}, loadActionDict=load_actions)
            next_state = {SOURCE_DEMAND_STATE:response[0][0][0][0]}
            states.append(current_state)
            source_agent_dict[0].update_state(next_state)

            if SOURCE_MODE is 'vanilla':
                max_change = max(abs(
                    source_agent_dict[0].update_qtable(
                        current_state=current_state, current_action=current_action,
                        reward = response[0][0][1],
                        mode=SOURCE_MODE, next_state = next_state
                    )), max_change) #response should be negative
                next_action = source_agent_dict[0].take_action()

            elif SOURCE_MODE is 'sarsa':
                next_action = source_agent_dict[0].take_action()
                max_change = max(abs(
                    source_agent_dict[0].update_qtable(
                        current_state=current_state, current_action=current_action,
                        reward = response[0][0][1],
                        next_state=next_state, next_action=next_action, mode=SOURCE_MODE, #clip=[-25,25]  # clip the increments to a certain range
                    )), max_change)  # response should be negative


            max_change_state_action = [source_agent_dict[0].state,current_action]
        print(day,':',source_agent_dict[0].get_explore_rate(day),':',max_change,':',max_change_state_action,':',np.mean(source_agent_dict[0].qtable))
        if max_change<0.001:
            break
        source_agent_dict[0].set_explore_rate(source_agent_dict[0].get_explore_rate(day))
        # load_agent_dict[0].set_learning_rate(load_agent_dict[0].get_learning_rate(day))
        if (day+1)%500==0:
            source_agent_dict[0].update_policy()
            # np.save(MODEL_PATH+'/qtable_'+str(day),load_agent_dict[0].qtable)
            # np.save(MODEL_PATH+'/visitcounts_'+str(day),load_agent_dict[0].visit_counts)
            # np.save(MODEL_PATH+'/policy_'+str(day),load_agent_dict[0].policy)
            with open(MODEL_PATH+'/'+SOURCE_MODE+'_agent_'+str(day)+'.pickle', 'wb') as f:
                pickle.dump(source_agent_dict[0], f)

    end = time.time()
    return end-start

env, source_agent_dict, load_agent = setup()

timetaken = train(0,10000)
