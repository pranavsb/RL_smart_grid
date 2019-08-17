from Environment import Environment
from QTableAgent import QTableAgent
import time, os
import numpy as np
import pickle
from Utils import get_battery_reward_factor

LOAD_BATTERY_STATE = 'battery'
LOAD_PRICE_STATE = 'price'
LOAD_MEAN_BATTERY_STATE = 'mb'
LOAD_VARIANCE_BATTERY_STATE = 'vb'

RANDOMIZE_BATTERY = True
LEARNING_RATE = 0.03
DISCOUNT_FACTOR = 0.95
NUM_DUM_LOADS = 999
mode = 'vanilla'
STATES = 2
MOVING_BUCKETS = True
HISTORY_FACTOR = .999
#CONSTANT_DEMAND = False

#PENALTY_FACTOR = 0.5

MODEL_PATH = os.getcwd()#+'/basic_qlearning_models/dumloads'+str(NUM_DUM_LOADS)+'/df'+str(DISCOUNT_FACTOR)+'/lr'+str(LEARNING_RATE)
#MODEL_PATH+='/basic_qlearning_models'
MODEL_PATH+='/sharpe_ratio_models'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/'+str(STATES)+'states'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if MOVING_BUCKETS:
    MODEL_PATH+='/moving_buckets'
else:
    MODEL_PATH+='/static_buckets'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# if CONSTANT_DEMAND:
#     MODEL_PATH+='/constant_demand'
# else:
#     MODEL_PATH+='/csv_demand'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if RANDOMIZE_BATTERY:
    MODEL_PATH+='/randomize_battery'
else:
    MODEL_PATH+='/continuous_battery'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/dumloads'+str(NUM_DUM_LOADS)
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/df'+str(DISCOUNT_FACTOR)
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_PATH+='/lr'+str(LEARNING_RATE)
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# MODEL_PATH+='/pf_' + str(PENALTY_FACTOR)
# if not os.path.isdir(MODEL_PATH):
#     os.makedirs(MODEL_PATH)


def setup():
    env = Environment()
    env.add_connections({0:[0]})
    env.add_dumb_loads(0,NUM_DUM_LOADS)
    env.set_environment_ready()
    # env.reset(RANDOMIZE_BATTERY)
    load_agent_dict = {0:QTableAgent(env.get_load_action_space(),
                                     {LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0)},
                                      # LOAD_MEAN_BATTERY_STATE:env.get_battery_bounds(0)[0],
                                      # LOAD_VARIANCE_BATTERY_STATE:env.get_battery_bounds(0)[1]},
                                     {LOAD_BATTERY_STATE:20, LOAD_PRICE_STATE:10},
                                      # LOAD_MEAN_BATTERY_STATE:10, LOAD_VARIANCE_BATTERY_STATE:10},
                                     default_action=1,
                                     discount_factor=DISCOUNT_FACTOR,
                                     moving_buckets= MOVING_BUCKETS
                                    )}
    load_agent_dict[0].set_learning_rate(LEARNING_RATE)
    return env, load_agent_dict


def train(startday=0, endday=200000):
    start=time.time()
    mean = None
    variance = 1
    #demands = np.zeros(endday-startday)
    for day in range(startday, endday):
        states = []
        actions = []
        max_change = 0
        max_change_state_action = []
        response = env.reset(RANDOMIZE_BATTERY)
        next_state = {LOAD_BATTERY_STATE: response[1][0][0][0],
                      LOAD_PRICE_STATE: response[1][0][0][1][-1],
                      # LOAD_MEAN_BATTERY_STATE: response[1][0][0][2],
                      # LOAD_VARIANCE_BATTERY_STATE: response[1][0][0][3],
                      }
        load_agent_dict[0].update_state(next_state)
        next_action = load_agent_dict[0].take_action()

        for step in range(env.get_max_timestep()+1):
            # print(env.get_current_timestep(),step)
            current_state = next_state
            current_action = next_action
            actions.append(current_action)
            response = env.step(loadActionDict={0:current_action})
            next_state = {LOAD_BATTERY_STATE: response[1][0][0][0],
                          LOAD_PRICE_STATE: response[1][0][0][1][-1],
                          # LOAD_MEAN_BATTERY_STATE: response[1][0][0][2],
                          # LOAD_VARIANCE_BATTERY_STATE: response[1][0][0][3],
                          }
            states.append(current_state)
            if step%20==0:
                load_agent_dict[0].update_state(next_state, True)
            else:
                load_agent_dict[0].update_state(next_state, False)


            if mode is 'vanilla':
                if mean is None:
                    mean = response[1][0][1][1]*response[1][0][1][0]
                #demands[day] = response[1][0][1][1]
                sharp_reward = mean / ((variance ** 0.5)) #* demands.mean())
                #sharp_reward = mean / ((variance ** 0.5) * response[1][0][1][1])
                reward = (-1) * sharp_reward  # * response[1][0][1][1]
                mean += (1 - HISTORY_FACTOR) * (reward - mean)
                variance += (1 - HISTORY_FACTOR) * (
                        (reward - mean) ** 2 - variance)
                print(reward, mean, variance)
                max_change = max(abs(
                    load_agent_dict[0].update_qtable(
                        current_state=current_state, current_action=current_action,
                        reward=reward,
                        mode=mode, next_state = next_state, #clip=[-100,100]
                    )), max_change) #response should be negative
                next_action = load_agent_dict[0].take_action()

            elif mode is 'sarsa':
                next_action = load_agent_dict[0].take_action()
                max_change = max(abs(
                    load_agent_dict[0].update_qtable(
                        current_state=current_state, current_action=current_action,
                        reward = 0,
                        next_state=next_state, next_action=next_action, mode=mode, #clip=[-25,25]  # clip the increments to a certain range
                    )), max_change)  # response should be negative


            max_change_state_action = [load_agent_dict[0].state,current_action]
        print(day,':',load_agent_dict[0].get_explore_rate(day),':',max_change,':',max_change_state_action,':',np.mean(load_agent_dict[0].qtable))
        # if max_change<0.001:
        #     break
        load_agent_dict[0].set_explore_rate(load_agent_dict[0].get_explore_rate(day))
        # load_agent_dict[0].set_learning_rate(load_agent_dict[0].get_learning_rate(day))
        if (day+1)%500==0:
            load_agent_dict[0].update_policy()
            # np.save(MODEL_PATH+'/qtable_'+str(day),load_agent_dict[0].qtable)
            # np.save(MODEL_PATH+'/visitcounts_'+str(day),load_agent_dict[0].visit_counts)
            # np.save(MODEL_PATH+'/policy_'+str(day),load_agent_dict[0].policy)
            with open(MODEL_PATH+'/'+mode+'_agent_'+str(day)+'.pickle', 'wb') as f:
                pickle.dump(load_agent_dict[0], f)

        if day-1%20==0:
            load_agent_dict[0].set_state_bounds({LOAD_BATTERY_STATE:[0,100],LOAD_PRICE_STATE:env.get_price_bounds(0),})
                                      # LOAD_MEAN_BATTERY_STATE:env.get_battery_bounds(0)[0],
                                      # LOAD_VARIANCE_BATTERY_STATE:env.get_battery_bounds(0)[1]})
            # print(env.get_battery_bounds(0))


    end = time.time()
    return end-start

env, load_agent_dict = setup()

timetaken = train(0,10000)