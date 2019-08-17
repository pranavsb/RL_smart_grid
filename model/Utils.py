import math
import os
from constants import *
import pickle

def normalize_timestep(timestep, timestep_size, default_timestep_size):
    return int((timestep * timestep_size) / default_timestep_size)

def get_last_k(array, k=1, default = 0.0):
    myarr = [default]*k
    if len(array) == 0:
        return myarr
    if len(array)<k:
        myarr[-len(array):] = array[:]
    else:
        myarr = array[-k:]

    return myarr

def get_battery_reward_factor(action, current_battery_percentage, mean=0, variance=0, variance_max=0,
                              battery_factor_upper = 0.5, mean_factor_upper = 0, variance_factor_upper = 0):

    if action == 0:
        battery_factor =  battery_factor_upper/(1+math.exp((-current_battery_percentage+40)/10))
    elif action==2:
        battery_factor =  battery_factor_upper/(1+math.exp((current_battery_percentage-40)/10))
    else:
        battery_factor = battery_factor_upper/2


    mean_factor = mean_factor_upper * ((1-mean/50)**2)
    if variance_max>0:
        variance_factor = variance_factor_upper * min(1,(variance/variance_max))
    else:
        variance_factor = 0

    # print(1+battery_factor+mean_factor+variance_factor)
    return 1+battery_factor+mean_factor+variance_factor

def create_model_path(model_params_dict):
    model_params_dict[STATES] = ''.join(sorted(model_params_dict[STATES])).upper()
    MODEL_PATH = os.getcwd()  # +'/basic_qlearning_models/dumloads'+str(NUM_DUM_LOADS)+'/df'+str(DISCOUNT_FACTOR)+'/lr'+str(LEARNING_RATE)
    MODEL_PATH += '/basic_qlearning_models'
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_PATH += '/' + str(model_params_dict[STATES])
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if model_params_dict[MOVING_BUCKETS]:
        MODEL_PATH += '/moving_buckets'
    else:
        MODEL_PATH += '/static_buckets'
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if model_params_dict[RANDOMIZE_BATTERY]:
        MODEL_PATH += '/randomize_battery'
    else:
        MODEL_PATH += '/continuous_battery'
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_PATH += '/dumloads' + str(model_params_dict[NUM_DUM_LOADS])
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_PATH += '/df' + str(model_params_dict[DISCOUNT_FACTOR])
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_PATH += '/lr' + str(model_params_dict[LEARNING_RATE])
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    return MODEL_PATH

def read_model(model_params_dict):
    states = ''.join(sorted(model_params_dict[STATES])).upper()
    MODEL_PATH = os.getcwd()
    if SHARPE in model_params_dict.keys():
        if not model_params_dict[SHARPE]:
            MODEL_PATH += '/basic_qlearning_models'
        else:
            MODEL_PATH += '/sharpe_ratio_models'
    else:
        MODEL_PATH += '/basic_qlearning_models'
    # if model_params_dict[STATES] > 0:
    MODEL_PATH += '/' + str(states)
    if model_params_dict[MOVING_BUCKETS]:
        MODEL_PATH += '/moving_buckets'
    else:
        MODEL_PATH += '/static_buckets'
    if model_params_dict[RANDOMIZE_BATTERY]:
        MODEL_PATH += '/randomize_battery'
    else:
        MODEL_PATH += '/continuous_battery'
    MODEL_PATH += '/dumloads' + str(model_params_dict[NUM_DUM_LOADS]) + \
                  '/df' + str(model_params_dict[DISCOUNT_FACTOR]) + \
                  '/lr' + str(model_params_dict[LEARNING_RATE])  # + \
    # '/pf_'+str(load_agent_params[i][PF])

    # policy = np.load(MODEL_PATH + '/policy_'+str(DAY) + '.npy')
    # with open(MODEL_PATH+'/agent_'+str(load_agent_params[i][DAY])+'.pickle', 'rb') as f:
    with open(MODEL_PATH + '/' + model_params_dict[MODE] + '_agent_' + str(model_params_dict[DAY]) + '.pickle',
              'rb') as f:
        return pickle.load(f)