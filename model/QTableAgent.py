import numpy as np
import random
import math

class QTableAgent(object):

    def __init__(self, action_space, state_bounds=None, num_buckets = None, mini_bucket_count = None, initial_explore_rate = 0.9, discount_factor = 0.9, initial_learning_rate= 0.15, min_explore_rate = 0.005, min_learning_rate = 0.1, default_action = 0, moving_buckets = True):
        """

        :param action_space: list of actions
        :param state_bounds: dict of lower and upper bound of each state element, shape (n,2)
        :param num_buckets: dict of number of buckets for each state
        :param min_explore_rate
        :param min_learning_rate

        """

        assert (isinstance(num_buckets, dict))
        assert (isinstance(state_bounds, dict))
        assert (isinstance(action_space, list))
        assert (default_action in action_space)

        if mini_bucket_count is None:
            mini_bucket_count = dict((k,20) for k in num_buckets.keys())


        # assert (len(num_buckets) == len(state_space))

        self.default_action = default_action
        self.action = self.default_action
        self.action_space = action_space
        self.state_bounds = state_bounds
        self.num_actions = len(self.action_space)
        self.num_buckets = num_buckets
        self.state_index_mapping = dict((state_name, index) for index,state_name in enumerate(self.num_buckets.keys()))
        self.min_explore_rate = min_explore_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate = initial_learning_rate
        self.explore_rate = initial_explore_rate
        self.discount_factor = discount_factor
        self.qtable = np.nan
        self.visit_counts = np.nan
        self.policy = np.nan
        self.initialize_qtable()
        self.state = [0]*len(num_buckets.keys())
        self.prev_state = [0]*len(num_buckets.keys())
        self.mini_bucket_count = mini_bucket_count
        self.bucket_bounds = dict((k,np.linspace(self.state_bounds[k][0], self.state_bounds[k][1], self.num_buckets[k]+1)[1:-1]) for k in self.num_buckets.keys())
        self.mini_bucket_visit_counts = dict((k,np.zeros(self.num_buckets[k]*self.mini_bucket_count[k]+1)) for k in self.num_buckets.keys())
        self.num_visits = 0.0
        self.moving_buckets = moving_buckets
        self.exploration_decay_constant = 1000


    def set_state_bounds(self, state_bounds):
        assert (isinstance(state_bounds, dict))
        self.state_bounds = state_bounds

    def set_action_space(self, action_space):
        assert (isinstance(action_space, list))
        self.action_space = action_space

    def get_action_space(self):
        return self.action_space

    def get_state_bounds(self):
        return self.state_bounds

    def set_num_buckets(self, num_buckets):
        assert (isinstance(num_buckets, dict))
        self.num_buckets = num_buckets
        self.prev_state = [0]*len(num_buckets.keys())
        self.state = [0]*len(num_buckets.keys())
        self.initialize_qtable()
        raise Warning("States and Q_tables re-initialized")


    def get_num_buckets(self):
        return self.num_buckets

    def initialize_qtable(self):
        self.qtable = np.zeros((tuple(self.num_buckets[i] for i in sorted(self.num_buckets.keys(), key = lambda x: self.state_index_mapping[x]))+(self.num_actions,)))
        self.visit_counts = np.zeros((tuple(self.num_buckets[i] for i in sorted(self.num_buckets.keys(), key = lambda x: self.state_index_mapping[x]))+(self.num_actions,)))
        self.policy = np.ones(self.qtable.shape[:-1], dtype=np.uint8)


    def update_state(self,state, updatebounds = None):
        if updatebounds is None:
            updatebounds = self.moving_buckets
        assert (isinstance(state,dict))
        self.prev_state = self.state[:]
        for k in self.state_bounds.keys():
            if state[k]<self.state_bounds[k][0]:
                self.state_bounds[k][0] = state[k]
                # self.state[self.state_index_mapping[k]] = 0
            elif state[k]>self.state_bounds[k][1]:
                self.state_bounds[k][1] = state[k]
                # self.state[self.state_index_mapping[k]] = self.num_buckets[k]-1
            # else:
            #     bound_width = self.state_bounds[k][1] - self.state_bounds[k][0]
            #     if bound_width==0:
            #         self.state[self.state_index_mapping[k]] = 0
            #     else:
            #         offset = (self.num_buckets[k] - 1) * self.state_bounds[k][0] / bound_width
            #         scaling = (self.num_buckets[k] - 1) / bound_width
            #         self.state[self.state_index_mapping[k]] = int(round(scaling*state[k] - offset))
        self.num_visits+= 1e-4
        if self.moving_buckets:
            self.update_minibucket(state)
        if updatebounds:
            self.update_bucket_bounds()
        self.state = self.state_to_bucket(state)[:]




    def update_qtable(self, current_state, current_action, reward, next_state, next_action = None, mode = 'vanilla', clip = None):
        if mode is 'vanilla':
            best_q = np.amax(self.qtable[tuple(self.state_to_bucket(next_state))])
        elif mode is 'sarsa':
            best_q = self.qtable[tuple(self.state_to_bucket(next_state))+(next_action,)]
        increment = self.learning_rate * (reward + self.discount_factor * best_q -
                                          self.qtable[tuple(self.state_to_bucket(current_state)) + (current_action,)])

        if not clip is None:
            increment = max(clip[0], min(clip[1], increment))
        # print(reward, increment)
        self.qtable[tuple(self.state_to_bucket(current_state)) + (current_action,)] += increment
        self.visit_counts[tuple(self.state_to_bucket(current_state)) + (current_action,)] += 1e-4
        # if abs(increment)>abs(reward) and abs(reward)>0.0001:
            # print(increment, reward, best_q, self.qtable[tuple(self.state_to_bucket(current_state)) + (current_action,)])
            # print(self.state_to_bucket(current_state),current_action,self.state_to_bucket(next_state))
        return increment

    def sample_action(self):
        return self.action_space[random.randint(0,self.num_actions-1)]

    def take_action(self):

        if random.random()<self.explore_rate:
            self.action = self.sample_action()
        else:
            self.action = np.argmax(self.qtable[tuple(self.state)])
        return self.action

    def get_action(self, current_state, policy = 'onpolicy'):
        if policy == 'confidence' or policy == 'softmax':
            probs = self.qtable[tuple(self.state_to_bucket(current_state))]
            probs/=probs.sum()
            probs = probs*probs
            probs/=probs.sum()
            return np.argmin(np.random.multinomial(100,probs))
        elif policy == 'egreedy':
            if random.random() < self.explore_rate:
                return self.sample_action()
            else:
                return np.argmax(self.qtable[tuple(self.state_to_bucket(current_state))])
        elif policy == 'random':
            return self.sample_action()
        elif policy == 'default':
            return self.default_action
        elif policy == 'manual':
            if current_state['battery']<15:
                return 0
            else:
                return 2
        else:
            return np.argmax(self.qtable[tuple(self.state_to_bucket(current_state))])


    def get_explore_rate(self,t):
        return max(self.min_explore_rate, min(1, 1.0 - math.log10((t + 1) / self.exploration_decay_constant)))

    def get_learning_rate(self,t):
        return max(self.min_learning_rate, min(0.5, 1.0 - math.log10((t + 1) / 5000)))

    def set_explore_rate(self, explore_rate):
        self.explore_rate = explore_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def update_policy(self, remaining_states=None, filled_states = None):
        if remaining_states is None:
            remaining_states = self.get_max_state()
            # print('remainingstates: ',remaining_states)
        if filled_states is None:
            filled_states = []
        # print('filledstates',filled_states)
        if len(remaining_states)>0:
            for i in range(remaining_states[0]):
                self.update_policy(remaining_states[1:], filled_states+[i])
        else:
            self.policy[tuple(filled_states)] = np.argmax(self.qtable[tuple(filled_states)])



        # for i in range(self.policy.shape[0]):
        #     for j in range(self.policy.shape[1]):
        #         self.policy[i,j] = np.argmax(self.qtable[i,j,:])

    def state_to_bucket(self,state):
        buckets = [0]*len(self.num_buckets)

        for k in self.num_buckets.keys():
            # if state[k]<self.state_bounds[k][0]:
            #     buckets[self.state_index_mapping[k]] = 0
            # elif state[k]>self.state_bounds[k][1]:
            #     buckets[self.state_index_mapping[k]] = self.num_buckets[k]-1
            # else:
            #     bound_width = self.state_bounds[k][1] - self.state_bounds[k][0]
            #     if bound_width == 0:
            #         self.state[self.state_index_mapping[k]] = 0
            #     else:
            #         offset = (self.num_buckets[k] - 1) * self.state_bounds[k][0] / bound_width
            #         scaling = (self.num_buckets[k] - 1) / bound_width
            #         buckets[self.state_index_mapping[k]] = int(round(scaling*state[k] - offset))
            if state[k]<self.bucket_bounds[k][0]:
                buckets[self.state_index_mapping[k]] = 0
            elif state[k]>self.bucket_bounds[k][-1]:
                buckets[self.state_index_mapping[k]] = self.num_buckets[k]-1
            else:
                minb, maxb = 0, self.bucket_bounds[k].shape[0]
                # mid = int((minb + maxb) / 2)
                while minb<=maxb:
                    mid = int((minb+maxb)/2)
                    if mid == minb:
                        buckets[self.state_index_mapping[k]] = mid+1
                        break
                    if state[k]<self.bucket_bounds[k][mid]:
                        maxb = mid
                    else:
                        minb=mid

        return buckets

    def get_max_state(self):
        max_state = [0]*len(self.num_buckets.keys())
        for key in self.num_buckets.keys():
            max_state[self.state_index_mapping[key]] = self.num_buckets[key]
        return max_state

    def update_minibucket(self, state):
        for k in self.state_bounds.keys():
            if (self.state_bounds[k][1]-self.state_bounds[k][0]) == 0:
                self.mini_bucket_visit_counts[k][0]+= 1e-4
            else:
                self.mini_bucket_visit_counts[k][int((state[k]-self.state_bounds[k][0])/(self.state_bounds[k][1]-self.state_bounds[k][0])*self.num_buckets[k]*self.mini_bucket_count[k])] += 0.0001

    def update_bucket_bounds(self, moving_buckets = None):
        if moving_buckets is None:
            moving_buckets = self.moving_buckets
        if not moving_buckets:
            self.bucket_bounds = dict((k, np.linspace(self.state_bounds[k][0], self.state_bounds[k][1], self.num_buckets[k] + 1)[1:-1]) for k in self.num_buckets.keys())
        else:
            for k in self.bucket_bounds.keys():
                sum = 0
                bound_count = 0
                rem_sum = self.num_visits
                # print(self.mini_bucket_visit_counts[k].sum(), self.num_visits)
                # assert(abs(self.mini_bucket_visit_counts[k].sum() - self.num_visits) < 1e-6)
                for i in range(self.mini_bucket_visit_counts[k].shape[0]):
                    sum+=self.mini_bucket_visit_counts[k][i]
                    if sum>rem_sum/(self.num_buckets[k]-bound_count):
                        self.bucket_bounds[k][bound_count] = (i+1)*(self.state_bounds[k][1]-self.state_bounds[k][0])/(len(self.mini_bucket_visit_counts[k])+1)+self.state_bounds[k][0]
                        rem_sum -= sum
                        sum = 0
                        bound_count+=1
                        # if k == 'vb':
                        #     print(rem_sum, bound_count)
                    if rem_sum<= 1e-4:
                        for j in range(bound_count, self.num_buckets[k]-1):
                            self.bucket_bounds[k][j] = self.bucket_bounds[k][j-1]
                        break

                    if bound_count>=self.num_buckets[k]-1:
                        break

                # if k == 'vb':
                #     print(self.bucket_bounds[k])
