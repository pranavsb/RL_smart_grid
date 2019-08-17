from Load import Load
from Source import Source
from Utils import get_last_k
from DemandRange import DemandRange

class Environment():

    def __init__(self, sourceDict = None, loadDict = None, envReady = False, timestep_size = 5, look_ahead = 5):

        if sourceDict is None:
            sourceDict = {}

        if loadDict is None:
            loadDict = {}

        self.load_dict = loadDict
        self.source_dict = sourceDict
        self.env_ready = envReady
        self.timestep = 0
        self.timestep_size = timestep_size          #minutes
        self.max_timestep = int(24*60/timestep_size)-1
        self.look_ahead = look_ahead
        self.test_mode = False

        self.load_action_dict = {}
        for loadID in self.load_action_dict.keys():
            self.load_action_dict[loadID] = Load.no_agent_action

        self.source_action_dict = {}
        for sourceID in self.source_dict.keys():
            self.source_action_dict[sourceID] = Source.no_agent_action

        self.load_feedback_dict = {}
        for loadID in self.load_dict.keys():
            self.load_feedback_dict[loadID] = [[],[0.0,0.0,0.0]]      #observation, reward

        self.source_feedback_dict = {}
        for sourceID in self.source_action_dict.keys():
            self.source_feedback_dict[sourceID] = [[], 0.0]




    def _add_load(self, load):
        self.load_action_dict[load.loadID] = Load.no_agent_action
        self.load_feedback_dict[load.loadID] =  [[],[0.0,0.0,0.0]]
        self.load_dict[load.loadID] = load


    def add_load(self, loadParams=None):
        if loadParams is None:
            loadParams = {}
        load = Load(**loadParams)
        self._add_load(load)


    def add_loads(self, n = None, loadParamsList = None):
        if n is None:
            for loadParams in loadParamsList:
                self.add_load(loadParams)
        else:
            for i in range(n):
                try:
                    self.add_load(loadParamsList[i])
                except:
                    self.add_load()


    def add_source(self, sourceParams=None):
        if sourceParams is None:
            sourceParams = {}
        source = Source(**sourceParams)
        self.source_feedback_dict[source.sourceID] =  [[],0.0]
        self.source_action_dict[source.sourceID] = Source.no_agent_action
        self.source_dict[source.sourceID] = source

    def remove_load(self, loadID, sourceID = None):

        if sourceID is None:
            pass
        elif isinstance(sourceID,list):
            for id in sourceID:
                self.source_dict[id].remove_load(loadID)
        else:
            self.source_dict[sourceID].remove_load(loadID)
        self.load_dict.pop(loadID)
        self.load_action_dict.pop(loadID)
        self.load_feedback_dict.pop(loadID)
        if sourceID is None:
            print('You have not provided the sourceID. May cause errors if sources were connected')

    def remove_source(self, sourceID):
        self.source_dict.pop(sourceID)
        self.source_feedback_dict.pop(sourceID)
        self.source_action_dict.pop(self)
        print('All connections with loads will be removed')


    def add_load_to_source(self, loadParams = None, sourceID = None):
        if loadParams is None:
            loadParams = {}
        load = Load(**loadParams)

        if sourceID is None:
            sourceID = Source.num_sources - 1

        if load.loadID not in self.load_dict.keys():
            self._add_load(load)

        self.source_dict[sourceID].add_load(load.loadID)

    def set_environment_ready(self, envReady = True, test_mode = False):
        self.env_ready = envReady
        self.test_mode = test_mode
        self.reset(True)
        for loadID in self.load_dict.keys():
            self.load_dict[loadID].get_battery().mean_battery_bounds.reset_future_bounds()
            self.load_dict[loadID].get_battery().variance_battery_bounds.reset_future_bounds()
            self.load_dict[loadID].demand_bounds.reset_future_bounds()
            # print('batterymean',self.load_dict[loadID].get_battery().mean_battery_bounds.get_bounds())
            # print('batterystd',self.load_dict[loadID].get_battery().variance_battery_bounds.get_bounds())
            # print('loaddemand',self.load_dict[loadID].demand_bounds.get_bounds())
        for sourceID in self.source_dict.keys():
            self.source_dict[sourceID].price_bounds.reset_future_bounds()
            self.source_dict[sourceID].demand_bounds.reset_future_bounds()
            # print('price',self.source_dict[sourceID].price_bounds.get_bounds())
            # print('sourcedemand',self.source_dict[sourceID].demand_bounds.get_bounds())


    def is_environment_ready(self):
        return self.env_ready


    def step(self, sourceActionDict = None, loadActionDict = None):
        if sourceActionDict is None:
            sourceActionDict = {}

        if loadActionDict is None:
            loadActionDict = {}

        if not self.env_ready:
            print('Environment not ready for simulation yet')
            return -1
        self._update_action_dicts(sourceActionDict,loadActionDict)
        for sourceID in self.source_dict.keys():
            self._handle_source_step(sourceID)

        self.timestep+=1
        done = False if self.timestep < self.max_timestep else True
        return [self.source_feedback_dict.copy(), self.load_feedback_dict.copy(), done]



    def _update_action_dicts(self,sourceActionDict, loadActionDict):
        for sourceID in sourceActionDict.keys():
            self.source_action_dict[sourceID] = sourceActionDict[sourceID]
        for loadID in loadActionDict.keys():
            self.load_action_dict[loadID] = loadActionDict[loadID]
        # print (self.source_action_dict)


    def _handle_source_step(self, sourceID):

        demand = 0.0

        loadIDs = self.source_dict[sourceID].get_loads()
        for loadID in loadIDs:
            curdemand = self.load_dict[loadID].step(self.timestep, self.timestep_size, self.load_action_dict[loadID])
            demand += curdemand[0]+curdemand[1]
            self.load_feedback_dict[loadID][1][0] = curdemand[0]
            if self.test_mode:
                self.load_feedback_dict[loadID][1][1] = curdemand[1]
            else:
                self.load_feedback_dict[loadID][1][1] = curdemand[1]+curdemand[0]*curdemand[2]

        demand += self.source_dict[sourceID].dumb_load_range.generate_demand()

        self.source_dict[sourceID].update_current_demand(demand)
        self._prepare_source_feedback(sourceID)

        for loadID in loadIDs:
            self._prepare_load_feedback(loadID,sourceID)


    def _prepare_load_feedback(self,loadID, sourceID):
        self.load_feedback_dict[loadID][1][2] = self.source_dict[sourceID].get_previous_price()           #Prepare Load Reward
        # if self.load_action_dict[loadID] is Load.action_space[2]:
        #     self.load_feedback_dict[loadID][1][0] *= 2000

        self.load_feedback_dict[loadID][0] = [self.load_dict[loadID].get_battery().get_current_battery_percentage(),
                                              get_last_k(self.source_dict[sourceID].get_prices(), self.load_dict[loadID].get_look_ahead()),
                                              self.load_dict[loadID].get_battery().get_mean_battery(),
                                              self.load_dict[loadID].get_battery().get_variance_battery()]         #Prepare Load Observation


    def _prepare_source_feedback(self, sourceID):
        # print('Action for source %d is %d' % (sourceID,self.source_action_dict[sourceID]))
        self.source_feedback_dict[sourceID][1] = self.source_dict[sourceID].step(self.source_action_dict[sourceID])
        self.source_feedback_dict[sourceID][0] = get_last_k(self.source_dict[sourceID].get_demands(), self.source_dict[sourceID].get_look_ahead())


    def add_dumb_loads(self, sourceID, n=None, ranges = None):
        if sourceID not in self.source_dict.keys():
            source_param_dict = {sourceID:{}}
            source_param_dict[sourceID]['sourceID'] = sourceID
            self.add_source(source_param_dict[sourceID])
        self.source_dict[sourceID].add_dumb_loads(n,ranges)



    def add_connections(self,connect_graph, source_param_dict = None, load_param_dict = None):
        """

        :param connect_graph: dictionary with key = sourceID, value = list of loadIDs connected to sourceID
        :param source_param_dict: dictionary with key = sourceID, value = params dictionary for initialization of corresponding source
        :param load_param_dict: dictionary with key = loadID, value = params dictionary for initialization of corresponding load

        """

        if source_param_dict is None:
            source_param_dict = {}

        if load_param_dict is None:
            load_param_dict = {}

        for sourceID in connect_graph.keys():

            try:
                source_param_dict[sourceID]['sourceID'] = sourceID
                self.add_source(source_param_dict[sourceID])
            except KeyError:
                if not sourceID in self.source_dict.keys():
                    source_param_dict[sourceID] = {}
                    source_param_dict[sourceID]['sourceID'] = sourceID
                    self.add_source(source_param_dict[sourceID])

            for loadID in connect_graph[sourceID]:
                try:
                    load_param_dict[loadID]['loadID'] = loadID
                    self.add_load_to_source(load_param_dict[loadID], sourceID)
                except KeyError:
                    if loadID in self.load_dict.keys():
                        self.source_dict[sourceID].add_load(loadID)
                    else:
                        load_param_dict[loadID] = {}
                        load_param_dict[loadID]['loadID'] = loadID
                        self.add_load_to_source(load_param_dict[loadID], sourceID)



    def remove_connections(self,connect_graph):
        for sourceID in connect_graph.keys():
            for loadID in connect_graph[sourceID]:
                self.source_dict[sourceID].remove_load(loadID)


    def reset(self, battery_reset = False):
        """

        :param battery_reset: false - continuous, true - random, 0 - battery = 0%
        :return:
        """
        temp = 0
        self.timestep = 0
        for i in range(self.look_ahead):
            temp = self.step()

        self.timestep = 0

        for sourceID in self.source_dict.keys():
            self.source_dict[sourceID].reset_day()

        for loadID in self.load_dict.keys():
            self.load_dict[loadID].reset_day(battery_reset)

        return temp

    def set_timestep_size(self, timestep):
        self.timestep_size = timestep

    def get_timestep_size(self):
        return self.timestep_size

    def get_current_timestep(self):
        return self.timestep

    def set_look_ahead(self, lookahead):
        self.look_ahead = lookahead

    def get_look_ahead(self):
        return self.look_ahead

    def get_load_action_space(self):
        return Load.action_space

    def get_source_action_space(self):
        return Source.action_space

    def sample_source_action(self,sourceID):
        return self.source_dict[sourceID].sample_action()

    def sample_load_action(self,loadID):
        return self.load_dict[loadID].sample_action()

    def get_price_bounds(self, sourceID):
        return self.source_dict[sourceID].price_bounds.get_bounds()

    def get_battery_bounds(self, loadID):
        return self.load_dict[loadID].get_battery().get_battery_bounds()

    def get_demand_bounds(self, loadID):
        return self.load_dict[loadID].demand_bounds.get_bounds()

    def get_overall_demand_bounds(self, sourceID):
        return self.source_dict[sourceID].demand_bounds.get_bounds()

    def get_max_timestep(self):
        return self.max_timestep

    def get_battery_capacity(self, loadID):
        return self.load_dict[loadID].get_battery().battery_capacity