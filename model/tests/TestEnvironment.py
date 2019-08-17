from Environment import Environment
env = Environment()
env.add_connections({0:[1,2,3], 1:[4,5,6,7,8]})
env.set_environment_ready()
a = env.reset()
print(a)