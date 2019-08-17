# Reinforcement Learning for energy storage optimization in the Smart Grid

## Pranav Bijapur, Pranav Chakradhar, Rishab M, Srinivas S

### Feb 2018 - May 2018

#### Under the MIT license

In the existing grid, there is a large demand-supply mismatch which results in unutilized power potential. 

A Smart Grid is an electrical grid which provides a platform for com- munication between Source and Loads. Reinforcement Learning (RL) has been shown to achieve excellent performance in dynamic power and cost management of systems. This project capitalizes on the above observations, and aims to provide a platform for implementing RL agents on a simulation of the Smart Grid for analytical purposes. 

We implemented a general, extensible Environment of a Smart Grid with the abillity to simulate interactions between multiple Sources and Loads. Using the Environment, we implemented RL Battery Agents - specifically, using Q-learning and SARSA. We also analysed on a use case of the smart grid: Implementation of a smart Battery Agent to power usage optimizations in a Dynamic Pricing scenario.

There are 2-state (price, battery level) and 4-state (2-state + mean battery level, variance battery level) models of the Battery Agent. We also developed a "moving buckets" algorithm to incorporate states with highly uneven distributions of visit frequency into the model.
