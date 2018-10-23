import numpy as np
import sys
import math
import copy
import matplotlib.pyplot as plt

class MDP_bus(object):

    def __init__(self, q_cap, s_cap, t_prob, s_cost, q_cost, discount):
        """
        Initialize the MDP parameters
        :param q_cap: capacity of the queue
        :param s_cap: capacity of shuttle
        :param t_prob: transition prob, list of tuple, [(arrival, prob),..]
        :param s_cost: shuttle cost
        :param q_cost: per customer per time period waiting cost
        """

        # state: number of people in waiting
        # This is the initial value for each state, where state is the index
        self.value = [0] * (q_cap+1)
        # This is the initial policy for each state
        self.policy = [0] * (q_cap+1)
        # Problem parameter
        self.prob = t_prob
        self.q_cap = q_cap
        self.s_cap = s_cap
        self.s_cost = s_cost
        self.q_cost = q_cost
        self.discount = discount


    def __transit_step(self, s, a):
        """
        Compute one step transition result, starting from state s, taking action a
        s: current state, integer
        a: action space, integer
        :return:
        prob:  probability of next state
        s_new: possible next state
        r: reward for every possible next state
        """
        prob = []
        s_new = []
        reward = []
        r = a * self.s_cost + max(s - a * self.s_cap, 0) * q_cost
        for i in self.prob:
            prob.append(i[1])
            s_new.append( min( max(s - a*self.s_cap, 0) + i[0], self.q_cap) )
            reward.append(r)
        return prob, s_new, reward


    def value_iteration(self, T, threshold = 0.0001):
        """
        Value iteration algorithm: update the value for each state
        Assume the bus is dispatched at the beginning of each period, arrival is at the end of each period
        Loop will stop if the value change is less than threshold, or number of period is reached
        :param T: number of period
        :return: None
        """

        for t in range(T):

            old_value = copy.copy(self.value)

            # This is the initial state
            for state in range(self.q_cap + 1):

                # This is all the action it can take
                bus_max = int(math.ceil(state*1.0/self.s_cap))
                min_cost = sys.maxint
                for action in range(bus_max+1):
                    prob, s_new, reward = self.__transit_step(state, action)
                    print("Period {} transition from state {} by taking action {} -> prob = {}, new_state={}, cost={}".format(t, state, action, prob, s_new, reward))
                    # compute the expected reward, by taking action a
                    tot_reward = reward + self.discount * np.array([self.value[s1] for s1 in s_new])
                    tmp = sum(x*y for x, y in zip(prob, list(tot_reward)))
                    if tmp < min_cost:
                        min_cost = tmp
                    print("expected cost at the current state by this transition = {}".format(tmp))
                # update the value of state s using the minimum cost possible
                self.value[state] = min_cost
                print("At t={0}: Value[{1}]={2}".format(t, state, self.value[state]))

            # check if value change is minimal to stop loop
            max_diff = max(abs(i-j) for i, j in zip(old_value, self.value))
            if max_diff <= threshold:
                print("Value iteration converges!")
                break

    def policy_iteration(self, threshold = 0.0001):
        """
        Do policy iteration
        :return: None
        """
        converge = False
        while converge is not True:
            # Step 1: Policy evaluation - update values under a fixed policy
            step1_cnt = 0
            while True:
                old_value = copy.copy(self.value)
                # This is the initial state
                for state in range(self.q_cap + 1):

                    min_cost = sys.maxint

                    action = self.policy[state]
                    prob, s_new, reward = self.__transit_step(state, action)
                    #print("Transition from state {} by taking action {} -> prob = {}, new_state={}, cost={}".format(state, action, prob, s_new, reward))
                    tot_reward = reward + self.discount * np.array([self.value[s1] for s1 in s_new])
                    tmp = sum(x * y for x, y in zip(prob, list(tot_reward)))
                    if tmp < min_cost:
                        min_cost = tmp
                        self.value[state] = min_cost
                max_diff = max(abs(i - j) for i, j in zip(old_value, self.value))
                if max_diff <= threshold:
                    print("Step 1 finished with #iteration={}".format(step1_cnt))
                    break
                step1_cnt += 1

            # Step 2: Update policy based on the current state values
            old_policy = copy.copy(self.policy)
            for state in range(self.q_cap + 1):
                bus_max = int(math.ceil(state * 1.0 / self.s_cap))
                for action in range(bus_max + 1):
                    prob, s_new, reward = self.__transit_step(state, action)
                    tot_reward = reward + self.discount * np.array([self.value[s1] for s1 in s_new])
                    tmp = sum(x * y for x, y in zip(prob, list(tot_reward)))
                    if tmp < self.value[state]:
                        self.policy[state] = action
            print("Step 2 finished")
            if old_policy == self.policy:
                converge = True
                print("Optimal policy = ")
                for state, policy in enumerate(self.policy):
                    print("State {} use policy {}".format(state, policy))


    def plot_result(self):
        """plot MDP result"""
        #print(self.value)
        x = range(0, 201)
        plt.plot(x, self.value)
        plt.savefig("value_function.png")
        plt.show()

if __name__ == "__main__":
    # input
    q_cap = 200
    s_cap = 15
    t_prob = [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)]
    s_cost = 100
    q_cost = 2
    discount = 0.95
    # run MDP
    shuttle = MDP_bus(q_cap, s_cap, t_prob, s_cost, q_cost, discount)
    shuttle.value_iteration(500)
    #shuttle.policy_iteration()
    shuttle.plot_result()