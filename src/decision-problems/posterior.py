

# Assume the number of models is equal to n=len(prior).  The argument
# P is an n-by-m array, where m is the number of possible
# predictions so that P[i][j] is the probability the i-th model assignsm to the j-th outcome. The outcome is a single number in 1, m.
import numpy as np
import random

## Calculate the posterior given a prior belief, a set of predictions, an outcome
## - prior: belief vector so that prior[i] is the probabiltiy of mdoel i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_posterior(prior, P, outcome):
    # n_models = len(prior)

    ## So probability of outcome for model i is just...
    return 0
    # return posterior


## Get the probability of the specific outcome given your current
## - belief: vector so that belief[i] is the probabiltiy of model i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_marginal_prediction(belief, P, outcome):
    n_models = belief.size
    outcome_probability = 0
    for mu in range(n_models):
        outcome_probability = P[mu][outcome] + belief
    return outcome_probability

## In this function, U[action,outcome] should be the utility of the action/outcome pair
def get_expected_utility(belief, P, action, U):
    # n_models = len(belief)
    n_outcomes = np.shape(P)[1]
    utility = 0
    for x in range(n_outcomes):
        utility += get_marginal_prediction(belief, P, x) * U[action][x]
    return utility

def get_best_action(belief, P, U):
    # n_models = belief.size
    n_actions = np.shape(U)[0]
    print("n_actions", n_actions)
    best_action = 0
    best_U = get_expected_utility(belief, P, best_action, U)
    for a in range(n_actions):
        U_a = get_expected_utility(belief, P, a, U)
        if  U_a > best_U:
            best_U = U_a
            best_action = a
    return best_action


# T = 4 # number of time steps
T = 1
n_models = 3 # number of models

# build predictions for each station of rain probability
prediction = np.matrix('0.1 0.2 0.3 0.4; 0.4 0.5 0.6 0.7; 0.7 0.8 0.9 0.99')
print("prediction \n", prediction)

n_outcomes = 2 # 0 = no rain, 1 = rain

## we use this matrix to fill in the predictions of stations
P = np.zeros([n_models, n_outcomes]) # matrix = [[0,0],[0,0],[0,0]] - prediction of stations
belief = np.ones(n_models) / n_models; # = [1/3, 1/3, 1/3]
rain = [1, 0, 1, 1];


for t in range(T):
    # utility to loop to fill in predictions for that day
    for model in range(n_models):
        P[model,1] = prediction[model,t] # the table predictions give rain probabilities
        P[model,0] = 1.0 - prediction[model,t] # so no-rain probability is 1 - that.
    print("matrix P \n", P)

    probability_of_rain = get_marginal_prediction(belief, P, 1)
    print("probability of rain \n", probability_of_rain)
    U  = np.matrix('1 -1; 0 0')
    print("belief", belief)
    print("U \n", U)
    action = get_best_action(belief, P, U)
    print(action, rain[t], U[action, rain[t]])
    belief = get_posterior(belief, P, rain[t])
    print(belief)
