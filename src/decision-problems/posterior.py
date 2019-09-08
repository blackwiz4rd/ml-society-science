

# Assume the number of models is equal to n=len(prior).  The argument
# P is an n-by-m array, where m is the number of possible
# predictions so that P[i][j] is the probability the i-th model assignsm to the j-th outcome. The outcome is a single number in 1, m.
import numpy as np
import random

## Calculate the posterior given a prior belief, a set of predictions, an outcome
## - prior: belief vector so that prior[i] is the probabiltiy of mdoel i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome (rain[t] = 1 if rains)
def get_posterior(prior, P, outcome):
    # n_models = len(prior)
    n_models = prior.size
    posterior = np.zeros(3)
    cumulative_prob = 0
    for mu in range(n_models):
        cumulative_prob += P[mu][outcome] * prior[mu]
        posterior[mu] = P[mu][outcome] * prior[mu]
    ## So probability of outcome for model i is just...
    return posterior / cumulative_prob


## Get the probability of the specific outcome given your current
## - belief: vector so that belief[i] is the probabiltiy of model i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_marginal_prediction(belief, P, outcome):
    n_models = belief.size
    outcome_probability = 0
    for model in range(n_models):
        # outcome_probability = P[model][outcome] + belief
        outcome_probability += P[model][outcome] * belief[model]
    return outcome_probability

## In this function, U[action,outcome] should be the utility of the action/outcome pair
def get_expected_utility(belief, P, action, U):
    # n_models = len(belief)
    n_outcomes = np.shape(P)[1] # number of columns = 2 = outcomes = {0=no rain,1=rain}
    utility = 0
    for x in range(n_outcomes):
        # utility += U[action][x] * get_marginal_prediction(belief, P, x)
        utility += U[action,x] * get_marginal_prediction(belief, P, x)
    return utility

## The best utility is the one which maximizes the reward
def get_best_action(belief, P, U):
    # n_models = belief.size
    n_actions = np.shape(U)[0] # number of rows = 2 = actions
    best_action = 0 # can be {0, 1}
    best_U = get_expected_utility(belief, P, best_action, U)

    for a in range(1,n_actions):
        U_a = get_expected_utility(belief, P, a, U)
        if  U_a > best_U:
            best_U = U_a
            best_action = a
    return best_action


T = 4 # number of time steps (days)
n_models = 3 # number of models (CNN, SMHI, YR)

# build predictions for each station of rain probability, given as hypotesis
prediction = np.matrix('0.1 0.2 0.3 0.4; 0.4 0.5 0.6 0.7; 0.7 0.8 0.9 0.99')
print("prediction: probability of rain for the model on that day")
print(prediction, "\n")

n_outcomes = 2 # 0 = no rain, 1 = rain

## we use this matrix to fill in the predictions of stations
P = np.zeros([n_models, n_outcomes]) # matrix = [[0,0],[0,0],[0,0]] = prediction of rain for each model
belief = np.ones(n_models) / n_models; # = [1/3, 1/3, 1/3] = uniform prior for each model
rain = [1, 0, 1, 1]; # rains on day t if 1 (Y)
print("rain on day t\n",rain)

U  = np.matrix('1 -1; 0 0')
print("U \n", U)

for t in range(T): # t = 0..3 = day
    # utility to loop to fill in predictions for that day
    # there is actually no need to make a matrix, the info could be derived when it is necessary
    for model in range(n_models):
        P[model,1] = prediction[model,t] # prediction gives rain probabilities
        P[model,0] = 1.0 - prediction[model,t] # so no-rain probability is 1 - that.
    print("matrix P: no rain, rain for each model on day %i" % t)
    print(P)

    # on day t
    # marginal since it is on a specific outcome
    probability_of_rain = get_marginal_prediction(belief, P, 1) # outcome = 1 means predict rain
    print("probability of rain = marginal prediction \n", probability_of_rain)
    action = get_best_action(belief, P, U)
    print("action %i, rain %i, U[action,rain[t]] %i " %(action, rain[t], U[action, rain[t]]))
    belief = get_posterior(belief, P, rain[t]) # careful! we are overriding belief
    print("belief", belief)
