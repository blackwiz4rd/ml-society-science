## Generate data first

import numpy as np

## Calculate the distance between two people
## (helper function, if necessary)
def distance(x, y):

## Get the similarity between two people.
## I recommend that this actually retunrs a similarity vector that sums to 1, and has a zero value for the user himself
def get_similarity(data, u):
    n_users = data.shape[0]
    similarity = np.ones(n_users) / (n_users - 1)
    similarity[u] = 0
    


## data[u,m] = 0 if somebody hasn't watched a movie
def infer_ratings(data):
    n_users = data.shape[0]
    n_movies = data.shape[1]
    inferred_ratings = data
    ## Do a loop wher you fill in values for each user and movie
    for u in range(n_users):
        
        ## calculate neighbour weights
        ## Get prediction for movies
    return inferred_ratings


    
        
def generate_random_data(n_users, n_movies, gamma):
    data = np.zeros([n_users, n_movies])
    rating_dist = [0.1, 0.2, 0.3, 0.3, 0.1]
    for u in range(n_users):
        n_ratings = np.random.geometric(gamma)
        ratings = np.random.permutation(range(n_movies))[0:n_ratings]
        for m in ratings:
            data[u,m] = 1 + np.random.choice(5,1, p=rating_dist)
    return data
