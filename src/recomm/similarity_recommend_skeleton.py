## Generate data first

import numpy as np

## Calculate the distance between two people
## (helper function, if necessary)
def distance(x, y):
    identity_func = np.logical_and(x != 0, y != 0)
    return np.sum(np.abs(x[identity_func]-y[identity_func])) # ignoring not shared movies

## Get the similarity between two people.
## I recommend that this actually retunrs a similarity vector that sums to 1, and has a zero value for the user himself
def get_similarity(data, u):
    n_users = data.shape[0]
    # similarity = np.ones(n_users) / (n_users - 1) # uniform similarity
    # similarity[u] = 0
    similarity = []
    for j in range(n_users):
        if u != j:
            print(data[u], data[j])
            similarity.append(np.exp(-distance(data[u],data[j])))
        else:
            similarity.append(0)
    print("weight %i: " % u)
    print(similarity)
    similarity = np.array(similarity)
    tot_divider = np.sum(np.exp(-similarity))

    # similarity = similarity/np.sum(similarity)
    print("user %i similarities sum to 1? %i" %(u, np.sum(similarity)) )
    return similarity / tot_divider

## data[u,m] = 0 if somebody hasn't watched a movie
def infer_ratings(data):
    n_users = data.shape[0]
    n_movies = data.shape[1]
    inferred_ratings = data
    ## Do a loop wher you fill in values for each user and movie
    similarity = []
    print(similarity)
    for u in range(n_users):
        ## calculate neighbour weights
        ## Get prediction for movies
        similarity.append(get_similarity(data, u))
        for m in range(n_movies):
            # use data as well
            inferred_ratings[u, m] = np.sum(np.dot(similarity[-1] * data[u]))
    print("similarity matrix")
    similarity = np.array(similarity)
    print(similarity)
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

np.random.seed(10)
data = generate_random_data(5,8,0.5)
print("randomly generated data")
print(data)
# fill the zeros
print("data + inferred ratings")
inferred_ratings = infer_ratings(data)
print(inferred_ratings)
