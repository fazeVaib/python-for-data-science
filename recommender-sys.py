import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')

# train model
model.fit(data['train'], epochs=30, num_threads=2)


def trialRecomm(mode, data, user_ids):

    # no. of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print results
        print("User {}".format(user_id))
        print("    Knows positives:")

        for x in known_movies[:3]:
            print("        {}".format(x))
        
        print("    Recommended:")

        for x in top_items[:3]:
            print("        {}".format(x))


trialRecomm(model, data, [10, 20, 30])