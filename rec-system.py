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
model.fit(data['train'], epochs=30)


def trialRecomm(mode, data, user_ids):
    # no. of users and movies in training data
    n_users, n_items = data['train'].shape

    