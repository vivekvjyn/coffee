import csv

import pandas
from scipy.sparse import coo_matrix
import numpy as np
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares


def load_data_matrix(user_artist_counts_path):
    """Load a CSV file containing user,artist_id,count lines into a matrix
    that can be used to build a CF model"""
    # from https://github.com/benfred/implicit/blob/871e0c7229b012108131b6211cd617e23a3b24bf/implicit/datasets/lastfm.py#L58
    data = pandas.read_table(
        user_artist_counts_path, sep=',', header=1, usecols=[0, 1, 2], names=["user", "artist", "plays"], dtype={'user': str, 'artist': str, 'plays': np.int32}, na_filter=False
    )
    data["user"] = data["user"].astype("category")
    data["artist"] = data["artist"].astype("category")
    plays = coo_matrix(
        (
            data["plays"].astype(np.float32),
            (data["user"].cat.codes.copy(), data["artist"].cat.codes.copy()),
        )
    ).tocsr()
    return data["artist"].cat.categories, data["user"].cat.categories, plays


def build_model(plays):
    # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
    # and to reduce the weight given to popular items
    artist_user_plays = bm25_weight(plays, K1=100, B=0.8)
    model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
    model.fit(artist_user_plays.tocsr())
    return model


def get_artist_map(musicbrainz_artist_path):
    """Load a MusicBrainz data file which maps artist MBIDs to the Artist name"""
    artist_map = {}
    with open(musicbrainz_artist_path) as fp:
        r = csv.DictReader(fp)
        for line in r:
            artist_map[line['artist_mbid']] = line['name']
    return artist_map


def artist_index(artists, artist_mbid):
    """Given an artist MBID, find its index in the matrix used to build the CF model"""
    positions = np.nonzero(artists == artist_mbid)[0]
    if positions.size == 0:
        raise ValueError("Not found")
    if positions.size > 1:
        raise ValueError("Unexpectedly found >1")
    return positions[0]

