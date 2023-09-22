
import numpy as np
import pandas as pd
from random import randrange
from itertools import zip_longest

def min_max(arr: np.array):
    return np.min(arr), np.max(arr)
    
def is_range_intersect(range1, range2):
    return  pd.Interval(*min_max(range1), closed='both').overlaps(pd.Interval(*min_max(range1), closed='both')) 

def random_colour():
    return (randrange(255), randrange(255), randrange(255))

def pairwise(iterable):
    return zip_longest(iterable, iterable[1:], fillvalue=iterable[0])

def iter_list_from_value(lst, value):
    # Iterate through list, starting from value, and looping to start to end at value
    i = lst.index(value)
    yield from lst[i:] + lst[:i]