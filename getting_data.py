#chapters 9 and 10 of the textbook
#egrep.py
import sys, re
import math
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

#sys.argv is the list of command-line arguments
#sys.argv[0] is the name of the program itself
#sys.argv[1] will be the regex specified at the command line
regex = sys.argv[1]

#for every line passed into the script
for line in sys.stdin:
  #if it matches the regex, write it to stdout
  if re.search(regex, line):
    sys.stdout.write(line)
    
#line_count.py
count = 0
for line in sys.stdin:
  count += 1
  
#print goes to sys.stdout
print(count)

from collections import Counter, defaultdict
try:
  num_words = int(sys.argv[1])
except:
  print("usage: most_common_words.py num_words")
  sys.exit(1) #nonzero exit code indicates error
  
counter = Counter(word.lower()
                  for line in sys.stdin
                  for word in line.strip().split()
                  if word)

for word, count in counter.most_common(num_words):
  sys.stdout.write(str(count))
  sys.stdout.write("\t")
  sys.stdout.write("\n")
  
def get_domain(email_address: str) -> str:
  """split on @ and return the last piece"""
  return email_address.lower().split("@")[-1]

assert get_domain('joelgrus@gmail.com') == 'gmail.com'
assert get_Domain('joel@m.datasciencester.com') == 'm.datasciencester.com'

import csv
from bs4 import BeautifulSoup
import requests

def paragraph_mentions(text: str, keyword: str) -> bool:
  """returns true if a <p> inside the text mentions the keyword"""
  soup = BeautifulSoup(text, 'html5lib')
  paragraphs = [p.get_text() for p in soup('p')]
  return any(keyword.lower() in paragraph.lower() for paragraph in paragraphs)

import json
from dateutil.parser import parse
import os
import webbrowser
from twython import Twython, TwythonStreamer

tweets = []

class MyStreamer(TwythonStreamer):
  def on_success(self, data):
    """python dict representing a tweet"""
    #english-language tweets only
    if data.get('lang') == 'en':
      tweets.append(data)
      print(f"received tweet #{len(tweets)}")
      
    #stop when we've collected enough
    if len(tweets) >= 100:
      self.disconnect()
      
  def on_error(self, status_code, data):
    print(status_code, data)
    self.disconnect()
    
def bucketize(point: float, bucket_size: float) -> float:
  """floor the point to the next lower multiple of bucket_size"""
  return bucket_size*math.floor(point/bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
  """buckets the points and counts how many in each bucket"""
  return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
  histogram = make_histogram(points, bucket_size)
  plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
  plt.title(title)
  
import random
from scratch.probability import inverse_normal_cdf

def random_normal() -> float:
  """Returns a random draw from a standard normal distribution"""
  return inverse_normal_cdf(random.random())

from scratch.statistics import correlation, standard_deviation
from scratch.linear_algebra import Matrix, Vector, make_matrix, vector_mean, subtract, magnitude, dot, scalar_multiply
from scratch.gradient_descent import gradient_step

def correlation_matrix(data: List[Vector]) -> Matrix:
  """returns the len(data) x len (data) matrix whose (i,j)-th entry is the corr. btwn data[i] and data[j]"""
  def correlation_ij(i: int, j: int) -> float:
    return correlation(data[i], data[j])
  
  return make_matrix(len(data), len(data), correlation_ij)

from collections import namedtuple

import datetime

from typing import NamedTuple

from dataclasses import dataclass

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
  """returns mean and st dev for each position"""
  dim = len(data[0])
  means = vector_mean(data)
  stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(dim)]
  
  return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0. 1]
assert stdevs == [2, 1, 0]

def rescale(data: List[Vector]) -> List[Vector]:
  """rescales input data so each position has mean 0 and stdev 1"""
  dim = len(data[0])
  means, stdevs = scale(data)
  #copy of each vector
  rescaled = [v[:] for v in data]
  for v in rescaled:
    for i in range(dim):
      if stdevs[i] > 0:
        v[i] = (v[i] - means[i]) / stdevs[i]
  return rescaled

python -m pip install tqdm
import tqdm

def de_mean(data: List[Vector]) -> List[Vector]:
  """recenters data to have mean 0 in every dimension"""
  mean = vector_mean(data)
  return [subtract(vector, mean) for vector in data]

def direction(w: Vector) -> Vector:
  mag = magnitude(w)
  return [w_i / mag for w_i in w]

def directional_variance(data: List[Vector], w: Vector) -> float:
  """returns variance of x in direction of w"""
  w_dir = direction(w)
  return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
  """gradient of directional variance with respect to w"""
  w_dir = direction(w)
  return [sum(2*dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]

def first_principal_component(data: List[Vector], n: int = 100, step_size: float = 0.1) -> Vector:
  #random guess
  guess = [1.0 for _ in data[0]]
  
  with tqdm.trange(n) as t:
    for _ in t:
      dv = directional_variance(data, guess)
      gradient = directional_variance_gradient(data, guess)
      guess = gradient_step(guess, gradient, step_size)
      t.set_description(f"dv: {dv:.3f}")
      
  return direction(guess)

def project(v: Vector, w: Vector) -> Vector:
  """return the projection of v onto the direction w"""
  projection_length = dot(v, w)
  return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
  """projects v onto w and subtracts result from v"""
  return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
  return [remove_projection_from_vector(v,w) for v in data]

def pca(data: List[Vector], num_components: int) -> List[Vector]:
  components: List[Vector] = []
  for _ in range(num_components):
    component = first_principal_component(data)
    components.append(component)
    data = remove_projection(data, component)
    
  return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
  return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
  return [transform_vector(v, components) for v in data]
