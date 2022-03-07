import random
from typing import TypeVar, List, Tuple, NamedTuple, Dict
import csv
from collections import Counter, defaultdict
from ds2.linear_algebra import Vector, distance
from matplotlib import pyplot as plt
import requests
import tqdm
X = TypeVar('X') #generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
  """split data into fractions [prob, 1-prob]"""
  data = data[:] #make a shallow copy
  random.shuffle(data)
  cut = int(len(data) * prob)
  return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

#the proportions should be correct
assert len(train) == 750
assert len(test) == 250

#and the original data should be preserved in some order
assert sorted(train + test) == data

Y = TypeVar('Y')

def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
  #generate indices and split them
  idxs = [i for i in range(len(xs))]
  train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
  
  return ([xs[i] for i in train_idxs],
          [xs[i] for i in test_idxs],
          [ys[i] for i in train_idxs],
          [ys[i] for i in test_idxs])
          
xs = [x for x in range(1000)] #xs are 1...1000
ys = [2*x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250
assert all(y == 2*x for x, y in zip(x_train, y_train))
assert all(y == 2*x for x, y in zip(x_test, y_test)

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
  correct = tp + tn
  total = tp + fp + fn + tn
  return correct / total
  
assert accuracy(70, 4930, 13930, 981070) == 0.98114

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
  return tp / (tp + fp)
  
assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
  return tp / (tp + fn)
  
assert recall(70, 4930, 13930, 981070) == 0.005

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
  p = precision(tp, fp, fn, tn)
  r = recall(tp, fp, fn, tn)
  return 2 * p * r / (p + r)
  
def raw_majority_vote(labels: List[str]) -> str:
  votes = Counter(labels)
  winner, _ = votes.most_common(1)[0]
  return winner
  
assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

def majority_vote(labels: List[str]) -> str:
  """assumes that labels are ordered from nearest to farthest"""
  vote_counts = Counter(labels)
  winner, winner_count = vote_counts.most_common(1)[0]
  num_winners = len([count for count in vote_counts.values() if count == winner_count])
  
  if num_winners == 1:
    return winner
  else:
    return majority_vote(labels[:-1])
    
assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

class LabeledPoint(NamedTuple):
  point: Vector
  label: str
  
def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
  #order labeled points from nearest to farthest
  by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
  k_nearest_labels = [lp.label for lp in by_distance[:k]]
  return majority_vote(k_nearest_labels)
  
data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

with open('iris.data', 'w') as f:
  f.write(data.text)
    
def parse_iris_row(row: List[str]) -> LabeledPoint:
  """sepal_length, sepal_width, petal_length, petal_width, class"""
  measurements = [float(value) for value in row[:-1]]
  # class is e.g. "Iris-virginica"; we just want "virginica"
  label = row[-1].split("-")[-1]
    
  return LabeledPoint(measurements, label)

with open('iris.data') as f:
  reader = csv.reader(f)
  iris_data = [parse_iris_row(row) for row in reader]
    
# We'll also group just the points by species/label so we can plot them.
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
  points_by_species[iris.label].append(iris.point)
    
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x']  # we have 3 classes, so 3 markers
    
fig, ax = plt.subplots(2, 3)
    
for row in range(2):
  for col in range(3):
    i, j = pairs[3 * row + col]
    ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
    ax[row][col].set_xticks([])
    ax[row][col].set_yticks([])
    
    for mark, (species, points) in zip(marks, points_by_species.items()):
      xs = [point[i] for point in points]
      ys = [point[j] for point in points]
      ax[row][col].scatter(xs, ys, marker=mark, label=species)
    
ax[-1][-1].legend(loc='lower right', prop={'size': 6})
plt.show()

random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0
for iris in iris_test:
  predicted = knn_classify(5, iris_train, iris.point)
  actual = iris.label
  
  if predicted == actual:
    num_correct += 1
    
  confusion_matrix[(predicted, actual)] += 1
  
pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)

def random_point(dim: int) -> Vector:
  return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int) -> List[float]:
  return [distance(random_point(dim), random_point(dim))
          for _ in range(num_pairs)]
          
dimensions = range(1, 101)
    
avg_distances = []
min_distances = []
    
random.seed(0)
for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensionality"):
  distances = random_distances(dim, 10000)
  avg_distances.append(sum(distances) / 10000)
  min_distances.append(min(distances))
  
min_avg_ratio = [min_dist / avg_dist for min_dist, avg_dist in zip(min_distances, avg_distances)]
