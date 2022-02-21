import math
from typing import List

Vector = List[float]

height_weight_age = [70, 170, 40]
#70: inches, 170: pounds, 40: years

grades = [95, 80, 75, 62] #exam1, exam2, exam3, exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    
    return [v_i + w_i for v_i, w_i in zip(v,w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v,w)]
    
assert subtract([5, 7, 9], [4, 5, 6] == [1, 2, 3])

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    #Check that vectors is not empty
    assert vectors, "no vectors provided!"
    
    #Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    
    #the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]
    
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]
    
assert scalar_multiply (2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
    
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be the same length"
    
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
    
assert dot([1, 2, 3], [4, 5, 6]) == 32 #1*4 + 2*5 + 3*6

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)
    
assert sum_of_squares([1, 2, 3]) == 14 #1*1 + 2*2 + 3*3

def magnitude(v: Vector) -> float:
    """returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v)) #math.sqrt is square root function

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w)n)**2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
     """Computes the distance between v and w"""
     return math.sqrt(squared_distance(v, w))
     
def dist_w_magnitude(v: Vector, w: Vector) -> float:
    """computes distance between v and w"""
    return magnitude(subtract(v, w))
     
     
assert add([1, 2, 3], [10,9,8]) == [11,11,11], "something wrong with add()"
assert subtract([11,11,11], [1, 2, 3]) == [10,9,8], "trouble with subtract()"
assert vector_sum([[3,4], [5,6], [7,8]]) == [15, 18], "vector_sum() problem"
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4], "oopsie vector_mean()"
assert dot([1, 2, 3], [4, 5, 6]) == 32, "dot() issue"  # 1 * 4 + 2 * 5 + 3 * 6
