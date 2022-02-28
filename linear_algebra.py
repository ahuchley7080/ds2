from typing import List, Tuple, Callable

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v,w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v,w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v,w))


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function


def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))


#def distance(v: Vector, w: Vector) -> float:  # type: ignore
    #return magnitude(subtract(v, w))
    
Matrix = List[List[float]]

assert add([1, 2, 3], [10,9,8]) == [11,11,11], "something's wrong with add()"
assert subtract([11,11,11], [1, 2, 3]) == [10,9,8], "trouble with subtract()"
assert vector_sum([[3,4], [5,6], [7,8]]) == [15, 18], "vector_sum() problem"
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4], "oopsie vector_mean()"
assert dot([1, 2, 3], [4, 5, 6]) == 32, "dot() issue"  # 1 * 4 + 2 * 5 + 3 * 6
assert sum_of_squares([1, 2, 3]) == 14, "sum_of_squares() fail"  # 1 * 1 + 2 * 2 + 3 * 3
assert magnitude([3, 4]) == 5, "issue with magnitude()"

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


# The underscores indicate that these are "private" functions, as they're
# intended to be called by our median function but not by other people
# using our statistics library.
def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def shape(A: Matrix) -> Tuple[int, int]:
    """returns # of rows of a, # of columns of a"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 #number of elements in first row
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """returns i-th row of A as a Vector"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """returns j-th column of A as a Vector"""
    return [A_i[j] for A_i in A] #jth element of row A_i for each row A_i

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """returns a num_rows x num_cols matrix whose i,j-tih entry is entry_fn(i,j)"""
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]
#given i, create a list, one list for each i

def identity_matrix(n: int) -> Matrix:
    """returns the nxn identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extra assert statements to test all of the functions you will need
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3
assert magnitude([3, 4]) == 5
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns
assert distance([1,1],[4,1]) == 3.0
assert squared_distance([1,2,3],[2,3,4]) == 3
assert scalar_multiply(2, [1,2,3]) == [2,4,6]
assert magnitude([0,0,4,3]) == 5.0

# Work on an Identity Matrix
id = [  [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1] ]

assert get_column(id,2) == [0, 0, 1, 0, 0]
assert get_row(id,2) == [0, 0, 1, 0, 0]
assert get_column(id,2) == get_row(id,2)
assert identity_matrix(5) == id
assert make_matrix(5,5, lambda i,j: 1 if i == j else 0) == id
assert shape(id) == (5,5)
