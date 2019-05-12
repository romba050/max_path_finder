"""
"Dynamic programming" is a mathematical paradigm (it has nothing to do with programming per se) used to solve problems
where the solution relies on the solution of a subproblem (in this case the subproblem is the same pyramid with one less
 row). You save part of the solution in a so called dynamic programming matrix and then do backtracking through the
 matrix to get your full solution.


The dp (dynamic programming) matrix has the same dimension as your input triangle/graph/matrix. By filling out the dp
matrix, you consider all allowed paths in an efficient way. At this point you basically already solved the problem,
 however you want to print the path. But because the matrix only saves the max sums up to that point in the input
 graph/matrix, you have 2 possibilities to go get the path out of the dp matrix.

Poss. 1 (what I did): you backtrack through the matrix starting at the max sum of the bottom row going up.
Poss. 2: you save a 3rd matrix which remembers which path you take when filling out the dp matrix.

Poss1 is technically a little slower. But because the difference in runtime between poss. 1 and 2 is linear
in the number of integers of the triangle, it doesn't matter in the grand scheme of things which one you take.
And poss. 1 is easier to implement so it is more often seen in this kind of problems.
"""

import numpy as np

input_file = 'triangle_mat2.txt'


def get_num_lines(file):
    """ returns number of lines of a file """
    return sum(1 for line in open(file, 'r'))


mat = []
row = []
num_lines = get_num_lines(input_file)
# size of square matrix is equal to number of lines, which is equal to number of rows (empty newline is not counted!)
square_size = num_lines

with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        row = np.array([int(s) for s in line.split()])
        # pad (-> padding) row with 0s
        row = np.append(row, [0]*(square_size-len(row)))
        mat.append(row)

mat = np.array(mat)


def find_max_path(graph):
    """ Finds path from root to leaf of graph with max sum of alternating odd and even numbers.

     graph: given in form of a numpy 2D-array, where, interpreted as a lower left triangle matrix,
            the entries are the node values and the directed edges go from each value to the one below and the one on
            the below right diagonal. """

    n, _ = graph.shape
    dp_mat = np.zeros((n, n))
    odd = True

    ### create dynamic programming matrix
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            if i == 0:
                # initialization: root node is copied
                dp_mat[i, j] = graph[i, j]
                continue
            if j == 0:
                # take the one above (dp_mat[i-1, j]), since there is no left upper father to this node in the graph
                dp_mat[i, j] = graph[i, j] + dp_mat[i-1, j]

            else:
                left = dp_mat[i-1, j-1]
                top = dp_mat[i-1, j]
                # you can only go a path if it is both the better choice for the sum
                # and we change from an odd to an even (or vice-versa) in the original graph
                if left >= top and (graph[i-1, j-1] % 2) != (graph[i, j] % 2):
                    dp_mat[i, j] = graph[i, j] + left
                    continue
                if top >= left and (graph[i-1, j] % 2) != (graph[i, j] % 2):
                    dp_mat[i, j] = graph[i, j] + top
    print('Dynamic progamming matrix:')
    print(dp_mat)
    print()

    ### backtracking
    path = []
    start_j = np.argmax(dp_mat[-1, :])
    #print(f'Starting coordinates: {n-1, start_j}')
    print(f'Max sum: {dp_mat[n-1, start_j]}\n')
    path.append((n-1, start_j))

    j = start_j
    for i in range(n-1, 0, -1):  # goes from n-1 to 1
        # preceeding node to choose is only the maximum value in the graph
        # both get added the same value (graph[i, j]) in the algorithm so we don't have to account for this when
        # comparing to choose next node
        if i == j:  # there is no top node if we are at diagonal
            path.append((i-1, j-1))
            j = j-1
            continue
        if j == 0:  # there is no left node if we are at first column
            path.append((i-1, j))
            continue
        left = graph[i - 1, j - 1]
        top = graph[i - 1, j]
        if left > top:  # compare between top node and left node
            path.append((i-1, j-1))
            j = j-1
        else:
            path.append((i-1, j))

    # reverse array because we did backtracking
    path.reverse()

    return path


path = find_max_path(mat)
print(f'Path:\n{[node for node in path]}\n')

n = len(path)
path_x = [t[0] for t in path]
path_y = [t[1] for t in path]
path_matrix = np.zeros((n, n))
path_matrix[path_x, path_y] = 1
print(f'Path visualized:\n{path_matrix}')
