# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Thanks to Google Translate for converting the comments from Mandarin to English

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy

# The distance matrix D_ij value range: [0, 2m+1]. Where 0: diagonal element; 1-6: forward distance; 7-12: reverse distance; 13: both forward and reverse directions are unreachable and not connected.

def floyd_warshall(adjacency_matrix, direct_matrix):  # parameter: Symmetric adjacency matrix that does not distinguish between directions Adjacency matrix that distinguishes between directions
    (batchsize, nrows, ncols) = adjacency_matrix.shape
    shortest_dis = numpy.zeros([batchsize, nrows, ncols])  # set to 0
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef numpy.ndarray[long, ndim=2, mode='c'] M
    cdef numpy.ndarray[long, ndim=2, mode='c'] path
    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj     # cost_ikkj represents the cost
    cdef long* M_ptr  # Pointer operations
    cdef long* M_i_ptr
    cdef long* M_k_ptr
    for BatchNum in range(batchsize):
        oneBatch = adjacency_matrix[BatchNum].astype(numpy.int64)
        # NOTE: `int` was `long` originally
        adj_mat_copy = oneBatch.astype(int, order='C', casting='safe', copy=True)
        assert adj_mat_copy.flags['C_CONTIGUOUS']
        M = adj_mat_copy  # Adjacency Matrix
        path = numpy.zeros([n, n], dtype=numpy.int64) # path[i,j] represents the number of the first point from point i to point j. For example, the shortest path from point i to point j passes through point i, point a, point b, and point j in sequence. Then the value stored in path[i,j] is a.
        M_ptr = &M[0,0]
        # set unreachable nodes distance to 101, Take a maximum value, the number of object features is 0-100.
        for i in range(n):
            for j in range(n):
                path[i][j] = j    # Assume that point i is directly connected to point j, so the first point from i to j is j
                if i == j:
                    M[i][j] = 0  # Diagonal values are not connected
                elif M[i][j] == 0: # Except diagonals
                    M[i][j] = 101   # Not connected
                    path[i][j] = -1  # It means that i and j are not directly connected at this time. When only considering the distance of 1, there is no point that can be passed, so it is set to -1. This value can be changed later.

        # floyed algo main structure
        for k in range(n):
            M_k_ptr = M_ptr + n*k
            for i in range(n):
                M_i_ptr = M_ptr + n*i
                M_ik = M_i_ptr[k]
                for j in range(n):
                    cost_ikkj = M_ik + M_k_ptr[j]
                    M_ij = M_i_ptr[j]
                    if M_ij > cost_ikkj:
                        M_i_ptr[j] = cost_ikkj   # The shortest distance from i to j, the value corresponding to the address is modified directly by the pointer
                        path[i][j] = path[i][k]  # It means that although i and j are not directly connected, they can be indirectly connected. The sequence number of the first point passed from i to j is the same as the sequence number of the first point passed from i to k.

        for i in range(n):
            for j in range(n):
                if M[i][j] >= 6 and M[i][j] < 101:
                    M[i][j] = 6
                elif M[i][j] >= 101:
                    M[i][j] = 13
                if i == j:
                    M[i][j] = 0
        # Consider directionality
        for i in range(n):
            for j in range(n):
                if path[i][j] != -1: # Not equal to -1 means that i can at least reach j, and the stored value represents the number of the first point that i passes through to j. Here we can only distinguish between point i, point a, point b, and point j; the positive and negative directions of i->a.
                    if direct_matrix[BatchNum, i, path[i][j]] != 1: # path[i][j] represents the index of the first point passed when i goes to j, which is set to a. If i and point a are directly connected in the directed graph (equal to 1), it means that it is the forward distance, and the distance is added with 7 to distinguish it. Otherwise, it is the reverse distance.
                        if M[i][j] != 0 and M[i][j] != 13:
                            M[i][j] = M[i][j] + 6 # Add 6 in the reverse direction, not in the forward direction All values are 0~13: 0: diagonal element 1-6: forward distance 7-12: reverse distance 13: both forward and reverse directions are unreachable, no connection

        shortest_dis[BatchNum] = M
    shortest_dis = numpy.array(shortest_dis).astype(numpy.int64)

    return shortest_dis
