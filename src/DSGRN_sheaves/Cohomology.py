### Cohomology.py
### MIT LICENSE 2024 Alex Dowling

### Based on code by Jeremy Kun: 
### http://jeremykun.com/2013/04/10/computing-homology/

import numpy
import galois

def cohomology(D, E):
    """ Inputs two matrices D, E with D*E=0. Outputs a linearly independent set
        of vectors in ker D which form a basis for ker D / img E under the 
        quotient map"""
    
    A, B, Q = simultaneousReduce(D, E)
    B_pivotRows = [i for i in range(B.shape[0]) 
                   if not numpy.all(B[i, :] == 0*B[i, :])]
    B_reduced = B[B_pivotRows, :].copy().transpose().row_reduce()
    B_reducedRows = []; j = 0
    
    for i in range(B_reduced.shape[0]):
        while j < B_reduced.shape[1] and B_reduced[i,j]!=1:
            j += 1
        if j == B_reduced.shape[1]:
            break
        else:
            B_reducedRows.append(j)
            
    B_spanRows = [B_pivotRows[i] for i in B_reducedRows]
    A_zeroCols = [j for j in range(A.shape[1]) 
                  if numpy.all(A[:, j] == 0*A[:, j])]
    generators = [Q[:, i] for i in A_zeroCols if i not in B_spanRows]
    
    return generators

def row_swap(A, i, j):
    temp = A[i, :].copy()
    A[i, :] = A[j, :]
    A[j, :] = temp

def col_swap(A, i, j):
    temp = A[:, i].copy()
    A[:, i] = A[:, j]
    A[:, j] = temp

def row_scale(A, i, c):
    A[i, :] = A[i, :]*c

def col_scale(A, i, c):
    A[:, i] = A[:, i]*c

def row_combine(A, addTo, rowScale, scaleAmt):
    A[addTo, :] = A[addTo, :] + scaleAmt*A[rowScale, :]
    
def col_combine(A, addTo, colScale, scaleAmt):
    A[:, addTo] = A[:, addTo] + scaleAmt*A[:, colScale]
    
def simultaneousReduce(C, D):
    A = C.copy()
    B = D.copy()
    
    if A.shape[1] != B.shape[0]:
        raise Exception("Matrices have the wrong shape.")
        
    Q = A.copy()
    Q.resize(A.shape[1],A.shape[1])
    Q = Q*0
    for i in range(A.shape[1]):
        Q[i,i] = 1

    numRows, numCols = A.shape

    i,j = 0,0
    while True:
        if i >= numRows or j >= numCols:
            break

        if A[i][j] == 0:
            nonzeroCol = j
            while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
                nonzeroCol += 1

            if nonzeroCol == numCols:
                i += 1
                continue
            col_swap(A, j, nonzeroCol)
            row_swap(B, j, nonzeroCol)
            col_swap(Q, j, nonzeroCol)

        pivot = A[i,j]
        col_scale(A, j, pivot**-1)
        row_scale(B, j, pivot**-1)
        col_scale(Q, j, pivot**-1)

        for otherCol in range(0, numCols):
            if otherCol == j:
                continue
            if A[i, otherCol] != 0:
                scaleAmt = -A[i, otherCol]
                col_combine(A, otherCol, j, scaleAmt)
                row_combine(B, j, otherCol, -scaleAmt)
                col_combine(Q, otherCol, j, scaleAmt)

        i += 1; j+= 1
        
    return A,B,Q
