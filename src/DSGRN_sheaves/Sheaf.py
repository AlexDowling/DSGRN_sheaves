### Sheaf.py
### MIT LICENSE 2024 Alex Dowling

import galois
import numpy as np
import pychomp

from DSGRN_sheaves.Cohomology import *

class Sheaf:
    """ Class for storing the stalks and restriction maps of a sheaf. Currently 
        only supports vector spaces over a finite field of two elements. """
    
    def __init__(self, P, grading, stalk_dict, restr_dict):
        self.P = P
        self.grading = grading

        self.GF = galois.GF(2)
        self.stalk_dict = stalk_dict
        self.restr_dict = {}
        for (p1, p2) in restr_dict.keys():
            self.restr_dict.update({(p1, p2) : self.GF(restr_dict[(p1, p2)])})          
          
    def stalk(self, p):
        """ Inputs an element p of the sheaf's underlying poset. Outputs a 
            basis for the stalk at p. """

        return self.stalk_dict[p]
    
    def restriction(self, p1, p2, f = None):
        """ Inputs a pair of incident cells p1 and p2, and an optional element 
            to evaluate. Outputs the matrix for the stalk restriction map, or
            the stalk restriction map evaluated at the inputted element. """
        
        if p1 not in self.P.parents(p2):
            return None

        R = self.restr_dict[(p1, p2)]
        if f is None:
            return R
        elif self.GF(f).shape[0] != R.shape[1]:
            return None
        else:
            return np.matmul(R, self.GF(f))

def sheaf_cohomology(shf, return_delta=False):
    """ Inputs a cellular sheaf. Outputs a list of lists of generators for each
        sheaf cohomology group. May be given an optional argument 
        to return the boundary operators return_delta"""
    
    dim = len(shf.grading) - 1
    cochainRank_1 = sum([len(shf.stalk(cell)) for cell in shf.grading[0]])
    cochainRank_2 = sum([len(shf.stalk(cell)) for cell in shf.grading[dim]])
    delta = {-1 : shf.GF([[0] for i in range(cochainRank_1)])} 
    delta.update({dim : shf.GF([[0 for i in range(cochainRank_2)]])}) 
    
    for k in range(dim):
        cochainRank_1 = sum([len(shf.stalk(cell)) 
                             for cell in shf.grading[k]])
        cochainRank_2 = sum([len(shf.stalk(cell)) 
                             for cell in shf.grading[k+1]])     
        D = shf.GF([[0 for i in range(cochainRank_1)] 
                    for j in range(cochainRank_2)])   
        col = 0
        for cell_1 in shf.grading[k]:
            row = 0
            n = len(shf.stalk(cell_1))
            for cell_2 in shf.grading[k+1]:
                m = len(shf.stalk(cell_2))
                if cell_1 in shf.P.parents(cell_2):
                    D[row:row+m,col:col+n] = shf.restriction(cell_1, cell_2)
                row = row + m
            col = col + n
        delta.update({k : D})
        k += 1
    
    if return_delta:
        return [cohomology(delta[j],delta[j-1]) for j in range(dim+1)], delta
    return [cohomology(delta[j],delta[j-1]) for j in range(dim+1)]

def clean_stalks(shf):
    """ Given a sheaf, returns a sheaf with the same stalks at top dimensional
        cells, but strips stalks at lower dimensional cells of elements which 
        map to zero under every restriction map. """
    
    dim = len(shf.grading) - 1
    P = pychomp.Poset(shf.P.children_.clone())
    grading = shf.grading.copy()
    stalk_dict = {cell : shf.stalk(cell) for cell in grading[dim]}
    restr_dict = {}
    keep = {cell : list(range(len(stalk))) 
            for cell, stalk in stalk_dict.items()}
    
    for codim in range(1,dim+1):
        for cell_1 in grading[dim - codim]:
            keep[cell_1] = []
            n = len(shf.stalk(cell_1))
            for cell_2 in P.children(cell_1):
                R = shf.restr_dict[(cell_1, cell_2)]
                for j in range(n):
                    if j in keep[cell_1]:
                        continue
                    if any([R[i][j] == 1 for i in keep[cell_2]]):
                        keep[cell_1].append(j)
            keep[cell_1].sort()
            
            stalk_dict[cell_1] = [shf.stalk(cell_1)[i] for i in keep[cell_1]]
            for cell_2 in P.children(cell_1):
                R = shf.restr_dict[(cell_1, cell_2)]
                restr_dict[(cell_1, cell_2)] = [[R[i][j] for j in keep[cell_1]]
                                                 for i in keep[cell_2]]
            
    return Sheaf(P, grading, stalk_dict, restr_dict)
