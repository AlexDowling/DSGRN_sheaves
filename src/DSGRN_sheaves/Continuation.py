### Continuation.py
### MIT LICENSE 2024 Alex Dowling

import DSGRN
import DSGRN_utils
import pychomp
import itertools

from Sheaf import *
from Cohomology import *

def stg_union(stg_list, level = 8):
    """ Inputs a list of RookRulesCubicalComplex objects with the same 
        sets of vertices. Outputs a RookRulesCubicalComplex whose digraph is
        the of those from the complexes of given list."""

    total_stg = DSGRN_utils.RookRulesCubicalComplex(stg_list[0].parameter,
                                                    level)
    for stg in stg_list:
        for edge in stg.digraph.edges():
            total_stg.digraph.add_edge(edge[0], edge[1])
    return total_stg

def inequality_key(parameter_graph, index):
    """ Inputs a DSGRN parameter graph and an index in that graph. Outputs a
        (hashable) key for the inequalities of that parameter region."""
   
    lines = parameter_graph.parameter(index).partialorders().split("\n")
    return tuple([frozenset({tuple(s[6:len(s)-1].split(", "))})
                  for s in lines])

def get_permutation(A, B):
    """ Inputs a pair of tuples with the same elements, possibly reordered.
        Outputs a function which reorders tuples in the same way the elements
        of A are reordered to produce B. """
    
    mapping = [B.index(a) for a in A]
    def permutation(C, mapping=mapping):
        return tuple(C[i] for i in mapping)
    return permutation

def permute_key(key):
    """ Inputs a key for an element of the parameter complex. Outputs a key 
        whose coordinates are closed under permutations defined by pairs of 
        inequalities from that coordinate. """
    
    ineqs_list = []
    for ineqs in key[:-1]:
        ineqs_list.append({ineq for ineq in ineqs})
        for pair in itertools.combinations(ineqs, 2):
            permutation = get_permutation(pair[0], pair[1])
            for ineq in ineqs:
                ineqs_list[-1].add(permutation(ineq))            
    return tuple([frozenset(ineqs) for ineqs in ineqs_list] + [key[-1]])    

def full_parameter_complex(parameter_graph, parameter_indices = None, 
                           dim = None, length_cap = 2, level = 8):
    """ Inputs a DSGRN parameter graph, a collection of parameter indices, and
        a maximum dimension. Outputs a directed acyclic graph of keys, and
        a dictionary of RookRulesCubicalComplex objects."""
    
    param_dim = parameter_graph.dimension()
    if parameter_indices is None: 
        parameter_indices = list(range(parameter_graph.size()))
    if dim is None: 
        dim = param_dim
    parameter_complex = pychomp.DirectedAcyclicGraph()
    stg_dict = {}
 
    for i in parameter_indices:
        parameter = parameter_graph.parameter(i)
        key = inequality_key(parameter_graph, i) + (dim,)
        stg_dict.update({key : DSGRN_utils.RookRulesCubicalComplex(
                               parameter, level)})
        parameter_complex.add_vertex(key)
        
    for codim in range(1,dim+1):
        for i in parameter_indices:
            adj = [j for j in parameter_graph.adjacencies(i, "codim1") 
                         if j in parameter_indices]
            adj.extend([i for j in range(max(0,dim-len(adj)+1))])
            
            batches = [(i,) + comb 
                       for comb in itertools.combinations(adj, codim)]          
            for batch in batches:

                param_keys = [inequality_key(parameter_graph, j) 
                              for j in batch]
                skip = False
                for k in range(param_dim):
                    if len({key[k] for key in param_keys}) > length_cap:
                        skip = True
                        break
                if skip: continue
           
                base_key = (tuple(frozenset({}).union(*[k[d] 
                                                for k in param_keys]) 
                                                for d in range(param_dim))
                            + (dim - codim,))
                if length_cap > 2:
                    key = permute_key(base_key)
                else:
                    key = base_key

                batch_stgs = [stg_dict[inequality_key(
                                              parameter_graph, j) + (dim,)] 
                              for j in batch]
                if key in stg_dict.keys(): 
                    batch_stgs.append(stg_dict[key])
                
                stg_dict.update({key : stg_union(batch_stgs)})
                parameter_complex.add_vertex(key)
                
                for d in range(param_dim):
                    if len(base_key[d])>1:
                        for ineq in base_key[d]:
                            adj_key = (base_key[:d] 
                                               + (base_key[d].difference(
                                                  {ineq}),)
                                               + base_key[d+1:param_dim] 
                                               + (dim - codim + 1,))
                            if length_cap > 2:
                                adj_key = permute_key(adj_key)
                            if adj_key in parameter_complex.vertices():
                                parameter_complex.add_edge(key, adj_key)
                    adj_key = (base_key[:param_dim] + (dim - codim + 1,))
                    if length_cap > 2:
                                adj_key = permute_key(adj_key)
                    if adj_key in parameter_complex.vertices():
                                parameter_complex.add_edge(key, adj_key)

    return parameter_complex, stg_dict

def attractor_sheaf(parameter_complex, stg_dict, prune_grad='none'):
    """ Inputs a directed acyclic graph of keys and a dictionary of 
        RookRulesCubicalComplex objects with the same set of vertices. Outputs 
        the attractor sheaf on the corresponding poset."""
 
    dim = max([key[-1] for key in parameter_complex.vertices()])
    stalk_dict = {}
    restr_dict = {}
    grading = [set() for i in range(dim+1)]
    
    for key in parameter_complex.vertices():
        stg = stg_dict[key]
        (scc_dag, graded_complex) = pychomp.FlowGradedComplex(stg.complex(), 
                                                              stg.diagram())
        connection_matrix = pychomp.ConnectionMatrix(graded_complex)
        if prune_grad == 'all' or (prune_grad == 'some' and key[-1] == dim):
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                                 connection_matrix)
        else:
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                                 connection_matrix, False)
        stalk_dict.update({key : [frozenset({c for c in stg.digraph.vertices() 
                                             if graded_complex.value(c) == v}) 
                                  for v in morse_graph.vertices()]})
        grading[key[-1]].add(key)

    for (key_0, key_1) in parameter_complex.edges():
        R = [[0 for j in range(len(stalk_dict[key_0]))] 
                for i in range(len(stalk_dict[key_1]))]
        for j in range(len(stalk_dict[key_0])):      
            for i in range(len(stalk_dict[key_1])):
                if stalk_dict[key_1][i].issubset(stalk_dict[key_0][j]):
                    R[i][j] = 1
        restr_dict.update({(key_0, key_1) : R})
         
    return Sheaf(pychomp.Poset(parameter_complex), grading, stalk_dict, 
                 restr_dict)
