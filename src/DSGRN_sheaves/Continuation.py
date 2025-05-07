### Continuation.py
### MIT LICENSE 2024 Alex Dowling

import DSGRN
import DSGRN_utils
import pychomp
from itertools import product, combinations

from .Sheaf import *
from .Cohomology import *
from .CechCell import *

def blank_stg(parameter_graph):
    """Inputs a DSGRN parameter graph. 

        Outputs a CubicalBlowupGraph object with no adjacencies for its 
        multivalued map.
    """
    
    # FIX THIS: the initialization of stg builds the trivial multivalued map 
    # instead of the empty one
    parameter = parameter_graph.parameter(0)
    stg = DSGRN_utils.CubicalBlowupGraph(parameter=parameter, level=0)
    
    stg.digraph.graph_vertices = set(stg.blowup_complex(stg.dim))
    stg.digraph.adjacency_lists = {cell : set() 
                                   for cell in stg.blowup_complex(stg.dim)}
    return stg

def stg_union(stg_list, parameter_graph):
    """ Inputs a list of CubicalBlowupGraph objects with the same 
        sets of vertices. 
        
        Outputs a CubicalBlowupGraph object whose digraph is
        the union of those from the complexes of given list.
    """

    total_stg = blank_stg(parameter_graph)
    # For each vertex
    for v in total_stg.digraph.vertices():
        # get the adjacencies from ALL of the complexes
        all_v_adjacencies = [stg.digraph.adjacency_lists[v] 
                             for stg in stg_list]
        # and add them to the adjacencies in total_stg
        total_stg.digraph.adjacency_lists[v].update(*all_v_adjacencies)
    return total_stg

def build_parameter_complex(parameter_graph, parameter_indices = None, 
                            dim = None, length_cap = 2, level = 3):
    """ Inputs a DSGRN parameter graph, a collection of parameter indices, and
        a maximum dimension. 
        
        Outputs a directed acyclic graph of keys, and
        a dictionary of CubicalBlowupGraph objects.
    """
    
    # Initialize ingredients for parameter complex
    param_dim = parameter_graph.dimension()
    if parameter_indices is None: 
        parameter_indices = set(range(parameter_graph.size()))
    if dim is None: 
        dim = param_dim
    top_cells = {}
    parameter_complex = pychomp.DirectedAcyclicGraph()
    stg_dict = {}

    # Build the top dimensional cells and their state transition graphs first
    for index in parameter_indices:
        parameter = parameter_graph.parameter(index)
        cell = top_cech_cell(parameter_graph, index, dim)
        # Keep top_cells so we don't have to build them again later
        top_cells.update({index : cell})
        parameter_complex.add_vertex(cell)
        stg_dict.update({cell : DSGRN_utils.CubicalBlowupGraph(
                                parameter=parameter, level=level)})

    # Build the remaining cells from the top down (BFS ordering)
    for codim, index in product(range(1, dim+1), parameter_indices):
        top_cell = top_cells[index]
        adj_indices = list(set(parameter_indices).intersection(
                           parameter_graph.adjacencies(index, "codim1")))
        # If there aren't enough adjacent indices to coarsen with, 
        # stick in some copies of the top index
        adj_indices = adj_indices + max(0, dim - len(adj_indices) + 1)*[index]

        # Iterate through each combination of adjacent parameter indices to 
        # build the lower dimensional cells
        for combo in combinations(adj_indices, codim):
            # Get the CechCells corresponding to each index in the combination
            adj_cells = [top_cells[index] for index in combo]
            cells = adj_cells + [top_cell]
            # If the number of inequalities for a particular factor graph 
            # exceeds the length cap, skip this combination
            if (len(combo) >= length_cap and 
                any(len({cell[d] for cell in cells}) > length_cap
                    for d in range(param_dim))):
                continue
                
            # Build the CechCell corresponding to this combination
            cell_join = top_cell.join(adj_cells, dim - codim)
            # Each adjacent parameter's partial orders define a permutation 
            # from those at index to those at the adjacent index. Close the 
            # set of allowed partial orders under these permutations
            cell = cell_join.permute()
            parameter_complex.add_vertex(cell)
            
            # Get the CubicalBlowupGraph for all cells
            cell_stgs = [stg_dict[cell] for cell in cells]
            if cell in stg_dict:
                cell_stgs.append(stg_dict[cell])
            # and build a new CubicalBlowupGraph from their union
            stg_dict.update({cell : stg_union(cell_stgs, parameter_graph)})

            # Find all children of the cell
            child_cells = set(cell_join.get_child_cells())
            child_cells &= parameter_complex.vertices()
            for child_cell in child_cells:
                # and add an the corresponding order relations
                parameter_complex.add_edge(cell, child_cell.permute())
            # If it exists, add order relation for higher dimensional cell 
            # with the same inequalities
            expand_cell = CechCell(cell.inequality_sets, cell.dim + 1, 
                                   cell.labels)
            if expand_cell in stg_dict:
                parameter_complex.add_edge(cell, expand_cell)

    return parameter_complex, stg_dict

def morse_restriction(stalk_0, stalk_1):
    """ Inputs a two lists of morse sets. 
    
        Outputs the matrix which maps a morse set in stalk_0 to the 
        sum of morse sets in stalk_1 which are contained in it.
    """

    # Initialize matrix of zeroes
    R = [[0 for s0 in stalk_0] for s1 in stalk_1]
    # For each matrix entry, 
    for j, morse_0 in enumerate(stalk_0):      
        for i, morse_1 in enumerate(stalk_1):
            # if the morse set at index i in morse_1 is a subset of
            # the morse set at index j in morse_0,
            if morse_1.issubset(morse_0):
                # set the entry to 1
                R[i][j] = 1
    return R

def attractor_sheaf(parameter_complex, stg_dict, prune_grad='none'):
    """ Inputs a directed acyclic graph of keys and a dictionary of 
        RookRulesCubicalComplex objects with the same set of vertices. 
        
        Optional string argument prune_grad:
        - prune_grad = 'none' : never prune gradient directions
        - prune_grad = 'some' : prune gradient directions for top cells
        - prune_grad = 'all'  : prune all gradient directions
        
        Outputs the attractor sheaf on the corresponding poset.
        """
    
     # Initialize ingredients for a sheaf
    dim = max([cell.dim for cell in parameter_complex.vertices()])
    stalk_dict = {}
    restr_dict = {}
    grading = [set() for i in range(dim+1)]

    # 
    for cell, stg in stg_dict.items():
        # Build the connection matrix for the cell's state transition graph
        (scc_dag, graded_complex) = pychomp.FlowGradedComplex(stg.complex(), 
                                                              stg.digraph.adjacencies)
        connection_matrix = pychomp.ConnectionMatrix(graded_complex)
        # Prune gradient directions if optional prune_grad argument is supplied
        if prune_grad == 'all' or (prune_grad == 'some' and cell.dim == dim):
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                                 connection_matrix)
        else:
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                                 connection_matrix, False)
        # Load the list of Morse sets into the cell's stalk
        morse_sets = [frozenset({c for c in stg.digraph.vertices() 
                                 if graded_complex.value(c) == M}) 
                      for M in morse_graph.vertices()]
        stalk_dict.update({cell : morse_sets})
        # Specify the grading on the cell
        grading[cell.dim].add(cell)

    # For each incident pair of cells,
    for (cell_0, cell_1) in parameter_complex.edges():
        # build the restriction map
        R = morse_restriction(stalk_dict[cell_0], stalk_dict[cell_1])
        # and load it into restr_dict
        restr_dict.update({(cell_0, cell_1) : R})
         
    return Sheaf(pychomp.Poset(parameter_complex), grading, stalk_dict, 
                 restr_dict)
