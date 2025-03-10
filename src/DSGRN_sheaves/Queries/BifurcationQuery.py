### BifurcationQuery.py
### MIT LICENSE 2024 Alex Dowling

### Utilizes *FocusSearch subgraph isomorphism algorithm by Bonnici V, 
### Giugno R, Pulvirenti A, Shasha D, and Ferro A. published in “A subgraph 
### isomorphism algorithm and its application to biochemical data."

import DSGRN
import DSGRN_utils
from math import inf
from time import time
from sys import stdout

try:
    from multiprocess import Pool
    has_multiprocess = True
except ImportError:
    has_multiprocess = False

from ..Sheaf import *
from ..Cohomology import *
from ..Continuation import *

def greatest_constraint_first(G, grading):
    """ Greedy algorithm which returns an ordering on the vertices of a given 
        graph G, and a list of parent vertices. Will prioritize vertices with 
        higher grade in grading dictionary.
    """
    # See source* for description of algorithm

    n = len(G.vertices)
    V = G.vertices.copy()
    weights = list(grading.keys()).copy()
    weights.sort()
    
    current_w = weights[-1]
    ordering = [grading[current_w][0]]
    parents = [None]
    for v in grading[current_w]:
        if len(G.adjacencies(v)) > len(G.adjacencies(ordering[0])):
            ordering[0] = v
    V.remove(ordering[0])
    
    while V > set():
        if set(grading[current_w]) & V == set():
            weights.pop()
            current_w = weights[-1]
            continue
        m = len(ordering)
        u_next = None
        u_rank = (-inf, -inf, -inf)
        
        for u in set(grading[current_w]) & V:
            vis = {ordering[i] for i in range(m) 
                               if ordering[i] in G.adjacencies(u)}
            nei = {ordering[i] for i in range(m)
                               if (set(G.adjacencies(ordering[i]))
                                   & set(G.adjacencies(u))) > set()}
            unv = (V.difference(*[set(G.adjacencies(ordering[i])) 
                                  for i in range(m)])
                   & set(G.adjacencies(u)))
            if u_rank <= (len(vis), len(nei), len(unv)):
                u_next = u
                u_rank = (len(vis), len(nei), len(unv))
                
        u_parent = None
        for i in range(m):
            if u_next in G.adjacencies(ordering[i]):
                u_parent = ordering[i]
                break
        ordering.append(u_next)
        parents.append(u_parent)
        V.remove(u_next)
    
    return ordering, parents

class Node:
    def __init__(self, key):
        self.val = key
        self.children = []

def matching(parameter_graph, G, param_grading, match_grading, symmetry=False):
    """ Given a parameter graph and a match graph, finds all subgraphs of the 
        parameter graph isomorphic to the match graph. These isomorphisms must
        respect the grading dictionaries. 
    """
    # See source* for description of algorithm

    start = time()
    m = parameter_graph.size()
    n = len(G.vertices)
    ordering, parents = greatest_constraint_first(G, match_grading)
    G_deg = {i:len(G.adjacencies(ordering[i])) for i in range(len(ordering))}
    parameter_graph_deg = {i:len(parameter_graph.adjacencies(i, "codim1")) 
                           for i in range(m)}
    match_grade_dict = {v : max([grade for grade in match_grading.keys() 
                                 if v in match_grading[grade]]) 
                                 for v in G.vertices}
    root = Node(None)
    root.children = ([Node(i) 
                      for i in param_grading[match_grade_dict[ordering[0]]]])
    valid_paths = []

    def path_conditions(path):
        injective = path[-1] not in path[:-1]
        edge_cap = parameter_graph_deg[path[-1]] >= G_deg[len(path)-1]
        shape = all([path[-1] in parameter_graph.adjacencies(path[i], "codim1") 
                     for i in range(len(path)) 
                     if ordering[len(path)-1] in G.adjacencies(ordering[i])])
        return all([injective, edge_cap, shape])    
    
    def traverse_search_tree(root, depth, path):
        while root.children > []:
            next_node = root.children[0] 
            if path_conditions(path + [next_node.val]):
                new_path = path+[next_node.val]
                if depth < n - 2:
                    grade = param_grading[match_grade_dict[ordering[depth+2]]]
                    parent = parents[depth+2]
                    if parent is not None:
                        parent_match = new_path[ordering.index(parent)]
                        parent_nei = parameter_graph.adjacencies(
                                     parent_match, "codim1")
                    else:
                        parent_nei = range(m)
                    next_node.children = [Node(i) for i in parent_nei
                                          if i in grade 
                                          and i not in new_path]
                traverse_search_tree(next_node, depth+1, new_path)
            root.children.pop(0)
        if depth == n - 1:
            valid_paths.append(path)
    
    traverse_search_tree(root, -1, [])
    
    if not symmetry:
        valid_sets = {frozenset(path):path for path in valid_paths}
        valid_paths = list(valid_sets.values())

    stop = time()
    print(f"Graph matching took {stop-start:.2f} seconds. "\
          f"Found {len(valid_paths)} graph matches.")
    return valid_paths, ordering

def select_from_match(match, selection, ordering):
    return [match[i] for i in range(len(match)) if ordering[i][0] in selection]

class BifurcationQuery:
    """ Class for handling queries of parameter graph for bifurcations. 
    
        Initialize with:
        - parameter_graph : DSGRN parameter graph
        - vertices        : set of vertices of a "match" graph
        - edges           : set of edges of a "match" graph
        - param_grading   : a dictionary with integer grades as keys and 
                            sets of indices as values, e.g. parameter stability
        - match grading   : a dictionary with integer grades as keys and
                            sets of vertices in vertices as values
        - coho_criteria   : a list of dictionaries storing cohomology criteria
                            (see check cohomology for syntax)

        Once a query is initialized, use by running query.execute(). 
        The result is a list of lists of vertices. Each of these lists 
        satisfies:
        - the subgraph of parameter_graph defined by these indices is  
          isomorphic to the given "match" graph,
        - the grade of every vertex is preserved under this isomorphism,
        - the indices meet every cohomology criterion described in 
          coho_criteria.
        
        query.multi_execute() does the same, but leverages multiprocessing 
        from the multiprocess python package.
    """

    def assemble_grading(self, param_grading):
        # Compute a stability query for grading if specified
        if param_grading == "stability":
            self.param_grading = DSGRN_utils.StabilityQuery(
                                 self.parameter_graph.network())
        else:
            self.param_grading = param_grading
        param_list = list(range(self.parameter_graph.size()))
        # Ensure every parameter is graded at least -inf
        self.param_grading.update({-inf : [i for i in param_list 
                                   if all([i not in self.param_grading[grade] 
                                   for grade in self.param_grading.keys()])]})
        # Associate to each parameter the maximum grade it belongs to
        self.param_grade_dict = {i : max(
                                 [grade for grade in self.param_grading.keys() 
                                  if i in self.param_grading[grade]]) 
                                 for i in param_list}
   
    def assemble_graph(self, vertices, edges, match_grading):
        self.match_grading = match_grading
        # Ensure every vertex is graded at least -inf
        self.match_grading.update({-inf : v for v in vertices 
                                   if all([v not in self.match_grading[grade] 
                                   for grade in self.match_grading.keys()])}) 
        # Build a DSGRN.Graph object for the match graph
        E = set()
        for v,w in edges:
            E.add((v, w))
            E.add((w, v))   
        self.match_graph = DSGRN.Graph(vertices, E)
        
    def check_cohomology(self, match):
        """ Inputs a list of parameter indices that form a subgraph    
            isomorphic to the query's match graph. 

            Outputs the True/False if the match does/doesn't meet the query's
            cohomology criteria, and a list of the cohomology groups computed.
        """

        check = True
        cohomologies = []
        # Check each criteria
        for criteria in self.coho_criteria:
            # If the user supplies a custom criteria function, use that instead
            # of computing cohomology of a parameter complex
            custom = criteria.get('custom')
            if custom is not None:
                if custom(self.parameter_graph, match, self.ordering):
                    continue
                else:
                    check = False
                    break
            # Get settings from criteria
            selection = criteria.get('selection', 
                                     [v for v in self.match_graph.vertices])
            indices = [match[i] for i in range(len(match)) 
                                    if self.ordering[i] in selection]
            predicate = criteria.get('predicate', lambda sc : True)
            dim = criteria.get('dim', self.parameter_graph.dimension())
            length_cap = criteria.get('length_cap', 2)
            prune_grad = criteria.get('prune_grad', 'none')
            clean = criteria.get('clean_stalks', False)
            
            # Compute cohomology of parameter complex from configuration of 
            # settings
            parameter_complex, stg_dict = build_parameter_complex(
                                          self.parameter_graph, indices, dim,
                                          length_cap)
            shf = attractor_sheaf(parameter_complex, stg_dict, prune_grad)
            if clean:
                shf = clean_stalks(shf)
            shf_cohomology = sheaf_cohomology(shf)
            cohomologies.append(shf_cohomology)
            
            # Check whether or not the cohomology groups satisfy the predicate
            if not predicate(shf_cohomology):
                check = False
                break
        
        return check, cohomologies

    def find_shape_matches(self):
        self.shape_matches, self.ordering = matching(self.parameter_graph,
                                                     self.match_graph,
                                                     self.param_grading,
                                                     self.match_grading)

    def execute(self):
        """ Returns all subgraphs of the parameter graph which are isomorphic
            to the query's match graph, and satisfy the query's cohomology 
            conditions. """
        
        # Get the subgraphs of the parameter graph which have the same shape as
        # the match graph (called a shape match)
        if self.shape_matches is None:
            self.find_shape_matches()
        self.matches = []
        self.coho_list = []
        n = 0
        start = time()
        total = min(self.cap,len(self.shape_matches))
        # For each shape match, 
        for match in self.shape_matches:
            n = n+1
            
            stdout.write(f"\rEvaluating sheaf criteria. "
                         f"{n/total:.1%} complete.")
            
            # check if it satisfies the query's criteria
            check, cohomologies = self.check_cohomology(match)
            # If it does, add it to the list of matches
            if check:
                self.matches.append(match)
                self.coho_list.append(cohomologies)
            if n >= total:
                break
        stop = time()

        print(f"\nEvaluating sheaf criteria took {stop-start:.2f} seconds.")
        print(f"\nFound {len(self.matches)} matches!")
        return self.matches

    def multi_execute(self, processes=8):
        """ Input a number of parallel processes for Pool to use.
        
            Returns all subgraphs of the parameter graph which are isomorphic
            to the query's match graph, and satisfy the query's cohomology 
            conditions. 
            
            Similar to execute(), but uses parallel processing."""
        
        # If user doesn't have multiprocess, execute normally
        if not has_multiprocess:
            return self.execute()
        
        if self.shape_matches is None:
            self.shape_matches, self.ordering = matching(self.parameter_graph,
                                                         self.match_graph,
                                                         self.param_grading,
                                                         self.match_grading)
        self.matches = []
        self.coho_list = []
        start = time()
        total = min(self.cap,len(self.shape_matches))

        def check_match(match):
            result, coho = self.check_cohomology(match)
            return match, result, coho

        with Pool(processes=processes) as pool:
            results = pool.imap_unordered(check_match, 
                                          self.shape_matches[:total])
            for i, (match, result, coho) in enumerate(results):
                stdout.write(f"\rEvaluating sheaf criteria. "\
                             f"{i/total:.1%} complete.")
                if result:
                    self.matches.append(match)
                    self.coho_list.append(coho)

        stop = time()

        print(f"\nEvaluating sheaf criteria took {stop-start:.2f} seconds.")
        print(f"\nFound {len(self.matches)} matches!")
        return self.matches
        

    def __init__(self, parameter_graph, vertices, edges, param_grading={}, 
                 match_grading={}, coho_criteria=[], cap=inf, 
                 shape_matches=None, ordering=None):
        
            self.parameter_graph = parameter_graph
            self.assemble_grading(param_grading)
            self.assemble_graph(vertices, edges, match_grading)
            self.coho_criteria = coho_criteria
            self.cap = cap
            self.shape_matches = shape_matches
            self.ordering = ordering
