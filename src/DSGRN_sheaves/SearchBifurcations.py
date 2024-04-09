### SearchBifurcations.py
### MIT LICENSE 2024 Alex Dowling

### Utilizes FocusSearch subgraph isomorphism algorithm by Bonnici V, 
### Giugno R, Pulvirenti A, Shasha D, and Ferro A. published in “A subgraph 
### isomorphism algorithm and its application to biochemical data."

import DSGRN
import DSGRN_utils
import math
import time
import sys

from DSGRN_sheaves.Sheaf import *
from DSGRN_sheaves.Cohomology import *
from DSGRN_sheaves.Continuation import *

def greatest_constraint_first(G, grading):
    """ Greedy algorithm which returns an ordering on the vertices of a given 
        graph G, and a list of parent vertices. Will prioritize vertices with 
        higher grade in grading dictionary."""
    
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
        u_rank = (-math.inf, -math.inf, -math.inf)
        
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
        respect the grading dictionaries. """
    
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
    
    start = time.time()
    traverse_search_tree(root, -1, [])
    end = time.time()
    
    if not symmetry:
        smashed = []
        for path in valid_paths:
            if all([set(path) != set(p) for p in smashed]):
                smashed.append(path)
        print("Graph matching took",str(round(end-start,2)),"seconds. Found",
              str(len(smashed)),"graph matches.")
        return smashed, ordering
    print("Graph matching took",str(round(end-start,2)),"seconds. Found ",
              str(len(valid_paths))," graph matches.")
    return valid_paths, ordering

def select_from_match(match, selection, ordering):
    return [match[i] for i in range(len(match)) if ordering[i][0] in selection]

class BifurcationQuery:
    """ Data type for storing queries of parameter graph for bifurcations. """
    
    def assemble_grading(self, param_grading):
        if param_grading == "stability":
            self.param_grading = DSGRN_utils.StabilityQuery(
                                 self.parameter_graph.network())
        else:
            self.param_grading = param_grading
        param_list = list(range(self.parameter_graph.size()))
        self.param_grading.update({-math.inf : [i for i in param_list 
                                   if all([i not in self.param_grading[grade] 
                                   for grade in self.param_grading.keys()])]})
        self.param_grade_dict = {i : max(
                                 [grade for grade in self.param_grading.keys() 
                                  if i in self.param_grading[grade]]) 
                                 for i in param_list}
   
    def assemble_graph(self, vertices, edges, match_grading):
        self.match_grading = match_grading
        self.match_grading.update({-math.inf : v for v in vertices 
                                   if all([v not in self.match_grading[grade] 
                                   for grade in self.match_grading.keys()])}) 
        E = set()
        for v,w in edges:
            E.add((v, w))
            E.add((w, v))   
        self.match_graph = DSGRN.Graph(vertices, E)
        
    def check_cohomology(self, match, ordering):
        check = True
        cohomologies = []
        for criteria in self.coho_criteria:
            selection = criteria.get('selection', 
                                     [v[0] for v in self.match_graph.vertices])
            indices = [match[i] for i in range(len(match)) 
                                    if ordering[i][0] in selection]
            predicate = criteria.get('predicate', lambda sc : True)
            dim = criteria.get('dim', self.parameter_graph.dimension())
            length_cap = criteria.get('length_cap', 2)
            prune_grad = criteria.get('prune_grad', 'none')
            clean = criteria.get('clean_stalks', False)
               
            parameter_complex, stg_dict = full_parameter_complex(
                                          self.parameter_graph, indices, dim,
                                          length_cap)
            shf = attractor_sheaf(parameter_complex, stg_dict, prune_grad)
            if clean:
                shf = clean_stalks(shf)
            shf_cohomology = sheaf_cohomology(shf)
            cohomologies.append(shf_cohomology)
            
            if not predicate(shf_cohomology):
                check = False
                break
        
        return check, cohomologies

    def execute(self):
        """ Returns all subgraphs of the parameter graph which are isomorphic 
            to the query's match graph, and satisfy the query's cohomology 
            conditions. """
        
        shape_matches, ordering = matching(self.parameter_graph,
                                           self.match_graph,
                                           self.param_grading,
                                           self.match_grading)
        matches = []
        coho_list = []
        n = 0
        for match in shape_matches:
            n = n+1
            txt = (("\rBuilding sheaf on graph match {current} of {total}." 
                    + "   {percent}% complete.")
                   .format(current = n, total = len(shape_matches), 
                           percent = math.floor(n/len(shape_matches)*100)))
            sys.stdout.write(txt)
            sys.stdout.flush()
            
            check, cohomologies = self.check_cohomology(match, ordering)
            if check:
                matches.append(match)
                coho_list.append(cohomologies)
    
        print("\nFound "+str(len(matches))+" matches!")
        return matches, ordering, coho_list

    def __init__(self, parameter_graph, vertices, edges, param_grading = {},
                 match_grading = {}, coho_criteria = []):
            self.parameter_graph = parameter_graph
            self.assemble_grading(param_grading)
            self.assemble_graph(vertices, edges, match_grading)
            self.coho_criteria = coho_criteria
