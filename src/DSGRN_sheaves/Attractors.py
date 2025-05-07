### Attractors.py
### MIT LICENSE 2024 Alex Dowling

import itertools
import numpy as np
import pychomp
import DSGRN_utils

from .Sheaf import *

def build_morse_dict(parameter_complex, stg_dict, prune_grad='none'):
    dim = max(cell.dim for cell in parameter_complex.vertices())
    morse_dict = {}
    for cell in parameter_complex.vertices():
        stg = stg_dict[cell]
        (scc_dag, graded_comp) = pychomp.FlowGradedComplex(stg.complex(), 
                                                           stg.adjacencies())
        connection_matrix = pychomp.ConnectionMatrix(graded_comp)
        if prune_grad == 'all' or (prune_grad == 'some' and cell.dim == dim):
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_comp, 
                                                 connection_matrix)
        else:
            morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_comp, 
                                                 connection_matrix, False)
        morse_dict.update({cell : morse_graph})
    return morse_dict

def morse_nodes_from_section(shf, morse_dict, section):
    n = 0
    section_morse_nodes = {}
    for cell in shf.grading[0]:
        m = len(shf.stalk(cell))
        stalk_list = list(morse_dict[cell].vertices())
        stalk_nodes = [stalk_list[i] for i in range(m) if section[n+i] != 0]
        section_morse_nodes.update({cell : stalk_nodes})
        n = n + m
    return section_morse_nodes

def section_from_morse_nodes(shf, section_morse_sets):
    section = []
    for cell in shf.grading[0]:
        section = section + [int(M in section_morse_sets[cell]) 
                             for M in shf.stalk(cell)]
    return shf.GF(section)

def attractor_sections(shf, morse_dict, shf_cohomology=None):
    if shf_cohomology is None:
        shf_cohomology = sheaf_cohomology(shf)
    sections = shf_cohomology[0]
    att_secs = pychomp.DirectedAcyclicGraph()
    zero_sec = 0*sections[0]

    for c in itertools.product(*[[0, 1] for s in sections]):
        section = sum([cv*sec for cv, sec in zip(c, sections)], zero_sec)
        morse_nodes = morse_nodes_from_section(shf, morse_dict, section)
        for cell in shf.grading[0]:
            selected = morse_nodes[cell]
            P = pychomp.Poset(morse_dict[cell])
            if set(morse_nodes[cell]) != set(morse_nodes[cell]).union(
                                       *[set(P.descendants(m)) 
                                         for m in morse_nodes[cell]]):
                break
        else:
            att_secs.add_vertex(tuple([int(s==1) for s in section]))

    for s1, s2 in itertools.combinations(att_secs.vertices(), 2):
        if all([s1[i] <= s2[i] for i in range(len(s1))]):
            att_secs.add_edge(s2, s1)
            continue
        elif all([s2[i] <= s1[i] for i in range(len(s1))]):
            att_secs.add_edge(s1, s2)

    return pychomp.Poset(att_secs)