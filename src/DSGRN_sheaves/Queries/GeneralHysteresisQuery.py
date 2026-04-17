import DSGRN
import DSGRN_utils
import numpy as np
import galois
from numpy.linalg import matrix_rank

from ..Sheaf import *
from ..Cohomology import *
from ..Continuation import *
from ..Attractors import *
from .BifurcationQuery import *
from ..CechCell import *

class GeneralHysteresisQuery(BifurcationQuery):

    def build_grading(self, param_stability):
        if param_stability is None:
            param_stability = DSGRN_utils.StabilityQuery(
                                          self.parameter_graph.network())
        num_indices = self.parameter_graph.size()
        self.param_grading = {-1 : [i for i in range(num_indices)]}
        self.param_grading.update({1 : []})
        for key in param_stability.keys():
            if key > 1:
                self.param_grading[1] = (self.param_grading[1] 
                                         + list(param_stability[key]))

    def build_sheaf_data(self, indices):
        pc, stg_dict = build_parameter_complex(self.parameter_graph, 
                                               indices, 1)
        if len(indices) < 2:
            top_cell = top_cech_cell(self.parameter_graph, indices[0], 1)
            pc.add_edge(self.dummy, top_cell)
            stg_dict.update({self.dummy : stg_dict[top_cell]})
        shf = attractor_sheaf(pc, stg_dict)
        shf_cohomology = sheaf_cohomology(shf)
        rank = sum([len(shf.stalk(key)) for key in shf.grading[0]])
        return pc, stg_dict, shf, shf_cohomology, rank

    def build_slices(self, shf):
        row_slices = {}
        row = 0
        for cell in shf.grading[0]:
            row_slices.update({cell : slice(row, row+len(shf.stalk(cell)))})
            row = row + len(shf.stalk(cell))
        return row_slices

    def build_total_restriction(self, sheaf_data, c_sheaf_data, row_slices, 
                                l_edge_cell, r_edge_cell):
        pc, stg_dict, shf, shf_cohomology, rank = sheaf_data
        c_pc, c_stg_dict, c_shf, c_shf_cohomology, c_rank = c_sheaf_data
        
        R_tc = shf.GF([[0 for j in range(rank)] for i in range(c_rank)])
            
        col = 0
        for cell in shf.grading[0]:
            if len(shf.P.children(cell)) < 2:
                pass            
            elif l_edge_cell in shf.P.children(cell):
                c_edge_cell = next(c for c in shf.P.children(cell) 
                                   if c != l_edge_cell)
                target_cell = CechCell(c_edge_cell.inequality_sets, 0, 
                                       c_edge_cell.labels)
                R = morse_restriction(shf.stalk(cell), 
                                      c_shf.stalk(target_cell))
                col_slice = slice(col, col+len(shf.stalk(cell)))
                R_tc[row_slices[target_cell], col_slice] = R
            elif r_edge_cell in shf.P.children(cell):
                c_edge_cell = next(c for c in shf.P.children(cell) 
                                   if c != r_edge_cell)
                target_cell = CechCell(c_edge_cell.inequality_sets, 0, 
                                       c_edge_cell.labels)
                if self.length == 3:
                    target_cell = self.dummy
                R = morse_restriction(shf.stalk(cell), 
                                      c_shf.stalk(target_cell))
                col_slice = slice(col, col+len(shf.stalk(cell)))
                R_tc[row_slices[target_cell], col_slice] = R
            else:
                target_cell = cell
                R = shf.GF(np.eye(len(shf.stalk(cell))).astype(int))
                col_slice = slice(col, col+len(shf.stalk(cell)))
                R_tc[row_slices[target_cell], col_slice] = R
            col = col + len(shf.stalk(cell))

        return R_tc

    def build_left_restriction(self, l_sheaf_data, c_sheaf_data, row_slices, 
                               l_edge_cell):
        l_pc, l_stg_dict, l_shf, l_shf_cohomology, l_rank = l_sheaf_data
        c_pc, c_stg_dict, c_shf, c_shf_cohomology, c_rank = c_sheaf_data
        
        R_lc = l_shf.GF([[0 for j in range(l_rank)] for i in range(c_rank)])
            
        col = 0
        for cell in l_shf.grading[0]:
            if (l_edge_cell in l_shf.P.children(cell) 
                and len(l_shf.P.children(cell)) > 1):
                c_edge_cell = next(c for c in l_shf.P.children(cell) 
                                   if c != l_edge_cell)
                target_cell = CechCell(c_edge_cell.inequality_sets, 0, 
                                       c_edge_cell.labels)
                R = morse_restriction(l_shf.stalk(cell), 
                                      c_shf.stalk(target_cell))
                col_slice = slice(col, col+len(l_shf.stalk(cell)))
                R_lc[row_slices[target_cell], col_slice] = R
            elif l_edge_cell not in l_shf.P.children(cell):
                target_cell = cell
                if self.length == 3:
                    target_cell = self.dummy
                R = morse_restriction(l_shf.stalk(cell), 
                                      c_shf.stalk(target_cell))
                col_slice = slice(col, col+len(l_shf.stalk(cell)))
                R_lc[row_slices[target_cell], col_slice] = R
            col = col + len(l_shf.stalk(cell))

        return R_lc

    def build_right_restriction(self, r_sheaf_data, c_sheaf_data, row_slices, 
                                r_edge_cell):
        r_pc, r_stg_dict, r_shf, r_shf_cohomology, r_rank = r_sheaf_data
        c_pc, c_stg_dict, c_shf, c_shf_cohomology, c_rank = c_sheaf_data
        
        R_rc = r_shf.GF([[0 for j in range(r_rank)] for i in range(c_rank)])
        col = 0
        for cell in r_shf.grading[0]:
            if (r_edge_cell in r_shf.P.children(cell) 
                and len(r_shf.P.children(cell))) > 1:
                c_edge_cell = next(c for c in r_shf.P.children(cell) 
                                   if c != r_edge_cell)
                target_cell = CechCell(c_edge_cell.inequality_sets, 0, 
                                       c_edge_cell.labels)
                if self.length == 3:
                    target_cell = self.dummy
                R = morse_restriction(r_shf.stalk(cell), 
                                      c_shf.stalk(target_cell))
                col_slice = slice(col, col+len(r_shf.stalk(cell)))
                R_rc[row_slices[target_cell], col_slice] = R
            elif r_edge_cell not in r_shf.P.children(cell):
                target_cell = cell
                R = r_shf.GF(np.eye(len(r_shf.stalk(cell))).astype(int))
                col_slice = slice(col, col+len(r_shf.stalk(cell)))
                R_rc[row_slices[target_cell], col_slice] = R
            col = col + len(r_shf.stalk(cell))
        return R_rc

    def in_img(self, M, v):
            A = np.concatenate((M, v), axis=1)
            return matrix_rank(M) == matrix_rank(A)

    def check_section(self, section, att_secs, c_att_secs, 
                            R_tc, R_lc, R_rc, K_l, K_r):
        # TODO: make this match with Definition 4.3
        # make sure the global section has a unique immediate predecessor
        if len(att_secs.children(section)) != 1:
            return False
        # that section should be the zero section
        zero = list(att_secs.children(section))[0]
        if any(a!=0 for a in zero):
            return False
        # restrict the global section to the center
        # really that's happening in the `np.matmul` call, some formatting
        # is happening with `tuple`
        c_section = tuple([int(s==1) 
                           for s in np.matmul(R_tc, galois.GF2(section))])
        # this section should have a unique immediate predecessor
        if len(c_att_secs.children(c_section)) != 1:
            return False
        # that predecessor should have two predecessors of its own
        pred = list(c_att_secs.children(c_section))[0]
        if len(c_att_secs.children(pred)) != 2:
            return False

        # pick out each of the two predecessors from before
        s0 = galois.GF2([[a] for a in list(c_att_secs.children(pred))[0]])
        s1 = galois.GF2([[a] for a in list(c_att_secs.children(pred))[1]])     
        # determine the image of the sections from the left side into the center
        M_lc = np.matmul(R_lc, K_l)
        # determine the image of the sections from the right side into the center
        M_rc = np.matmul(R_rc, K_r)

        # don't know yet whether or not `s0` or `s1` is on the left or right!
        # `hyz01`: `s0` is on the left, `s1` is on the right
        # `s0` is inaccessible from the right, `s1` is inaccessible from the left
        hys01 = (self.in_img(M_lc, s0) and self.in_img(M_rc, s1) 
                 and not self.in_img(M_lc, s1) and not self.in_img(M_rc, s0))
        # `hyz02`: `s1` is on the left, `s0` is on the right
        # `s1` is inaccessible from the right, `s0` is inaccessible from the left
        hys10 = (self.in_img(M_lc, s1) and self.in_img(M_rc, s0) 
                 and not self.in_img(M_lc, s0) and not self.in_img(M_rc, s1))
        return hys01 or hys10
    
    def general_hysteresis(self, pg, match, ordering):
        # general_hysteresis
        #    pg: Parameter graph
        #    match: a particular path in parameter graph

        #    match is [*at least bistable nodes*, *at least monostable*, *at least monostable node*]
        c_match = match[:-2] # first n-2 are all at least bistable
        l_match = match[:-1] # also include the first "monostable" in `match`
        r_match = match[:-2] + [match[-1]]  # also include the last "monostable" in `match`
        
        l_edge_cell = top_cech_cell(self.parameter_graph, match[-2], 1)
        r_edge_cell = top_cech_cell(self.parameter_graph, match[-1], 1)

        # build the sheaf on the whole path
        sheaf_data = self.build_sheaf_data(match)
        pc, stg_dict, shf, shf_cohomology, rank = sheaf_data
        morse_dict = build_morse_dict(pc, stg_dict)
        # get sections of attractor sheaf which correspond to attractors at each cell
        att_secs = attractor_sections(shf, morse_dict)

        c_sheaf_data = self.build_sheaf_data(c_match)
        c_pc, c_stg_dict, c_shf, c_shf_cohomology, c_rank = c_sheaf_data
        c_morse_dict = build_morse_dict(c_pc, c_stg_dict)
        # get sections of attractor sheaf which correspond to attractors at each cell
        c_att_secs = attractor_sections(c_shf, c_morse_dict)

        l_sheaf_data = self.build_sheaf_data(l_match)
        l_pc, l_stg_dict, l_shf, l_shf_cohomology, l_rank = l_sheaf_data
        K_l = l_shf.GF(l_shf_cohomology[0]).transpose()

        r_sheaf_data = self.build_sheaf_data(r_match)
        r_pc, r_stg_dict, r_shf, r_shf_cohomology, r_rank = r_sheaf_data
        K_r = r_shf.GF(r_shf_cohomology[0]).transpose()  

        row_slices = self.build_slices(c_shf)

        R_tc = self.build_total_restriction(sheaf_data, c_sheaf_data, 
                                            row_slices, l_edge_cell, r_edge_cell)
        R_lc = self.build_left_restriction(l_sheaf_data, c_sheaf_data,
                                           row_slices, l_edge_cell)
        R_rc = self.build_right_restriction(r_sheaf_data, c_sheaf_data,
                                            row_slices, r_edge_cell)

        # check each global section to see if there is hysteresis inside it
        return any(self.check_section(section, att_secs, c_att_secs, R_tc, 
                                      R_lc, R_rc, K_l, K_r) 
                                      for section in att_secs)

    def __init__(self, parameter_graph, length, param_stability=None):
        if length < 3:
            raise ValueError("Length must be greater than 3.")
        self.parameter_graph = parameter_graph
        self.length = length
        self.build_grading(param_stability)
        self.dummy = CechCell(tuple(frozenset({('dummy',)})), 0)
        
        vertices = list(range(length))
        edges = [(i, i+1) for i in vertices[:-1]]
        match_grading = {-1 : [0, length-1], 1 : vertices[1:-1]}
        coho_criteria = [{"custom" : self.general_hysteresis}]

        super().__init__(parameter_graph, vertices, edges, 
                         self.param_grading, match_grading, coho_criteria)
