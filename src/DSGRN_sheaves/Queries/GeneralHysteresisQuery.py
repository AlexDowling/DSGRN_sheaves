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
        c_match = match[:-2]
        l_edge_index = match[-2]
        r_edge_index = match[-1]

        # 1. Build the center sheaf and its attractor-section poset ONCE.
        #    This is the one expensive (2^C) enumeration we can't avoid,
        #    and it's shared between the left and right searches.
        c_sheaf_data = self.build_sheaf_data(c_match)
        c_pc, c_stg_dict, c_shf, c_shf_cohomology, c_rank = c_sheaf_data
        c_morse_dict = build_morse_dict(c_pc, c_stg_dict)
        c_att_secs = attractor_sections(c_shf, c_morse_dict)

        # 2. Cheap linear scan of c_att_secs for candidate fold triples:
        #    pred with exactly one parent and exactly two children s0, s1,
        #    where s0 | s1 == pred is checked explicitly (closes the gap
        #    in the current check_section, which only checked cover-count).
        fold_candidates = self.find_fold_candidates(c_att_secs)
        if not fold_candidates:
            return False

        # 3. For each candidate, try to extend s0/s1 to an atom of the left/right
        #    poset using ONLY the one new endpoint cell -- not a full
        #    attractor_sections search over the whole left/right sub-path.
        for (pred, s0, s1, c_s) in fold_candidates:
            for (left_val, right_val) in [(s0, s1), (s1, s0)]:
                a = self.extend_atom_at_endpoint(c_shf, c_match, left_val,
                                                match, l_edge_index)
                b = self.extend_atom_at_endpoint(c_shf, c_match, right_val,
                                                match, r_edge_index)
                if a is None or b is None:
                    continue

                # 4. Confirm a genuine global section s exists on the whole path,
                #    consistent with a, c_s, b, and itself minimal (its own unique
                #    predecessor on Z_P is the zero section).
                if self.confirm_global_section(a, c_s, b, match, c_match,
                                            l_edge_index, r_edge_index):
                    return True

        return False

    def find_fold_candidates(self, c_att_secs):
        """ Scan the center attractor-section poset for candidate folds.

            A "fold" is a triple (pred, s0, s1) satisfying the folding 
            condition of Definition 4.3: pred is the unique immediate 
            predecessor of some c_s in the poset, and s0 | s1 == pred for 
            some pair of sections below pred. s0, s1 are candidates for
            a|center, b|center; c_s is the candidate for s|center.

            We search over ALL pairs of sections below pred (not just its 
            immediate children/covers), since a|center or b|center need 
            only be SOME section below pred whose join reaches it -- not 
            necessarily a cover of pred. A cover of pred might fail to 
            extend to a genuine atom on its own side (Step 3) while one of 
            its own descendants succeeds, so restricting to covers here 
            could silently drop valid folds.

            Returns a  list of (pred, s0, s1, c_s) tuples.
        """

        candidates = []
        for pred in c_att_secs.vertices():
            # pred must be the unique immediate predecessor of a well
            # defined c_s -- i.e. c_s's ONLY child is pred, not just one
            # of several. Checking both directions (pred has one parent,
            # and that parent has only pred as a child) rules out cases
            # where pred has a single parent that also covers some other,
            # unrelated element.
            parents = list(c_att_secs.parents(pred))
            if len(parents) != 1:
                continue
            c_s = parents[0]
            if len(c_att_secs.children(c_s)) != 1:
                continue

            # Cheap pre-filter before paying for the full down-set: if
            # pred covers only a single element c, every section strictly
            # below pred is forced to be <= c (any maximal chain down from
            # pred passes through its one and only cover), so no pair of
            # strictly-below sections can join back up past c to reach
            # pred. Skip without ever calling descendants().
            if len(c_att_secs.children(pred)) < 2:
                continue

            # Every section strictly below pred is a candidate for s0/s1 --
            # not just pred's immediate children. `descendants` gives the
            # full down-set (pychomp uses strict inequality, so pred itself
            # is excluded).
            below = list(c_att_secs.descendants(pred))

            # Check every pair for the folding equality directly -- don't
            # assume any structural shortcut (e.g. cover count) implies it.
            for s0, s1 in itertools.combinations(below, 2):
                joined = tuple(int(x or y) for x, y in zip(s0, s1))
                if joined == pred:
                    candidates.append((pred, s0, s1, c_s))

        return candidates

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
