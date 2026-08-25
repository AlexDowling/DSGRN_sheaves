import DSGRN
import DSGRN_utils
import numpy as np
import galois
import pychomp
from numpy.linalg import matrix_rank
from dataclasses import dataclass
from typing import Any

from ..Sheaf import *
from ..Cohomology import *
from ..Continuation import *
from ..Attractors import *
from .BifurcationQuery import *
from ..CechCell import *


# Let $`P = \{\zeta_i\}_{i=1}^k`$ be a path in the parameter graph
# $`\mathsf{PG}(RN)`$. The path is said to contain an *algebraic
# switch bifurcation* if there exists a global section
# $`s = \{s_\eta\}_{\eta\in \mathcal{Z}_P} \in
# \Gamma(\mathcal{Z}_P;\mathcal{S}^\mathsf{Att})`$ and two sections
# $`a \in \Gamma(\mathcal{Z}_P^\textnormal{left}; \mathcal{S}^\mathsf{Att})`$
# and $`b \in \Gamma(\mathcal{Z}_P^\textnormal{right};
# \mathcal{S}^\mathsf{Att})`$ satisfying:
#
# 1. **(minimality)** The global section $`s`$ has a unique immediate
#    predecessor in $`\Gamma(\mathcal{Z}_P; \mathcal{S}^\mathsf{Att})`$
#    which assigns to each cell $`\eta`$ the empty set. Likewise, the
#    sections $`a, b`$ have unique immediate predecessors in
#    $`\Gamma(\mathcal{Z}_P^\textnormal{left}; \mathcal{S}^\mathsf{Att})`$
#    and $`\Gamma(\mathcal{Z}_P^\textnormal{right}; \mathcal{S}^\mathsf{Att})`$
#    respectively, where each assigns the empty set at every cell it
#    is defined.
# 2. **(ordering)** $`a < s|_{\mathcal{Z}_P^\textnormal{left}}`$ and
#    likewise $`b < s|_{\mathcal{Z}_P^\textnormal{right}}`$.
# 3. **(folding)** $`a|_{\mathcal{Z}_P^\textnormal{center}} \vee
#    b|_{\mathcal{Z}_P^\textnormal{center}}`$ is the unique immediate
#    predecessor of $`s|_{\mathcal{Z}_P^\textnormal{center}}`$ in
#    $`\Gamma(\mathcal{Z}_P^\textnormal{center}; \mathcal{S}^\mathsf{Att})`$.

@dataclass
class HysteresisSheafData:
    #   the Cech poset built over the path's parameter indices
    parameter_complex: Any
    #   per-cell DSGRN state transition graphs, keyed by Cech cell
    state_transition_graphs: dict
    #   attractor sheaf built over the parameter complex
    sheaf: Sheaf
    #   cohomology of the attractor sheaf, one list of generators per degree
    sheaf_cohomology: list
    #   total dimension of the sheaf, summed over all stalks
    rank: int
    #   Morse graph for each cell's state transition graph
    morse_dict: dict
    #   poset of attractor sections of the sheaf
    attractor_sections: pychomp.Poset

class GeneralHysteresisQuery(BifurcationQuery):

    def build_grading(self, param_stability):
        #   build the parameter grading used to match paths: grade -1
        #   accepts any node, grade 1 requires multiple stable states.
        #   `param_stability` is an optional precomputed StabilityQuery

        #   fall back to computing stability directly if not supplied
        if param_stability is None:
            param_stability = DSGRN_utils.StabilityQuery(
                self.parameter_graph.network()
            )
        num_indices = self.parameter_graph.size()
        #   grade -1 matches any parameter node
        self.param_grading = {-1 : [i for i in range(num_indices)]}
        #   grade 1 matches only parameter nodes with multiple stable states
        self.param_grading.update({1 : []})
        for key in param_stability.keys():
            #   `key` counts stable equilibria; keep the multistable ones
            if key > 1:
                self.param_grading[1] = (
                    self.param_grading[1] + list(param_stability[key])
                )

    def build_sheaf_data(self, indices):
        #   build the attractor sheaf (and everything derived from it)
        #   over the sub-path given by `indices`, a list of parameter
        #   node indices

        #   build the parameter complex and its state transition graphs
        #   over the given path indices
        parameter_complex, state_transition_graphs = build_parameter_complex(
            self.parameter_graph, indices, 1
        )
        if len(indices) < 2:
            #   a single-vertex path has no edge to hang a sheaf on, so
            #   graft on a dummy 0-cell pointing at its own top cell
            top_cell = top_cech_cell(self.parameter_graph, indices[0], 1)
            parameter_complex.add_edge(self.dummy, top_cell)
            state_transition_graphs.update(
                {self.dummy : state_transition_graphs[top_cell]}
            )
        #   build the attractor sheaf over the complex and compute its
        #   cohomology, which is what witnesses a hysteresis loop
        sheaf = attractor_sheaf(
            parameter_complex, state_transition_graphs, 'some'
        )
        cohomology = sheaf_cohomology(sheaf)
        #   total dimension of the sheaf, summed over every cell's stalk
        rank = sum([len(sheaf.stalk(key)) for key in sheaf.grading[0]])
        #   the Morse graph and attractor-section poset are what
        #   `general_hysteresis` walks to look for candidate folds
        morse_dict = build_morse_dict(
            parameter_complex, state_transition_graphs
        )
        attractor_secs = attractor_sections(sheaf, morse_dict)
        return HysteresisSheafData(
            parameter_complex, state_transition_graphs, sheaf,
            cohomology, rank, morse_dict, attractor_secs
        )

    def general_hysteresis(self, pg, match, ordering):
        #   the coho_criteria callback: tests whether `match` (a set of
        #   path nodes, reordered by `ordering`) exhibits an algebraic
        #   switch bifurcation, per the definition at the top of the file

        #   get the whole path
        #   reorder the match so that it's ordered "left-to-right"
        match_sorted = [
            parameter_node
            for _, parameter_node in sorted(zip(ordering, match))
        ]
        #   compute the relevant sheaf data on the entire path
        path_data = self.build_sheaf_data(match_sorted)

        #   get the "center" of the path
        match_center = match_sorted[1:-1]
        #   compute the relevant sheaf data on the center
        center_data = self.build_sheaf_data(match_center)

        #   get the "left" side of the path
        match_left = match_sorted[:-1]
        left_node = match_sorted[0]
        #   compute the relevant sheaf data on the left
        left_data = self.build_sheaf_data(match_left)

        #   get the "right" side of the path
        match_right = match_sorted[1:]
        right_node = match_sorted[-1]
        #   compute the relevant sheaf data on the right
        right_data = self.build_sheaf_data(match_right)

        #   get the restriction maps
        R_path_to_left = self.build_side_restriction(
            path_data, left_data, right_node
        )
        R_path_to_right = self.build_side_restriction(
            path_data, right_data, left_node
        )
        R_left_to_center = self.build_side_restriction(
            left_data, center_data, left_node, True
        )
        R_right_to_center = self.build_side_restriction(
            right_data, center_data, right_node
        )
        #   get restriction from total path to center by composition
        R_path_to_center = np.matmul(R_left_to_center, R_path_to_left)
        #   shouldn't matter which direction we approach from
        assert (
            R_path_to_center == np.matmul(R_right_to_center, R_path_to_right)
        ).all()

        #   get the attractor sections
        #   whose immediate precessor is empty
        #   over the whole path get the zero section
        path_zero_section = (0,) * path_data.rank
        #   then get all its parents
        path_attractor_atoms = path_data.attractor_sections.parents(path_zero_section)
        #   over the left side get the zero section
        left_zero_section = (0,) * left_data.rank
        #   then get all its parents
        left_attractor_atoms = left_data.attractor_sections.parents(left_zero_section)
        #   over the right side get the zero section
        right_zero_section = (0,) * right_data.rank
        #   then get all its parents
        right_attractor_atoms = right_data.attractor_sections.parents(right_zero_section)

        #   following the notation at the beginning of the file
        #   by only considering atoms, we achieve (1.) minimality for free
        for s, a, b in product(
            path_attractor_atoms, 
            left_attractor_atoms,
            right_attractor_atoms            
        ):
            #   first we check (2.) (ordering) on the left side
            #   restrict 's' to the left side
            s_left = np.matmul(R_path_to_left, galois.GF2(s))
            #   convert back to tuple
            s_left = tuple(s_left.tolist())
            #   if 'a' is larger than 's_left'
            if left_data.attractor_sections.less(s_left, a):
                #   fails (2.), move to next
                continue

            #   next we check (2.) (ordering) on the right side
            #   restrict 's' to the right side
            s_right = np.matmul(R_path_to_right, galois.GF2(s))
            #   convert back to tuple
            s_right = tuple(s_right.tolist())
            #   if 'b' is larger than 's_right'
            if right_data.attractor_sections.less(s_right, b):
                #   fails (2.), move to next
                continue

            #   next we check (3.) (folding) in the middle
            #   restrict 's' to the center
            s_center = np.matmul(R_path_to_center, galois.GF2(s))
            #   convert back to tuple
            s_center = tuple(s_center.tolist())
            #   calculate immediate predecessors
            s_center_pred = center_data.attractor_sections.children(s_center)
            #   if s_center has multiple immediate predecessors
            if len(s_center_pred) != 1:
                #   fails (3.), return False
                continue
            #   otherwise get the immediate predecessor
            s_center_pred = next(iter(s_center_pred))

            #   restrict 'a' to the center
            a_center = np.matmul(R_left_to_center, galois.GF2(a))
            #   convert back to tuple
            a_center = tuple(a_center.tolist())
            #   restrict 'b' to the center
            b_center = np.matmul(R_right_to_center, galois.GF2(b))
            #   convert back to tuple
            b_center = tuple(b_center.tolist())
            #   calculate the union of these attractors
            a_vee_b_center = tuple(x | y for x, y in zip(a_center, b_center))
            #   if this union is not the immediate predecessor of 's'
            if s_center_pred != a_vee_b_center:
                #   fails (3.), move to next
                continue

            #   otherwise, we meet the criteria! done
            return True

        #   if we've checked all combos and still don't have a match, fails
        return False

    def stalk_slices(self, shf):
        #   map each cell of `shf` (a Sheaf) to the slice of a flattened
        #   section tuple that holds its stalk

        #   walk the cells in the same order they appear in a section
        slices = {}
        n = 0
        for cell in shf.grading[0]:
            #   get the width of this cell's stalk
            m = len(shf.stalk(cell))
            #   record the slice of the section tuple it occupies
            slices[cell] = slice(n, n + m)
            n += m
        return slices

    def build_side_restriction(self, side_data, center_data, side_node, flip_dummy=False):
        #   build the restriction matrix from `side_data` (the larger
        #   sheaf) down to `center_data` (the smaller one it contains),
        #   where `side_node` is the parameter node present in
        #   `side_data` but not `center_data`. `flip_dummy` picks which
        #   of the two calls sharing a length-3 center routes through
        #   the dummy cell

        #   initialize galois field
        gf = side_data.sheaf.GF
        #   initialize to zero map
        restriction = gf([
            [0 for _ in range(side_data.rank)]
            for _ in range(center_data.rank)
        ])

        #   get the side edge cell
        side_edge_cell = top_cech_cell(
            self.parameter_graph, side_node, 1
        )

        #   figure out if the center data has a dummy
        has_dummy = self.dummy in center_data.sheaf.grading[0]

        #   create dictionaries which maps cells to section indices
        side_stalk_slices   = self.stalk_slices(side_data.sheaf)
        center_stalk_slices = self.stalk_slices(center_data.sheaf)

        #   for each 0-cell in the side complex
        for source_cell in side_data.sheaf.grading[0]:
            #   source slice of restriction corresponds to this cell
            source_slice = side_stalk_slices[source_cell]

            #   get incident top cells
            incident_top_cells = side_data.sheaf.P.children(source_cell)
            #   if source cell is incident to side edge and is nondegenerate
            if (
                side_edge_cell in incident_top_cells
                and len(incident_top_cells) > 1
            ):
                #   get the other edge cell
                center_edge_cell = next(
                    cell for cell in incident_top_cells 
                    if cell != side_edge_cell
                )
                #   the target cell is the degenerate 0-cell defined by
                #   the inequalities of this parameter node (edge cell)
                target_cell = CechCell(
                    center_edge_cell.inequality_sets,
                    0,
                    center_edge_cell.labels
                )
                #   if there is a dummy and 'flip_dummy' is true
                if has_dummy and flip_dummy:
                    #   use dummy
                    target_cell = self.dummy
                #   get slice corresponding to this cell
                target_slice = center_stalk_slices[target_cell]

                #   build the restriction map between the two stalks
                R = morse_restriction(
                    side_data.sheaf.stalk(source_cell),
                    center_data.sheaf.stalk(target_cell)
                )
                #   load into total restriction
                restriction[target_slice, source_slice] = R
 
            elif side_edge_cell not in incident_top_cells:
                #   otherwise, if the source cell is not incident to the side
                #   edge, we know the cell is contained in the center complex!
                target_cell = source_cell
                #   get slice corresponding to this cell
                target_slice = center_stalk_slices[target_cell]

                #   construct the appropriate identity
                rank = len(side_data.sheaf.stalk(source_cell))
                R = gf(np.eye(rank).astype(int))

                #   if there is a dummy and 'flip_dummy' is false
                if has_dummy and not flip_dummy:
                    #   use dummy
                    target_cell = self.dummy
                    #   get slice corresponding to this cell
                    target_slice = center_stalk_slices[target_cell]
                    #   build the restriction map between the two stalks
                    R = morse_restriction(
                        side_data.sheaf.stalk(source_cell),
                        center_data.sheaf.stalk(target_cell)
                    )

                #   load into total restriction
                restriction[target_slice, source_slice] = R

        #   done
        return restriction

    def __init__(self, parameter_graph, length, param_stability=None):
        #   set up a query that searches paths of `length` nodes in
        #   `parameter_graph` for algebraic switch bifurcations;
        #   optional `param_stability` skips recomputing stability data

        #   a path needs at least 3 nodes for a left/center/right split
        if length < 3:
            raise ValueError("Length must be greater than 3.")
        self.parameter_graph = parameter_graph
        self.length = length
        self.build_grading(param_stability)
        #   placeholder cell standing in for a missing boundary edge
        #   when a side of the path collapses to a single vertex
        self.dummy = CechCell(tuple(frozenset({('dummy',)})), 0)

        #   build a simple path graph 0 -- 1 -- ... -- (length-1)
        vertices = list(range(length))
        edges = [(i, i+1) for i in vertices[:-1]]
        #   endpoints must be monostable (-1), interior nodes multistable (1)
        match_grading = {-1 : [0, length-1], 1 : vertices[1:-1]}
        coho_criteria = [{"custom" : self.general_hysteresis}]

        super().__init__(
            parameter_graph, vertices, edges,
            self.param_grading, match_grading, coho_criteria
        )
