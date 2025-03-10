### ParameterComplexFigure.py
### MIT LICENSE 2024 Alex Dowling

import DSGRN_utils
import pychomp
from math import ceil
from numpy import array

import matplotlib
import matplotlib.pyplot as plt

def plot_stg(stg, morse_sets=None, plot_bdry_cells=True, prune_grad=False,
             ax=None, label=None, figsize=(7,7), fname=None, visible=True):
    """ Inputs a CubicalBlowupGraph object. 
    
        Builds matplotlib figure  which shows the Morse sets and state 
        transition graph. """
    
    (scc_dag, graded_complex) = pychomp.FlowGradedComplex(
                                        stg.complex(), 
                                        stg.digraph.adjacencies)
    connection_matrix = pychomp.ConnectionMatrix(graded_complex)
    morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                         connection_matrix, prune_grad)
    if morse_sets is not None:
        all_morse_sets = [frozenset({c for c in stg.digraph.vertices() 
                                     if graded_complex.value(c) == M}) 
                          for M in morse_graph.vertices()]
        morse_nodes = [all_morse_sets.index(M) for M in morse_sets]
    else:
        morse_nodes = None
    
    saveable = False
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        saveable = fname is not None
    if label:
        ax.title.set_text(label)
    DSGRN_utils.PlotMorseSets(morse_graph, stg, graded_complex, ax=ax, 
                              morse_nodes=morse_nodes, 
                              plot_bdry_cells=plot_bdry_cells)
    if saveable:
        fig.savefig(fname=fname)
    if not visible:
        plt.close(fig)

class ParameterComplexFigure:

    def set_defaults(self):
        self.prune_grad = 'none'
        self.plot_bdry_cells = True
        self.figsize = (8,8)
        self.labels = True
        self.top_only = False
        self.visible = True

    def build_plot_list(self):
        if self.top_only:
            self.plot_list = [cell for cell 
                              in self.parameter_complex.vertices() 
                              if cell.dim == self.dim]
        else:
            self.plot_list = list(self.parameter_complex.vertices())

    def clean_axs(self):
        for ax in self.axs.reshape(-1):
            if not ax.lines:
                ax.axis('off')

    def plot_cell(self, cell, ax):
        stg = self.stg_dict[cell]
        if self.parameter_graph:
            label = str(cell.get_descendant_parameters(self.parameter_graph))
        else:
            label = None

        if self.prune_grad == 'all' or (self.prune_grad == 'some' 
                                        and cell.dim == self.dim):
            plot_stg(stg, plot_bdry_cells=self.plot_bdry_cells, 
                     prune_grad=True, ax=ax, label=label, 
                     figsize=self.figsize)
        else:
            plot_stg(stg, plot_bdry_cells=self.plot_bdry_cells, 
                     prune_grad=False, ax=ax, label=label, 
                     figsize = self.figsize)
    
    def plot(self, fname=None):
        self.rows = ceil(len(self.plot_list)/self.columns)
        self.fig, self.axs = plt.subplots(self.rows, self.columns, 
                                          figsize=self.figsize)
        self.axs = array(self.axs)
        self.ax_list = self.axs.reshape(-1)
        
        for i, cell in enumerate(self.plot_list):
            ax = self.ax_list[i]
            self.plot_cell(cell, ax)
        self.clean_axs()
        
        if fname:
            self.fig.savefig(fname=self.fname)
        if not self.visible:
            plt.close(self.fig)

    def __init__(self, parameter_complex, stg_dict, parameter_graph=None, 
                 columns=3):
        self.parameter_complex = parameter_complex
        self.stg_dict = stg_dict
        self.parameter_graph = parameter_graph
        self.columns = columns
        self.set_defaults()
        self.build_plot_list()
