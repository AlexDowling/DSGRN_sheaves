### PlotAttractorSheaf.py
### MIT LICENSE 2024 Alex Dowling

import DSGRN
import DSGRN_utils
import pychomp
import math

from DSGRN_sheaves.Sheaf import *
from DSGRN_sheaves.Continuation import *

import matplotlib
import matplotlib.pyplot as plt

def key_label(key, network = None):
    """ Inputs a key from parameter complex, outputs a label showcasing the
        same collections of inequalities. Optional network argument to 
        include labels on the dimensions."""
    
    key_label = ""
    for i in range(len(key)-1):
        if network is not None:
            key_label = key_label + network.name(i) + ' : '
        key_label = key_label + str(list(key[i])[0]).replace("'","")
        for j in range(1,len(key[i])):
            key_label = key_label + ", " + str(list(key[i])[j]).replace("'","")
        key_label = key_label + "\n"        
    return key_label

def plot_stg(stg, morse_nodes=None, key=None, network=None, prune_grad=False, 
             ax=None, visible=True, figsize=(7,7), fname=None):
    """ Inputs a RookRulesCubicalComplex object. Builds matplotlib figure which 
        shows the Morse sets and state transition graph. """
    
    (scc_dag, graded_complex) = pychomp.FlowGradedComplex(stg.complex(), 
                                                          stg.diagram())
    connection_matrix = pychomp.ConnectionMatrix(graded_complex)
    morse_graph = DSGRN_utils.MorseGraph(stg, scc_dag, graded_complex, 
                                         connection_matrix, prune_grad)
    saveable = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        saveable = fname is not None
    if key is not None:
        ax.title.set_text(key_label(key, network))
    DSGRN_utils.PlotMorseSets(morse_graph, stg, graded_complex, ax=ax, 
                              morse_nodes=morse_nodes)
    if saveable:
        fig.savefig(fname=fname)
    if not visible:
        plt.close(fig)

def plot_stg_complex(parameter_complex, stg_dict, morse_nodes={}, network=None, 
                     prune_grad='none', columns=2, figsize=(8,8), labels=False, 
                     top_only=True, visible=True, fname=None, tight=True):
    """ Inputs a parameter complex and dictionary of RookRulesCubicalComplex
        objects corresponding to the cells in the parameter complex. Builds a
        matplotlib figure which shows the Morse sets and state transition 
        graphs at each cell. """
    
    dim = max([key[-1] for key in parameter_complex.vertices()])
    if top_only:
        plot_list = [key for key in parameter_complex.vertices() 
                     if key[-1]==dim]
    else:
        plot_list = parameter_complex.vertices()
    rows = math.ceil(len(plot_list)/columns)
    
    fig, axs = plt.subplots(rows, columns, figsize = figsize)
    n = 0
    for key in plot_list:
        stg = stg_dict[key]
        if columns == 1 and rows == 1:
            ax = axs
        elif columns == 1 or rows == 1:
            ax = axs[n]
        else:
            ax = axs[n//columns, n%columns]
        if labels:
            ax.title.set_text(key_label(key, network))
        if prune_grad == 'all' or (prune_grad == 'some' and key[-1] == dim):
            plot_stg(stg, prune_grad=True, ax=ax, 
                     morse_nodes=morse_nodes.get(key))
        else:
            plot_stg(stg, prune_grad=False, ax=ax, 
                     morse_nodes=morse_nodes.get(key))
        n = n+1
        
    while n < rows*columns:
        axs[n//columns, n%columns].axis('off')
        n = n+1 
    if tight and labels:
        fig.tight_layout()
    if fname is not None:
        fig.savefig(fname = fname)
    if not visible:
        plt.close(fig)

def plot_stg_loop(parameter_complex, stg_dict, morse_nodes={}, network=None, 
                  prune_grad='none', figsize=(8,8), labels=False, 
                  top_only=True, visible=True, fname=None, tight=True):
    """ Inputs a parameter complex in the shape of a loop and dictionary of 
        RookRulesCubicalComplex objects. Builds a matplotlib figure showing the 
        Morse sets and state transition graphs at each cell. """
    
    transpose = parameter_complex.transpose()
    pg_top = prune_grad!='none'
    pg_bottom = prune_grad=='all'
    top_list = [key for key in parameter_complex.vertices() if key[-1] == 1]
    if top_only:
        fig, axs = plt.subplots(2, 2, figsize = figsize)
        w = 1
    else:
        fig, axs = plt.subplots(3, 3, figsize = figsize)
        w = 2
    
    def plot_key(key, ax, pg, morse_nodes):
        stg = stg_dict[key]
        if labels:
            ax.title.set_text(key_label(key, network))
        plot_stg(stg, prune_grad=pg_top, ax=ax, morse_nodes=morse_nodes)
    
    key1 = top_list[0]
    plot_key(key1, axs[0, 0], pg_top, morse_nodes.get(key1))
   
    adj_keys = [key for key in parameter_complex.vertices() if 
                transpose.adjacencies(key1).intersection(
                transpose.adjacencies(key)) > set()]
    adj_keys.remove(key1) 
    
    key2 = adj_keys[0]
    plot_key(key2, axs[0, w], pg_top, morse_nodes.get(key2))
    key3 = adj_keys[1]
    plot_key(key3, axs[w, 0], pg_top, morse_nodes.get(key3))
    key4 = [key for key in top_list if key not in adj_keys and key != key1][0]
    plot_key(key4, axs[w, w], pg_top, morse_nodes.get(key4))
    
    if not top_only:
        key = list(transpose.adjacencies(key1).intersection(
                   transpose.adjacencies(key2)))[0]
        plot_key(key, axs[0, 1], pg_bottom, morse_nodes.get(key))
        key = list(transpose.adjacencies(key1).intersection(
                   transpose.adjacencies(key3)))[0]
        plot_key(key, axs[1, 0], pg_bottom, morse_nodes.get(key))
        key = list(transpose.adjacencies(key2).intersection(
                   transpose.adjacencies(key4)))[0]
        plot_key(key, axs[1, 2], pg_bottom, morse_nodes.get(key))
        key = list(transpose.adjacencies(key3).intersection(
                   transpose.adjacencies(key4)))[0]
        plot_key(key, axs[2, 1], pg_bottom, morse_nodes.get(key))
        axs[1, 1].axis('off')
        
    if tight and labels:
        fig.tight_layout()
    if fname is not None:
        fig.savefig(fname = fname)
    if not visible:
        plt.close(fig)

def build_morse_nodes(shf, section, top_only=True):
    """ Given an attractor sheaf and an H^0 cohomology class, outputs a 
        dictionary which at each key returns the Morse sets attained by the 
        global section."""
    
    assignment = {}
    dim = len(shf.grading) - 1
    if section is not None:
        n = 0
        for key in shf.grading[0]:
            m = len(shf.stalk(key))
            assignment[key] = section[n:n+m]
            n = n+m
        for k in range(0,dim):
            for key_0 in shf.grading[k]:
                for key_1 in shf.P.children(key_0):
                    if key_1 not in assignment:
                        assignment[key_1] = shf.restriction(key_0, key_1,
                                                            assignment[key_0])
    morse_nodes = {}
    for key in assignment.keys():
        if top_only and key not in shf.grading[-1]:
            continue
        morse_nodes.update({key : [i for i in range(len(shf.stalk(key))) 
                                   if assignment[key][i] != 0]}) 
    return morse_nodes

def plot_shf(shf, stg_dict, section=None, network=None, prune_grad='none', 
             columns=2, figsize=(8,8), labels=False, top_only=True, 
             visible=True, fname=None, tight=True):
    """ Inputs an attractor sheaf and the dictionary of RookRulesCubicalComplex 
        objects generating it. Builds a plot for the Morse sets and
        state transition graphs. An optional H^0 cohomology class argument 
        restricts the plot to Morse sets attained by the global section. """
 
    if section is not None:
        morse_nodes = build_morse_nodes(shf, section, top_only)
    else:
        morse_nodes = {}
    plot_stg_complex(shf.P.children_, stg_dict, morse_nodes, network, 
                     prune_grad, columns, figsize, labels, top_only, visible, 
                     fname, tight)

def plot_shf_loop(shf, stg_dict, section=None, network=None, prune_grad='none', 
                  columns=2, figsize=(8,8), labels=False, top_only=True, 
                  visible=True, fname=None, tight=True):
    """ Inputs an attractor sheaf on a loop and the corresponding dictionary 
        of RookRulesCubicalComplex objects. Plots the Morse sets and
        state transition graphs. """
    
    if section is not None:
        morse_nodes = build_morse_nodes(shf, section, top_only)
    else:
        morse_nodes = {}
    plot_stg_loop(shf.P.children_, stg_dict, morse_nodes, network, 
                  prune_grad, figsize, labels, top_only, visible, fname, tight)
