import numpy as np 
import re 
import pandas as pd 
import os 
from pyqubo import Array, Binary, Placeholder, Constraint
import neal
from dwave.system import LeapHybridSampler, DWaveSampler, FixedEmbeddingComposite, FixedEmbeddingComposite
import dimod
from dimod import BinaryQuadraticModel
from dwave_qbsolv import QBSolv 
import gurobipy as gp 
from gurobipy import GRB 

import networkx as nx 
import minorminer

from utils.structure import idx2coord, coord2idx
from op.fm import load_fm_model


def translate_result(df):
    L = df.columns[df.values[0] == 1]
    table = np.zeros((len(L), 4))
    for m in range(len(L)):
        i, j = extract_idx(df.columns[df.values[0] == 1][m])
        x, y, z = idx2coord(j)
        table[m, 0] = i
        table[m, 1] = x
        table[m, 2] = y
        table[m, 3] = z
    return table


def table_to_map(table, spc_size, unit_site, elements):
    Nx, Ny, Nz = spc_size[0], spc_size[1], spc_size[2]
    nsites = int(len(unit_site)) * Nx * Ny * Nz
    
    data = np.zeros((len(elements), nsites))
    for i in range(len(table)):
        idx = coord2idx(table[i, 1], table[i, 2], table[i, 3])
        data[int(table[i, 0]), idx] = 1
    return data


def extract_idx(s):
    pattern = r'x\[(\d+)\]\[(\d+)\]'
    match = re.search(pattern, s)
    
    if match:
        i = int(match.group(1))
        j = int(match.group(2))
        return i, j
    else:
        print("pattern not found")
        return 1, -1
    
    
def stack_dataframe(df, df_stack):
    if df_stack.empty:
        df_stack = df
    else:
        try:
            df_stack = pd.concat([df_stack, df], ignore_index=True)
        except Exception as e:
            print(e)
    return df_stack


def composition_grid_search(nsites, nspecies, grid_range, current=None):
    if not current:
        current = []
    if nspecies == 1:
        if grid_range[0][0] <= nsites <= grid_range[0][1]:
            yield current + [nsites]
        return
    else:
        na_min, na_max = grid_range[0]
        next_grid_range = grid_range[1:] if len(grid_range) > 1 else [(1, nsites)]
        for i in range(na_min, min(na_max, nsites - nspecies + 1) + 1):
            if nsites - i >= nspecies - 1:
                yield from composition_grid_search(nsites - i, nspecies - 1, next_grid_range, current + [i])
                

def simulate_annealing(bqm):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm)
    return sampleset


def qbsolv_annealing(Q, subQuboSize=45):
    G = nx.complete_graph(subQuboSize)
    system = DWaveSampler()
    embedding = minorminer.find_embedding(G.edge, system.edgelist)
    sampler = FixedEmbeddingComposite(system, embedding)
    sampleset = QBSolv.sample_qubo(Q, solver=sampler, solver_limit=subQuboSize, label='QUBO Optimization')
    return sampleset


def hybrid_quantum_annealing(bqm):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm)
    return sampleset


def gurobi_annealing(Q, spc_size, elements, unit_site, time_limit=None, gap_limit=None):
    Nx, Ny, Nz = spc_size[0], spc_size[1], spc_size[2]
    nspecies = len(elements)
    nsites = int(len(unit_site)) * Nx * Ny * Nz
    
    quboMat = np.zeros((nsites * nspecies, nsites * nspecies))
    for i in range(nspecies):
        for j in range(nsites):
            for k in range(nspecies):
                for l in range(nsites):
                    try:
                        quboMat[i * nsites + j, k * nsites + l] = Q[('x[' + str(i) + '][' + str(j) + ']', 'x[' + str(k) + '][' + str(l) + ']')]
                    except KeyError:
                        pass
    quboMat = list(quboMat)
    model = gp.Model("QUBO")
    n = len(quboMat)
    
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if gap_limit is not None:
        model.setParam(GRB.Param.MIPGap, gap_limit)
        
    variables = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(n)]
    objective = sum(quboMat[i][j] * variables[i] * variables[j] for i in range(n) for j in range(n))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    solution = [int(v.x) for v in variables]
    return solution
                

class hamiltonian:
    def __init__(self, nspecies, nsites):
        self.nspecies = nspecies
        self.nsites = nsites
        self.x = Array.create('x', shape=(self.nspecies, self.nsites), vartype='BINARY')
        self.M = Placeholder('M')
        self.H = 0
        
    def construct_hamiltonian(self, model_txt):
        Q, offset = load_fm_model(model_txt, self.nspecies * self.nsites, self.nsites)
        for i in range(self.nspecies):
            for j in range(self.nsites):
                for k in range(self.nspecies):
                    for l in range(self.nsites):
                        self.H += Q[i * self.nsites + j, k * self.nsites + l] * self.x[i, j] * self.x[k, l]
                        
    def apply_constraints(self, composition, mode, scale):
        K1 = self.M if mode == 'tight' else self.M * scale
        for i in range(self.nspecies):
            self.H += K1 * (sum(self.x[i, :]) - composition[i]) ** 2
            
        K2 = self.M
        for j in range(self.nsites):
            for i in range(self.nspecies):
                self.H += K2 * self.x[i, j] * (sum(self.x[:, j]) - 1)
                
    def compile_hamiltonian(self):
        return self.H.compile()
    
    def translate(self, coeff, obj):
        model = self.compile_hamiltonian()
        if obj == 'bqm':
            bqm = model.to_bqm(feed_dict={'M': coeff})
            return bqm
        elif obj == 'qubo':
            qubo, offset = model.to_qubo(feed_dict={'M': coeff})
            return qubo, offset
        elif obj == 'ising':
            ising, offset = model.to_ising(feed_dict={'M': coeff})
            return ising, offset
        else:
            print("Invalid translated format!!!")
            

