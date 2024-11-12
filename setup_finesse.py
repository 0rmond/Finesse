## FINESSE PACKAGE IMPORTS ##
import finesse
from finesse.components import Laser, Lens, Mirror, Nothing, Space, Cavity

import numpy as np

from itertools import zip_longest, chain

def calculate_nr(T, wavel):
    # NOTE: wavel must be in microns!!
    T_0 = 24.5 # "reference temperature" Gayer et al 2008
    f = (T-T_0)*(T+T_0+ (2*273.16))

    # according to Gayer, from Covesion
    a1 = 5.756
    a2 = 0.0983
    a3 = 0.2020
    a4 = 189.32
    a5 = 12.52
    a6 = 1.32E-02

    b1 = 2.860E-06
    b2 = 4.700E-08
    b3 = 6.113E-08
    b4 = 1.516E-04

    C1 = a1
    C2 = b1*f
    C3 = (a2 + (b2*f))/((wavel**2)-((a3+(b3*f))**2))
    C4 = (a4+(b4*f))/((wavel**2)-(a5**2))
    C5 = -1*a6*(wavel**2)
    n = sum([C1, C2, C3, C4, C5])
    return n**0.5
    
def setup_simple(n_mirrors, cavity_length):
    """Sets up a simple finesse model that represents a cavity"""
    M = finesse.model.Model()
    M.lambda0 = 1550e-9

    start = Laser("L_1550", P=1, f=(300e6)/(1550e-9))
    mirrors = [Lens(f"m{n}", f=0.5) for n in range(0, n_mirrors)]
    M.add([start, *mirrors])
    
    s_start_m0 = Space("s_start_m0", portA=start.p1, portB=mirrors[0].p1, L=cavity_length/2)
    M.add(s_start_m0)
   
    ports = M.get_open_ports()
    spaces = [Space(f"s_m{n}", portA=ports[n], portB=ports[n+1], L=cavity_length) for n in range(0, 2*(n_mirrors-1), 2)]
    M.add(spaces)
    
    return M

def setup_cavity_with_crystal(n_mirrors, cavity_length, verbose=False):
    """ Sets up a finesse model that includes the cavity with a crystal inside it"""
    nr_crys = calculate_nr(114.2, 1.55)
    M = finesse.model.Model()
    M.lambda0 = 1550e-9
    start = Laser("L_1550", P=1, f=(300e6)/(1550e-9))

    components = np.array([[Nothing(f"L_crys_edge_{n}"), Nothing(f"crys_centre_{n}"), Nothing(f"R_crys_edge_{n}"), Lens(f"m_{n}", f=0.5)] for n in range(0, n_mirrors)]).flatten()
    model_components = [start,*components]
    M.add(model_components)

    for n in range(1, (n_mirrors*4), 4):
        M.connect(model_components[n-1], model_components[n], name=f"s_m_{n-1}_{n}", nr=1, L=(cavity_length/2)-10e-3, verbose=verbose)
        M.connect(model_components[n], model_components[n+1], name=f"s_L_cen_{n}_{n+1}", nr=nr_crys, L=10e-3, verbose=verbose)
        M.connect(model_components[n+1], model_components[n+2], name=f"s_cen_R{n+1}_{n+2}", nr=nr_crys, L=10e-3, verbose=verbose)
        M.connect(model_components[n+2], model_components[n+3], name=f"s_R_m_{n+3}_{n+4}", nr=1, L=(cavity_length/2)-10e-3, verbose=verbose)


    #print(M.component_tree(show_ports=True))
    return M

def get_relevant_qs(n_mirrors, cavity_length, q_initial):
    """Gets the beam parameters at crystal centre and each mirror in a model of the cavity with n_mirrors reflections.
    """
    M = setup_cavity_with_crystal(n_mirrors, cavity_length)
    nodes = M.optical_nodes
    beam_path = M.path(nodes[1], nodes[-1])
    left_crystal_nodes = [M.get_element(crystal) for crystal in filter(lambda component: "L_crys_edge_" in component.name, M.components)]
    right_crystal_nodes = [M.get_element(crystal) for crystal in filter(lambda component: "R_crys_edge" in component.name, M.components)]
    mirror_nodes = [M.get_element(mirror) for mirror in filter(lambda component: "m_" in component.name, M.components)]

    sim = M.propagate_beam(path=beam_path, q_in=q_initial)
    left_crystal_qs = [sim.q(f"{crystal.name}.p1.i") for crystal in left_crystal_nodes]
    left_crystal_zs = [sim.position(crystal) for crystal in left_crystal_nodes]

    right_crystal_qs = [sim.q(f"{crystal.name}.p1.i") for crystal in right_crystal_nodes]
    right_crystal_zs = [sim.position(crystal) for crystal in right_crystal_nodes]

    crystal_qs =  [x for x in chain.from_iterable(zip_longest(left_crystal_qs, right_crystal_qs)) if x]
    crystal_zs =  [x for x in chain.from_iterable(zip_longest(left_crystal_zs, right_crystal_zs)) if x]

    mirror_qs = [sim.q(f"{mirror.name}.p1.i") for mirror in mirror_nodes]
    mirror_zs = [sim.position(mirror) for mirror in mirror_nodes]

    left_mirror_qs = [mirror_qs[evens] for evens in range(0, n_mirrors, 2)]
    right_mirror_qs = [mirror_qs[odds] for odds in range(1, n_mirrors, 2)]

    left_mirror_zs = [mirror_zs[evens] for evens in range(0, n_mirrors, 2)]
    right_mirror_zs = [mirror_zs[odds] for odds in range(1, n_mirrors, 2)]

    return {
        "crystals": {"qs": crystal_qs, "zs": crystal_zs},
        "mirrors": {
            "left": {"qs": left_mirror_qs, "zs": left_mirror_zs},
            "right": {"qs": right_mirror_qs, "zs": right_mirror_zs}
        }
    }
