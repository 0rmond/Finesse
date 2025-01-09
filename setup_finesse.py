## FINESSE PACKAGE IMPORTS ##
import finesse
from finesse.components import Laser, Lens, Mirror, Nothing, Space, Cavity

import numpy as np

from itertools import zip_longest, chain
from functools import reduce

def calculate_nr(T, wavel):
    """Calculates refractive index, nr, of a crystal according to Sellmeir equations.
    """
    # NOTE: wavelength must be in microns!!
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

def setup_cavity_with_crystal(n_mirrors, cavity_length, verbose=False, show_graph=False):
    """ Sets up a finesse model that includes the cavity with a crystal inside it"""

    T = 114.2 #phase matching temperature
    nr_crys = calculate_nr(T, 1.55)
    M = finesse.model.Model()
    M.lambda0 = 1550e-9
    start = Laser("L_1550", P=1, f=(300e6)/(1550e-9))

    def unit_component(n):
    # Think of the following array as a unit cell in a crystal structure.
        return [Nothing(f"L_crys_edge_{n}"), Nothing(f"crys_centre_{n}"), Nothing(f"R_crys_edge_{n}"), Lens(f"m_{n}", f=0.5)]

    components = np.array([unit_component(n) for n in range(0, n_mirrors)]).flatten()
    model_components = [start,*components]
    M.add(model_components)

    # In between every component, we need to add a "space" with a refractive index, nr, and length, L.
    for n in range(1, (n_mirrors*4), 4):
        M.connect(model_components[n-1], model_components[n], name=f"s_m_{n-1}_{n}", nr=1, L=(cavity_length/2)-10e-3, verbose=verbose)
        M.connect(model_components[n], model_components[n+1], name=f"s_L_cen_{n}_{n+1}", nr=nr_crys, L=10e-3, verbose=verbose)
        M.connect(model_components[n+1], model_components[n+2], name=f"s_cen_R{n+1}_{n+2}", nr=nr_crys, L=10e-3, verbose=verbose)
        M.connect(model_components[n+2], model_components[n+3], name=f"s_R_m_{n+3}_{n+4}", nr=1, L=(cavity_length/2)-10e-3, verbose=verbose)

    """
    # Set the distance between the Laser and the first unit_component's crystal to be zero.
    # This is so that the beam waist propagates out of the crystal centre
    M.get_elements_of_type(Space)[0].L = 0 # set space between Laser and first crystal edge to be 0
    M.get_elements_of_type(Space)[1].L = 0 # squash the first half of the first crystal to zero 
    """

    if show_graph==True:
        M.plot_graph(graphviz=False)
        print(M.component_tree())

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
    # get INPUT on the left face because we need to know how the beam goes into the crystal
    left_crystal_qs = [sim.q(f"{crystal.name}.p1.i") for crystal in left_crystal_nodes]
    left_crystal_zs = [sim.position(crystal) for crystal in left_crystal_nodes]

    # get OUTPUT on the right face because we need to know how the beam exits the crystal
    right_crystal_qs = [sim.q(f"{crystal.name}.p1.o") for crystal in right_crystal_nodes]
    right_crystal_zs = [sim.position(crystal) for crystal in right_crystal_nodes]

    # interlace into one list
    crystal_qs =  [x for x in chain.from_iterable(zip_longest(left_crystal_qs, right_crystal_qs)) if x]
    crystal_zs =  [x for x in chain.from_iterable(zip_longest(left_crystal_zs, right_crystal_zs)) if x]

    # get the beam parameter that's incident on the mirror
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
def setup_focusing(q_init, cavity_length, d1, d2, d3, verbose=False, show_graph=False):
    """Setup that includes crystal, cavity input mirror, two lenses and a target to represent the collimator.
    """
    M = finesse.model.Model()
    M.lambda0 = q_init.wavelength #the beam that focuses down will be 775nm
    start = Laser("start", P=1)

    cavity_components = [Nothing("crys_centre"), Nothing("R_crys_edge"), Lens("m", f=0.5)]
    lenses = [Lens("l1", f=0.15), Lens("l2", f=0.2)]
    target = Nothing("collimator")
    M.add([start, *cavity_components, lenses, target])

    M.connect(start, cavity_components[0], name="s_start_crystal_centre", L=0, verbose=verbose)
    M.connect(cavity_components[0], cavity_components[1], name="s_crystal_centre_crystal_edge", nr=1.3, L=10e-3, verbose=verbose)
    M.connect(cavity_components[1], cavity_components[2], name="s_crystal_edge_mirror", nr=1, L=(cavity_length/2)-10e-3, verbose=verbose)
    M.connect(cavity_components[2], lenses[0], name="s_mirror_l1", nr=1, L=d1, verbose=verbose)
    M.connect(lenses[0], lenses[1], name="s_l1_l2", nr=1, L=d2, verbose=verbose)
    M.connect(lenses[1], target, name="s_l2_collimator", nr=1, L=d3, verbose=verbose)

    end = Laser("end", P=1)
    M.add(end)
    M.connect(target, end, name="s_t_end", nr=1, L=1e-3, verbose=verbose)

    if show_graph == True:
        print(M.component_tree())
        M.plot_graph(graphviz=False)

    return M

def get_focusing_qs(q_init, cavity_length, d1, d2, d3, reverse=False):
    """Get the beam parameters for focusing simulation
    """
    M = setup_focusing(q_init, cavity_length, d1, d2, d3)
    nodes = M.optical_nodes if not reverse else list(reversed(M.optical_nodes))
    #pprint(nodes)
    beam_path = M.path(nodes[1], nodes[-3]) if not reverse else M.path(nodes[1], nodes[-3])
    #beam_path = M.path(nodes[1], nodes[-3]) if not reverse else M.path(nodes[1], nodes[-2])


    sim = M.propagate_beam(path=beam_path, q_in=q_init)
    
    component_positions = sim.positions

    results = sim.all_segments("beamsize")

    zs = np.array(list(reduce(lambda acc, curr: [*acc, *curr], [results[segment][0] for segment in results])))
    ws = np.array(list(reduce(lambda acc, curr: [*acc, *curr], [results[segment][1]["beamsize"] for segment in results])))


    return {"ws": ws, "zs": zs, "component_positions": component_positions}
