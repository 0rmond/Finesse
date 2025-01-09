def get_specific_results(finesse_model, q_initial, component_type):
    """Simulates a beam propegating through specific components in a model
    """
    M = finesse_model
    nodes = M.optical_nodes
    beam_path = M.path(nodes[1], nodes[-1])

    sim = M.propagate_beam(path=beam_path, q_in=q_initial)
    relevant_components = M.get_elements_of_type(Lens)

def get_all_results(finesse_model, q_initial):
    """Simulates a beam propegating through ALL components in a model
    """
    M = finesse_model
    nodes = M.optical_nodes
    beam_path = M.path(nodes[1], nodes[-1])

    sim = M.propagate_beam(path=beam_path, q_in=q_initial)

    results = sim.all_segments("beamsize")

    return results

def get_zs_and_ws(results):
    """From the results of a simulation, get the beam waists, ws, and their positions, zs.
    """

    zs = np.array(list(reduce(lambda acc, curr: [*acc, *curr], [results[segment][0] for segment in results])))
    ws = np.array(list(reduce(lambda acc, curr: [*acc, *curr], [results[segment][1]["beamsize"] for segment in results])))

    return {"zs": zs, "ws": ws}

