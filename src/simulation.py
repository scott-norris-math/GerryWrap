import copy
import sys
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        updaters, constraints, accept, Election, metrics)
from gerrychain.proposals import recom
from gerrychain.metrics import efficiency_gap, mean_median
from functools import partial
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from datetime import datetime
from shapely import wkt
from collections import defaultdict
from typing import Iterable, Any, NamedTuple
import pickle
from addict import Dict
import math

import common as cm
import utilities as ut
import data_transform as dt
import proposed_plans as pp


def build_canonical_partition_list(partition: Partition):
    partition_assignment = [(x, partition.assignment[x]) for x in partition.graph.nodes]
    return dt.build_canonical_assignments_list(partition_assignment)


def build_canonical_plan_from_partition(node_id_to_index: dict[str, int], partition: Partition) -> frozenset[frozenset]:
    district_plans_lookup = defaultdict(set)
    for x, y in partition.assignment.items():
        district_plans_lookup[int(y)].add(node_id_to_index[x])
    return frozenset(frozenset(x) for x in district_plans_lookup.values())


def calculate_plan_hash_from_partition(node_id_to_index: dict[str, int], partition: Partition) -> int:
    return hash(build_canonical_plan_from_partition(node_id_to_index, partition))


def diff_plan(plan1: np.ndarray, plan2: np.ndarray):
    data_type = np.uint16

    changed_indices = [i for i, (x, y) in enumerate(zip(plan1, plan2)) if not x == y]
    if len(changed_indices) == 0:
        return {
            'changed_indices': np.empty([0], data_type),
            'added_nodes': np.empty([0], data_type),
            'removed_nodes': np.empty([0], data_type)
        }
    elif len(changed_indices) == 2:
        first_changed_index = changed_indices[0]
        return {
            'changed_indices': np.array(changed_indices, data_type),
            'added_nodes': np.array(list(plan2[first_changed_index].difference(plan1[first_changed_index])), data_type),
            'removed_nodes': np.array(list(plan1[first_changed_index].difference(plan2[first_changed_index])),
                                      data_type)
        }
    else:
        raise RuntimeError("Each plan must differ by zero or two partitions")


def display_partition(initial_partition: Partition, partition: Partition):
    for x in partition.graph.nodes:
        print([x, initial_partition.assignment[x], partition.assignment[x]])


def build_geographic_partition(graph: Graph, assignment_column: str) -> GeographicPartition:
    return GeographicPartition(graph, assignment=assignment_column, updaters=[])


def load_geodataframe(directory: str, redistricting_data_filename: str = None) -> gpd.GeoDataFrame:
    redistricting_data_directory = pp.build_redistricting_data_directory(directory)

    if redistricting_data_filename is None:
        node_data = pd.read_csv(f'{redistricting_data_directory}nodes_TX_2020_cntyvtd_sldu.csv')
        node_data['geometry'] = node_data['polygon'].apply(wkt.loads)
        county_vtd_geodata = gpd.GeoDataFrame(node_data, geometry='geometry')
        county_vtd_geodata.geometry = county_vtd_geodata.geometry.buffer(0)

        county_data = pd.read_csv(f'{redistricting_data_directory}nodes_TX_2020_cnty_sldu.csv')
        county_data['geometry'] = county_data['polygon'].apply(wkt.loads)
        county_geodata = gpd.GeoDataFrame(county_data, geometry='geometry')
        county_geodata.geometry = county_geodata.geometry.buffer(0)

        county_geodata['geoid'] = county_geodata['geoid'].astype('str').apply(lambda p: p.zfill(3))
        county_vtd_geodata['geoid'] = county_vtd_geodata['geoid'].astype('str').apply(lambda p: p.zfill(9))

        geodata = pd.concat([county_vtd_geodata, county_geodata]).drop_duplicates().set_index('geoid', drop=False)
    else:
        node_data = pd.read_parquet(
            f'{redistricting_data_directory}{redistricting_data_filename}')

        node_data['geometry'] = node_data['polygon'].apply(wkt.loads)
        node_geodata = gpd.GeoDataFrame(node_data, geometry='geometry')
        node_geodata.geometry = node_geodata.geometry.buffer(0)

        fix_election_columns_text(node_geodata)
        geodata = node_geodata.set_index('geoid', drop=False)
    return geodata


def fix_election_columns_text(df: pd.DataFrame) -> None:
    df.rename(columns={
        'USSen_2020_D_Hegar_general': 'USSen_2020_general_D_Hegar',
        'USSen_2020_R_Cornyn_general': 'USSen_2020_general_R_Cornyn',
        'President_2020_D_Biden_general': 'President_2020_general_D_Biden',
        'President_2020_R_Trump_general': 'President_2020_general_R_Trump'}, inplace=True)


def load_graph_with_geometry(directory: str, dual_graph_filename: str, geodata: gpd.GeoDataFrame) -> Graph:
    seeds_directory = cm.build_seeds_directory(directory)
    networkXGraph = nx.read_gpickle(seeds_directory + dual_graph_filename)

    graph = Graph(networkXGraph, geometry=geodata)
    graph.join(geodata, columns=["geometry"], right_index="geoid")

    # TODO: this may be unnecessary - check
    filtered_df = geodata[geodata['geoid'].isin(graph.nodes().keys())]
    graph.geometry = filtered_df.geometry
    graph.geometry.index = filtered_df.geoid

    return graph


def join_data_to_dual_graph(graph: Graph, data: pd.DataFrame) -> None:
    print(f"Index: {data.index}")
    print(f"Columns: {data.columns}")

    # TODO: these joins can all be done at once
    graph.join(data, columns=['President_2020_general_D_Biden', 'President_2020_general_R_Trump'],
               right_index='geoid')
    graph.join(data, columns=['USSen_2020_general_D_Hegar', 'USSen_2020_general_R_Cornyn'], right_index='geoid')

    graph.join(data, columns=['o17_hisp_pop', 'o17_pop'], right_index='geoid')

    # TODO: use methods in common for aggregating
    all_o17_columns = [col for col in data.columns if (col.startswith('o17') and not col.startswith('o17_nonhisp'))]
    nonhisp_o17_columns = [col for col in data.columns if col.startswith('o17_nonhisp')]

    # Black
    black_o17_columns = [col for col in all_o17_columns if 'black' in col]
    data['black_o17_sum'] = data[black_o17_columns].sum(axis=1)

    # Non-Hispanic Black
    black_nonhisp_o17_columns = [col for col in nonhisp_o17_columns if 'black' in col]
    data['black_nonhisp_o17_sum'] = data[black_nonhisp_o17_columns].sum(axis=1)

    graph.join(data, columns=['black_o17_sum', 'black_nonhisp_o17_sum'], right_index='geoid')


def build_updaters():
    updaters_dict = {"population": updaters.Tally("total_pop", alias="population"),
                     "countysplits": updaters.county_splits("countysplits", "county"),
                     "o17_hisp_pop": updaters.Tally("o17_hisp_pop"),
                     "o17_pop": updaters.Tally("o17_pop"),
                     "black_o17_sum": updaters.Tally("black_o17_sum"),
                     "black_nonhisp_o17_sum": updaters.Tally("black_nonhisp_o17_sum")}

    elections = [
        Election("SEN20", {"Democratic": "USSen_2020_general_D_Hegar", "Republican": "USSen_2020_general_R_Cornyn"}),
        Election("PRES20",
                 {"Democratic": "President_2020_general_D_Biden", "Republican": "President_2020_general_R_Trump"}),
    ]
    updaters_dict.update({election.name: election for election in elections})

    return updaters_dict, elections


def build_assignment_column(chamber: str, is_plan_proposal: bool):
    if not is_plan_proposal:
        return dt.get_bq_chamber_name(chamber)
    else:
        return 'district'


def determine_population_limit(chamber: str):
    return {
        'USCD': .01,
        'TXSN': .02,
        'TXHD': .05
    }[chamber]


def load_county_district_graph(directory, filename):
    seeds_directory = cm.build_seeds_directory(directory)
    graph = nx.read_gpickle(f'{seeds_directory}{filename}')
    set_bipartitite_flag(graph)
    return graph


def build_chain(chamber: str, directory: str, settings, number_steps: int) -> (MarkovChain, Iterable[Election]):
    geodata = load_geodataframe(directory, settings.redistricting_data_filename)

    graph = load_graph_with_geometry(directory, settings.networkX_graph_filename, geodata)
    join_data_to_dual_graph(graph, geodata)

    updaters, elections = build_updaters()

    # need to assign districts
    is_plan_proposal = True
    assignment_column = build_assignment_column(chamber, is_plan_proposal)
    initial_partition = GeographicPartition(graph, assignment=assignment_column, updaters=updaters)

    # Now we will really generate many plans and save data
    proposal = partial(recom,
                       pop_col="total_pop",
                       pop_target=calculate_idead_population(initial_partition),
                       epsilon=settings.epsilon,
                       node_repeats=2)

    if chamber == 'TXHD':
        county_district_graph = load_county_district_graph(directory, settings.country_district_graph_filename)
        state = {
            'graph': county_district_graph,
            'node_id_to_index': build_node_id_to_index(initial_partition)
        }
        accept_function = build_accept_function(state)
    else:
        accept_function = accept.always_accept

    number_cut_edges = len(initial_partition["cut_edges"])
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * number_cut_edges
    )

    # number_county_splits = len(initial_partition["countysplits"])
    # split_bound = constraints.refuse_new_splits("countysplits")
    # split_bound = constraints.UpperBound(
    #    lambda p: len(p["countysplits"]), number_county_splits
    # )

    chosen_constraints = [
        # District populations must stay within % of equality
        constraints.within_percent_of_ideal_population(initial_partition, settings.epsilon),
        compactness_bound,
    ]

    chain = MarkovChain(
        proposal=proposal,
        constraints=chosen_constraints,
        accept=accept_function,
        initial_state=initial_partition,
        total_steps=number_steps
    )

    if chamber == 'TXHD':
        state['chain'] = chain

    return chain, elections


def run_chain(chain: MarkovChain, elections: Iterable[Election], output_directory: str) -> None:
    def build_maps_directory(directory):
        return f'{directory}maps/'

    ut.ensure_directory_exists(output_directory)

    node_id_to_index = build_node_id_to_index(chain.initial_state)
    save_columns_map(output_directory, node_id_to_index)

    data = initialize_data(chain.total_steps, elections)

    plans_array = None
    plans_output_size = 100000
    step_number = 0
    maps_directory = build_maps_directory(output_directory)
    ut.ensure_directory_exists(maps_directory)
    for partition in chain:
        if step_number % 10000 == 0:
            partition.plot()
            plt.savefig(f'{maps_directory}mapstep{step_number}.png')
            # plt.show()

        if step_number % 100 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} Step: {step_number}")

            # print("Partition Assignments")
            # display_partition(chain.initial_state, partition)
            #
            # print("Partition List")
            # partition_as_list = build_canonical_partition_list(partition)
            # print(partition_as_list)

        plan_index = step_number % plans_output_size
        if plan_index == 0:
            if step_number > 0:
                file_index = (step_number // plans_output_size) - 1
                np.savez_compressed(f"{newdir}/plans_{file_index}.npz", plans_array)
            number_nodes = len(partition.graph.nodes)
            plans_array = np.ndarray([plans_output_size, number_nodes], np.uint8)

        current_plan = build_canonical_plan_from_partition(node_id_to_index, partition)
        for district_index, node_set in enumerate(current_plan):
            district_index_npu8 = np.uint8(district_index) + 1
            for node_index in node_set:
                plans_array[plan_index, node_index] = district_index_npu8

        update_data(partition, data, step_number)

        step_number += 1

    save_data(output_directory, data)
    save_compressed_data(output_directory, data)
    save_partial_plans(output_directory, plans_array, plans_output_size, step_number)


def build_node_id_to_index_from_strings(strings: Iterable[str]):
    return {x: i for i, x in enumerate(sorted(strings))}


def build_node_id_to_index(partition: Partition):
    return build_node_id_to_index_from_strings(partition.assignment)


def save_columns_map(output_directory: str, node_id_to_index: dict[str, int]):
    outfile = open(f'{output_directory}/columns_map.pickle', 'wb')
    pickle.dump(node_id_to_index, outfile)
    outfile.close()


def initialize_data(number_steps: int, elections: Iterable[Election]) -> Dict:
    data = Dict()

    data.elections = elections
    data.election_data = pd.DataFrame([], range(0, number_steps), [election.name for election in elections])

    # Votes for each election
    # Seven spots, for seven elections
    data.votes = [[], []]

    # mean-median, efficiency gap
    #
    # hmss = number of districts won, for each particular election
    # (although the elections we discuss here are all state-wide
    #      i.e. this is a proxy for # districts won)
    data.mms = []
    data.egs = []
    data.hmss = []

    # this will store the partisan gini scores for each plan, and each election
    data.gini = []

    data.HVAP_data = []
    data.BVAP_data = []
    data.BHVAP_data = []

    data.pop_vec = []
    data.cut_vec = []

    return data


def update_data(partition: Partition, data: Dict, step_number: int) -> None:
    data.mms.append([])
    data.egs.append([])
    data.hmss.append([])
    for election_number, election in enumerate(data.elections):
        election_data = partition[election.name]

        data.votes[election_number].append(election_data.percents("Democratic"))
        data.mms[-1].append(mean_median(election_data))
        data.egs[-1].append(efficiency_gap(election_data))
        data.hmss[-1].append(election_data.wins("Democratic"))

    data.election_data.loc[step_number] = {election.name: partition[election.name].percents("Democratic") for election
                                           in data.elections}

    data.gini.append([metrics.partisan_gini(partition[election.name]) for election in data.elections])

    data.HVAP_data.append(
        list(partition["o17_hisp_pop"][key] / partition["o17_pop"][key] for key in partition["population"]))

    data.BVAP_data.append(
        list(partition["black_o17_sum"][key] / partition["o17_pop"][key] for key in partition["population"]))

    data.BHVAP_data.append(
        list((partition["o17_hisp_pop"][key] + partition["black_nonhisp_o17_sum"][key]) / partition["o17_pop"][key]
             for key in partition["population"]))

    data.pop_vec.append(list(partition["population"].values()))
    data.cut_vec.append(len(partition["cut_edges"]))


def save_data(directory: str, data: Dict) -> None:
    for i, election in enumerate(data.elections):
        cm.save_vector_csv(f"{directory}votes_{election.name}.npz", data.votes[i])
    cm.save_vector_csv(f"{directory}gini.csv", data.gini)
    cm.save_vector_csv(f"{directory}mms.csv", data.mms)
    cm.save_vector_csv(f"{directory}egs.csv", data.egs)
    cm.save_vector_csv(f"{directory}hmss.csv", data.hmss)

    cm.save_vector_csv(f"{directory}hisp_perc.csv", data.HVAP_data)
    cm.save_vector_csv(f"{directory}black_perc.csv", data.BVAP_data)
    cm.save_vector_csv(f"{directory}black_hisp_perc.csv", data.BHVAP_data)

    cm.save_vector_csv(f"{directory}pop_vec.csv", data.pop_vec)
    cm.save_vector_csv(f"{directory}cut_edge.csv", data.cut_vec)


def save_compressed_data(directory: str, data: Dict) -> None:
    for i, election in enumerate(data.elections):
        np.savez(f"{directory}votes_{election.name}.npz", np.array(data.votes[i]))
    np.savez(f"{directory}gini.npz", np.array(data.gini))
    np.savez(f"{directory}mms.npz", np.array(data.mms))
    np.savez(f"{directory}egs.npz", np.array(data.egs))
    np.savez(f"{directory}hmss.npz", np.array(data.hmss))

    np.savez(f"{directory}hisp_perc.npz", np.array(data.HVAP_data))
    np.savez(f"{directory}black_perc.npz", np.array(data.BVAP_data))
    np.savez(f"{directory}black_hisp_perc.npz", np.array(data.BHVAP_data))

    np.savez(f"{directory}pop_vec.npz", np.array(data.pop_vec))
    np.savez(f"{directory}cut_edge.npz", np.array(data.cut_vec))


def save_partial_plans(directory: str, plans_array: np.ndarray, plans_output_size: int, step_number: int) -> None:
    plan_index = step_number % plans_output_size
    if plan_index == 0:
        plan_index = plans_output_size
    partial_plans_array = plans_array[0:plan_index]
    file_index = math.ceil(step_number / plans_output_size) - 1
    np.savez_compressed(f'{directory}/plans_{file_index}.npz', partial_plans_array)


# defect code


class CountyDefect(NamedTuple):
    county: str
    number_whole_districts: int
    defect_whole: int
    number_intersected_districts: int
    defect_intersected: int


def determine_isolated_counties(graph: nx.Graph, counties: Iterable[str], minimum_whole_counties: int) -> list[str]:
    county_defects = calculate_county_defects(graph, counties)
    return [x.county for x in county_defects if
            x.number_whole_districts >= minimum_whole_counties and
            x.number_whole_districts == x.number_intersected_districts]


def calculate_county_defects(graph: nx.Graph, counties: Iterable[str]) -> list[CountyDefect]:
    # The "county-line" rule prefers minimal county & district splitting. We implement as follows:
    # seats_share = county population / distrinct ideal population
    # Ideally, county should wholly contain floor(seats_share) and intersect ceiling(seats_share) districts
    # Ex: County seats_share=2.4, so it should ideally wholly contain 2 districts and intersect a 3rd.
    # whole_defect = |actual wholly contained - floor(seats_share)|
    # intersect_defect = |actual intersected - ceil(seats_share)|
    # defect = whole_defect + intersect_defect

    # Assume that: You have already identified Cnodes, they are passed
    # Dnodes = bipartite.set(B)  - Cnodes

    data = []
    for county in counties:
        # Number of whole districts; i.e. Dnodes which are ONLY connected to this county
        number_whole_districts = sum(graph.degree[district_node] == 1
                                     for district_node in graph[county])
        defect_whole = abs(graph.nodes[county]['whole_target'] - number_whole_districts)
        number_intersected_districts = graph.degree[county]
        defect_intersected = abs(
            graph.nodes[county]['intersect_target'] - number_intersected_districts)

        data.append(CountyDefect(county, number_whole_districts, defect_whole,
                                 number_intersected_districts, defect_intersected))

    return data


def calculate_defect(graph: nx.Graph, counties: Iterable[str]) -> int:
    county_defects = calculate_county_defects(graph, counties)
    print([x for x in county_defects if (x.defect_whole + x.defect_intersected) > 0])
    return sum([(x.defect_whole + x.defect_intersected) for x in county_defects])


def set_bipartitite_flag(county_district_graph: nx.Graph) -> None:
    for node_id, node in county_district_graph.nodes.items():
        if isinstance(node_id, str):
            node['bipartite'] = 0
        else:
            node['bipartite'] = 1


def extract_counties(county_district_graph: nx.Graph):
    return {node for node, data in county_district_graph.nodes(data=True) if data['bipartite'] == 0}


def extract_defect_targets(county_district_graph: nx.Graph) -> (dict[str, int], dict[str, int]):
    counties = extract_counties(county_district_graph)

    whole_targets = {}
    intersect_targets = {}
    for county in counties:
        whole_targets[county] = county_district_graph.nodes[county]['whole_target']
        intersect_targets[county] = county_district_graph.nodes[county]['intersect_target']

    return whole_targets, intersect_targets


def build_county_district_graph(dual_graph: Graph, assignment: dict[str, int], whole_targets: dict[str, int],
                                intersect_targets: dict[str, int]) -> nx.Graph:
    graph = nx.Graph()

    county_districts = {(node['county'], assignment[geoid]) for geoid, node in dual_graph.nodes.items()}
    for county, district in county_districts:
        graph.add_node(county)
        graph.add_node(district)
        graph.add_edge(county, district)

    counties = {x for x, y in county_districts}
    for county in counties:
        graph.nodes[county]['whole_target'] = whole_targets[county]
        graph.nodes[county]['intersect_target'] = intersect_targets[county]

    return graph


def update_county_district_graph(dual_graph: nx.Graph, county_district_graph: nx.Graph, old_assignment: dict[str, id],
                                 new_assignment: dict[str, id], copy_graph: bool) -> nx.Graph:
    updated_graph = copy.deepcopy(county_district_graph) if copy_graph else county_district_graph

    changed_counties = []
    changed_districts = []
    for geoid, district in new_assignment.items():
        if district != old_assignment[geoid]:
            changed_counties.append(dual_graph.nodes[geoid]['county'])
            changed_districts.append(district)

    unique_changed_districts = set(changed_districts)

    # Remove edges from copy
    for district in unique_changed_districts:
        edges_to_remove = list(updated_graph.edges(district))
        updated_graph.remove_edges_from(edges_to_remove)

    # Now go through new_assign, find new counties
    changed_counties_districts = zip(changed_counties, changed_districts)
    for district in unique_changed_districts:
        new_counties = set([x for x, y in changed_counties_districts if y == district])
        for county in new_counties:
            updated_graph.add_edge(county, district)

    return updated_graph


def build_accept_function(state: dict[str, Any]):
    return lambda x: better_defect_accept(x, state)


def better_defect_accept(partition: Partition, state: dict[str, Any]):
    """Measure defect. If defect is better, accept.
    If defect is worse, random prob of acceptance"""
    # This is the bipartite graph that encodes the "parent"
    if 'step_number' not in state:
        state['step_number'] = 0
    step_number = state['step_number']
    state['step_number'] = step_number + 1

    if partition.parent is None:
        return 1

    graph = state['graph']

    county_nodes = {node for node, data in graph.nodes(data=True) if data['bipartite'] == 0}

    # old_defect = calculate_defect(county_district_graph, county_nodes)
    if 'initial_defect' not in state:
        old_defect = calculate_defect(graph, county_nodes)
        state['initial_defect'] = old_defect
        print(f"Initial Defect: {old_defect}")
    else:
        old_defect = state['initial_defect']

    parent_assignment = partition.parent.assignment
    assignment = partition.assignment
    updated_graph = update_county_district_graph(partition.graph, graph, parent_assignment, assignment,
                                                 True)
    new_defect = calculate_defect(updated_graph, county_nodes)

    bound = 1
    k = 3  # severity of penalizing defect
    # If new_bal<old_bal, leave prob(accept)=1
    if new_defect > old_defect:
        bound = np.exp(-k * (new_defect - old_defect))

    accept = np.random.random() < bound

    if 'plan_hashes' not in state:
        state['plan_hashes'] = set()
    plan_hashes = state['plan_hashes']

    if 'previous_parent' not in state:
        state['previous_parent'] = []
    previous_parent = state['previous_parent']

    if accept:
        # print(f"Accepted: {[old_defect,new_defect,bound,Qaccept]}")
        state['graph'] = updated_graph
        plan_hash = calculate_plan_hash_from_partition(state['node_id_to_index'], partition)
        plan_hashes.add(plan_hash)
        if len([x for x, y, z in previous_parent if x == plan_hash]) == 0:
            previous_parent.append((plan_hash, partition, graph))
        state['failures'] = 0
    else:
        if 'failures' not in state:
            state['failures'] = 0
        failures = state['failures']

        go_back = (failures > 20) and (np.random.random() < .1)
        if go_back and len(previous_parent) > 0:
            _, previous_state, previous_graph = previous_parent.pop()
            state['chain'].state = previous_state
            print(f"Went Back: {len(previous_parent)}")
            state['failures'] = 0
            state['graph'] = previous_graph
        else:
            state['failures'] += 1

    if step_number % 20 == 0:
        print(f'{[old_defect, new_defect, bound, accept]} Number Plans: {len(plan_hashes)}')

    return accept


# utilities


def display_population_deviations(initial_partition):
    ideal_population = calculate_idead_population(initial_partition)
    print(len(initial_partition))
    print(ideal_population)
    populations = initial_partition["population"]
    numpy.concatenate(populations.values())
    print((numpy.max([x for x in populations.values()]) - ideal_population) / ideal_population)
    print((numpy.min([x for x in populations.values()]) - ideal_population) / ideal_population)


def calculate_idead_population(partition):
    return sum(partition["population"].values()) / len(partition)


def calculate_size(canonical_plans):
    size = 0
    for plan in canonical_plans:
        for district_group in plan:
            size += sys.getsizeof(district_group)
    return size


def build_TXSN_random_seed_settings() -> Dict:
    settings = Dict()
    settings.networkX_graph_filename = 'TX_2020_cntyvtd_sldu_seed_1000000_graph.gpickle'
    settings.epsilon = determine_population_limit('TXSN')
    return settings


def build_proposed_plan_settings(chamber: str, plan: int) -> Dict:
    settings = Dict()
    settings.networkX_graph_filename = f'graph_TX_2020_cntyvtd_{chamber}_{plan}.gpickle'
    settings.redistricting_data_filename = f'nodes_TX_2020_cntyvtd_{chamber}_{plan}.parquet'
    settings.country_district_graph_filename = f'adj_TX_2020_cntyvtd_{chamber}_{plan}.gpickle'
    settings.epsilon = determine_population_limit(chamber)
    return settings


def calculate_defects(dual_graph, county_district_graph, plans):
    previous_assignment = None
    defects = []
    counties = extract_counties(county_district_graph)
    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(i)

        assignment = cm.build_assignment(dual_graph, plan)

        if i == 0:
            whole_targets, intersect_targets = extract_defect_targets(county_district_graph)
            county_district_graph = build_county_district_graph(dual_graph, assignment, whole_targets,
                                                                intersect_targets)
        else:
            county_district_graph = update_county_district_graph(dual_graph, county_district_graph,
                                                                 previous_assignment, assignment, False)

        previous_assignment = assignment

        defects.append(calculate_defect(county_district_graph, counties))
    return defects


if __name__ == '__main__':
    def main():
        directory = 'C:/Users/rob/projects/election/rob/'

        x = 0
        ensemble_description = 'TXHD_2101_old_script_1'
        plans = cm.load_plans(directory, ensemble_description, x * 10)

        if False:
            canonical_plans = set()
            number_previous_canonical_plans = 0
            for x in range(0, 22):
                print(f"Plans: {x}")
                plans = cm.load_plans(directory, ensemble_description, x)
                for i, plan in enumerate(plans):
                    if i % 1000 == 0:
                        print(i)

                    canonical_plans.add(cm.calculate_plan_hash(plan))

                number_canonical_plans = len(canonical_plans)
                number_new_canonical_plans = number_canonical_plans - number_previous_canonical_plans
                print(
                    f"Number Canonical Plans: {number_new_canonical_plans} {number_canonical_plans} {number_canonical_plans / ((x + 1) * 10000)}")
                number_previous_canonical_plans = number_canonical_plans

        if False:
            calculate_defects()

        if False:
            defects = []

            for i, plan in enumerate(plans):
                if i % 1000 == 0:
                    print(i)

                assignment = cm.build_assignment(dual_graph, plan)
                whole_targets, intersect_targets = extract_defect_targets(graph)
                county_district_graph = build_county_district_graph(dual_graph, assignment, whole_targets,
                                                                    intersect_targets)
                defects.append(calculate_defect(county_district_graph, counties))

        if True:
            chamber = 'TXHD'
            settings = build_proposed_plan_settings(chamber, 2176)
            # settings = build_TXSN_random_seed_settings(),

            number_steps = 500000  # 300  #

            suffix = '2176_Reduced_Test'
            ensemble_number = 1
            ensemble_description = cm.build_ensemble_description(chamber, suffix, ensemble_number)
            output_directory = cm.build_ensemble_directory(directory, ensemble_description)

            chain, elections = build_chain(chamber, directory, settings, number_steps)
            run_chain(chain, elections, output_directory)

    main()
