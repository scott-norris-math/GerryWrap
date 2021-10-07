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
from addict import Dict
import Math

import common as cm
import utilities as ut
import data_transform as dt


def build_canonical_partition_list(partition: Partition):
    partition_assignment = [(x, partition.assignment[x]) for x in partition.graph.nodes]
    return dt.build_canonical_assignments_list(partition_assignment)


def build_canonical_plan_from_partition(node_id_to_index: dict[str, int], partition: Partition):
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


def load_graph_with_geometry(directory: str, dual_graph_filename: str, redistricting_data_filename: str,
                             use_older_data_join_method: bool) -> Graph:
    # Load up seed
    # Have a routine to attach the data to the networkx graph
    seeds_directory = f"{directory}seeds/"

    if use_older_data_join_method:
        networkXGraph = nx.read_gpickle(seeds_directory + "graph_TX_2020_cntyvtd_sldu_contract_S2101.gpickle")
    else:
        networkXGraph = nx.read_gpickle(seeds_directory + dual_graph_filename)

    if use_older_data_join_method:
        county_vtd_data = pd.read_csv(seeds_directory + "nodes_TX_2020_cntyvtd_sldu.csv")
        county_vtd_data['geometry'] = county_vtd_data['polygon'].apply(wkt.loads)
        county_vtd_geodata = gpd.GeoDataFrame(county_vtd_data, geometry='geometry')
        county_vtd_geodata.geometry = county_vtd_geodata.geometry.buffer(0)

        county_data = pd.read_csv(seeds_directory + "nodes_TX_2020_cnty_sldu.csv")
        county_data['geometry'] = county_data['polygon'].apply(wkt.loads)
        county_geodata = gpd.GeoDataFrame(county_data, geometry='geometry')
        county_geodata.geometry = county_geodata.geometry.buffer(0)

        county_geodata['geoid'] = county_geodata['geoid'].astype('str').apply(lambda p: p.zfill(3))
        county_vtd_geodata['geoid'] = county_vtd_geodata['geoid'].astype('str').apply(lambda p: p.zfill(9))

        geodata = pd.concat([county_vtd_geodata, county_geodata]).drop_duplicates().reset_index(drop=True)
    else:
        county_vtd_data = pd.read_parquet(
            seeds_directory + f"../redistricting_data/{redistricting_data_filename}")

        county_vtd_data['geometry'] = county_vtd_data['polygon'].apply(wkt.loads)
        county_vtd_geodata = gpd.GeoDataFrame(county_vtd_data, geometry='geometry')
        county_vtd_geodata.geometry = county_vtd_geodata.geometry.buffer(0)

        county_vtd_geodata.rename(columns={
            "USSen_2020_D_Hegar_general": "USSen_2020_general_D_Hegar",
            "USSen_2020_R_Cornyn_general": "USSen_2020_general_R_Cornyn",
            "President_2020_D_Biden_general": "President_2020_general_D_Biden",
            "President_2020_R_Trump_general": "President_2020_general_R_Trump"}, inplace=True)
        geodata = county_vtd_geodata.reset_index(drop=True)

    graph = Graph(networkXGraph, geometry=county_vtd_geodata.geometry)
    graph.join(geodata, columns=["o17_hisp_pop", "o17_pop", "geometry"], right_index="geoid")

    # TODO: probably should have been used when forming the graph - try with TXSN
    graph.geometry = geodata.geometry
    graph.geometry.index = geodata.geoid

    # TODO: probably only needed for old method
    filtered_df = geodata[geodata['geoid'].isin(graph.nodes().keys())]
    graph.geometry = filtered_df.geometry
    graph.geometry.index = filtered_df.geoid

    return graph


def build_geographic_partition(graph, is_plan_proposal):
    if not is_plan_proposal:
        district_column = 'sldu'
    else:
        district_column = 'district'

    return GeographicPartition(graph, assignment=district_column, updaters=[])


def join_data_to_dual_graph(directory, networkXGraph):
    df = pd.read_csv(directory + "/nodes_TX_2020_cntyvtd_sldu.csv")

    # gdf.set_geometry("polygon")

    df['geometry'] = df['polygon'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.geometry = gdf.geometry.buffer(0)

    # Replace geoids with strings in gdf as well
    gdf['geoid'] = gdf['geoid'].astype('str').apply(lambda p: p.zfill(9))

    # Now, OTHER data file
    # This one has counties
    #    We need this becaue our graph has BOTH VTDs and counties (collapsed into one node)
    df_county = pd.read_csv(directory + "/nodes_TX_2020_cnty_sldu.csv")
    df_county['geometry'] = df_county['polygon'].apply(wkt.loads)
    gdf_county = gpd.GeoDataFrame(df_county, geometry='geometry')
    gdf_county.geometry = gdf_county.geometry.buffer(0)

    # Geoids are strings in graph, but numbers in dataframe.
    #
    # Replace gdfCY geoids with strings
    gdf_county['geoid'] = gdf_county['geoid'].astype('str').apply(lambda p: p.zfill(3))

    # Join the two dataframes
    new_df = pd.concat([gdf, gdf_county]).drop_duplicates().reset_index(drop=True)

    # We need to join
    # Converts the "networkX" structure to the gerrychain structure
    graph = Graph(networkXGraph, geometry=gdf.geometry)
    graph.join(new_df, columns=["President_2020_general_D_Biden", "President_2020_general_R_Trump"],
               right_index="geoid")
    graph.join(new_df, columns=["USSen_2020_general_D_Hegar", "USSen_2020_general_R_Cornyn"], right_index="geoid")
    graph.geometry = new_df.geometry
    graph.geometry.index = new_df.geoid

    # We are working with full NEWDF
    graph.join(new_df, columns=["o17_hisp_pop", "o17_pop", "geometry"], right_index="geoid")
    all_o17_columns = [col for col in new_df.columns if (col.startswith('o17') and not col.startswith('o17_nonhisp'))]
    nonhisp_o17_columns = [col for col in new_df.columns if col.startswith('o17_nonhisp')]

    # All black
    black_o17_columns = [col for col in all_o17_columns if 'black' in col]
    new_df['black_o17_sum'] = new_df[black_o17_columns].sum(axis=1)

    # NON-Hispanic black
    black_nonhisp_o17_columns = [col for col in nonhisp_o17_columns if 'black' in col]
    new_df['black_nonhisp_o17_sum'] = new_df[black_nonhisp_o17_columns].sum(axis=1)
    graph.join(new_df, columns=["black_o17_sum", "black_nonhisp_o17_sum"], right_index="geoid")
    # graph.nodes['001']

    l1 = list(graph.nodes().keys())

    l2 = list(new_df['geoid'])

    linter = list(set(l1) & set(l2))
    # [len(l1), len(l2), len(linter)]

    test_df = new_df[new_df['geoid'].isin(l1)]

    graph.geometry = test_df.geometry
    graph.geometry.index = test_df.geoid

    # testdf.geometry.index
    return graph


def run_chain():
    directory = "seeds/"

    # networkXGraph = nx.read_gpickle(directory + "/TX_2020_cntyvtd_sldu_seed_1000000_graph.gpickle")
    networkXGraph = nx.read_gpickle(directory + "graph_TX_2020_cntyvtd_TXHD_2101.gpickle")
    graph = join_data_to_dual_graph(directory, networkXGraph)

    updaters = build_updaters()

    initial_partition = GeographicPartition(graph, assignment='sldu', updaters=updaters)

    # Now we will really generate many plans and save data
    epsilon = .02
    proposal = partial(recom,
                       pop_col="total_pop",
                       pop_target=ideal_population,
                       epsilon=epsilon,
                       node_repeats=2)

    county_district_graph = nx.read_gpickle(directory + "adj_TX_2020_cntyvtd_TXHD_2101.gpickle")
    state = {'graph': county_district_graph}
    accept_function = build_accept_function(state, 0)  # TODO: implement i inside accept_function

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2 * len(initial_partition["cut_edges"])
    )

    # split_bound = constraints.refuse_new_splits("countysplits")
    split_bound = constraints.UpperBound(
        lambda p: len(p["countysplits"]),
        len(initial_partition["countysplits"])
    )

    number_steps = 50000
    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            # District populations must stay within % of equality
            constraints.within_percent_of_ideal_population(next_seed_partition, epsilon),
            compactness_bound,
        ],
        accept=accept.always_accept,  # accept_function
        initial_state=initial_partition,
        total_steps=number_steps
    )

    output_directory = "./Outputs_TXSN_GerryRoutines/"
    ut.ensure_directory_exists(output_directory)

    run_chain_impl(chain, initial_partition, output_directory)


def build_updaters():
    updaters = {"population": updaters.Tally("total_pop", alias="population"),
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
    updaters.update({election.name: election for election in elections})

    return updaters


def run_chain_impl(chain: MarkovChain, output_directory: str) -> None:
    node_id_to_index = {x: i for i, x in enumerate(sorted(chain.initial_state.assignment))}
    outfile = open(f"{output_directory}/columns_map.pickle", 'wb')
    pickle.dump(node_id_to_index, outfile)
    outfile.close()

    data = initialize_data(chain.total_steps)

    plans_array = None
    plans_output_size = 100000
    step_number = 0
    for partition in chain:
        if step_number % 10000 == 0:
            partition.plot()
            plt.savefig(f'{map_directory}mapstep{step_number}.png')
            plt.show()

        if step_number % 100 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} Step: {step_number}")

            print("Partition Assignments")
            display_partition(initial_partition, partition)

            print("Partition List")
            partition_as_list = build_canonical_partition_list(partition)
            print(partition_as_list)

        plan_index = step_number % plans_output_size
        if plan_index == 0:
            if step_number > 0:
                file_index = (step_number // plans_output_size) - 1
                np.savez_compressed(f"{newdir}/plans_{file_index}.npz", plans_array)
            plans_array = np.ndarray([plans_output_size, number_nodes], np.uint8)

        current_plan = build_plan(partition)
        for district_index, node_set in enumerate(current_plan):
            district_index_npu8 = np.uint8(district_index)
            for node_index in node_set:
                plans_array[plan_index, node_index] = district_index_npu8

        update_data(partition, data, step_number)

    save_data(output_directory, data)
    save_compressed_data(output_directory, data)
    save_partial_plans(plans_array, plans_output_size, step_number)


def initialize_data(number_steps: int) -> Dict:
    data = Dict()

    data.election_data = pd.DataFrame([], range(0, number_steps), [election.name for election in elections])

    data.HVAP_data = []
    data.BVAP_data = []
    data.BHVAP_data = []

    # this will store the partisan gini scores for each plan, and each election
    data.gini = []

    # Population, cut_edges
    data.pop_vec = []
    data.cut_vec = []

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

    return data


def update_data(partition: Partition, data: Dict, step_number: int) -> None:
    data.pop_vec.append(list(partition["population"].values()))
    data.cut_vec.append(len(partition["cut_edges"]))

    for election in elections:
        election_data = partition[election.name]
        data.votes[elect].append(election_data.percents("Democratic"))
        data.mms[-1].append(mean_median(election_data))
        data.egs[-1].append(efficiency_gap(election_data))
        data.hmss[-1].append(election_data.wins("Democratic"))

    data.election_data.loc[step_number] = {election.name: partition[election.name].percents("Democratic") for election
                                           in elections}

    data.gini.append([metrics.partisan_gini(partition[election.name]) for election in elections])

    data.HVAP_data.append(
        partition["o17_hisp_pop"][key] / partition["o17_pop"][key] for key in partition["population"])

    data.BVAP_data.append(
        partition["black_o17_sum"][key] / partition["o17_pop"][key] for key in partition["population"])

    data.BHVAP_data.append(
        (partition["o17_hisp_pop"][key] + partition["black_nonhisp_o17_sum"][key]) / partition["o17_pop"][key]
        for key in partition["population"])

    data.mms.append([])
    data.egs.append([])
    data.hmss.append([])


def save_data(directory: str, data: Dict) -> None:
    cm.save_vector_csv(f"{directory}hisp_perc.npz", data.HVAP_data)
    cm.save_vector_csv(f"{directory}black_perc.npz", data.BVAP_data)
    cm.save_vector_csv(f"{directory}black_hisp_perc.npz", data.BHVAP_data)
    cm.save_vector_csv(f"{directory}gini.npz", data.gini)
    cm.save_vector_csv(f"{directory}mms.npz", data.mms)
    cm.save_vector_csv(f"{directory}egs.npz", data.egs)
    cm.save_vector_csv(f"{directory}hmss.npz", data.hmss)
    cm.save_vector_csv(f"{directory}pop_vec.npz", data.pop_vec)
    cm.save_vector_csv(f"{directory}cut_edge.npz", data.cut_vec)
    for election in data.elections:
        cm.save_vector_csv(f"{directory}votes_{election.name}.npz", data.votes[j])


def save_compressed_data(directory: str, data: Dict) -> None:
    np.savez(f"{directory}hisp_perc.npz", np.array(data.HVAP_data))
    np.savez(f"{directory}black_perc.npz", np.array(data.BVAP_data))
    np.savez(f"{directory}black_hisp_perc.npz", np.array(data.BHVAP_data))
    np.savez(f"{directory}gini.npz", np.array(data.gini))
    np.savez(f"{directory}mms.npz", np.array(data.mms))
    np.savez(f"{directory}egs.npz", np.array(data.egs))
    np.savez(f"{directory}hmss.npz", np.array(data.hmss))
    np.savez(f"{directory}pop_vec.npz", np.array(data.pop_vec))
    np.savez(f"{directory}cut_edge.npz", np.array(data.cut_vec))
    for election in data.elections:
        np.savez(f"{directory}votes_{election.name}.npz", np.array(data.votes[j]))


def save_partial_plans(plans_array: np.ndarray, plans_output_size: int, step_number: int) -> None:
    plan_index = step_number % plans_output_size
    if plan_index == 0:
        plan_index = plans_output_size
    partial_plans_array = plans_array[0:plan_index].copy()
    file_index = math.ceil(step_number / plans_output_size) - 1
    np.savez_compressed(f"{newdir}/plans_{file_index}.npz", partial_plans_array)


# defect code


def calculate_defect(county_district_graph, county_nodes, verbose=False) -> int:
    # The "county-line" rule prefers minimal county & district splitting. We implement as follows:
    # seats_share = county population / distrinct ideal population
    # Ideally, county should wholly contain floor(seats_share) and intersect ceiling(seats_share) districts
    # Ex: County seats_share=2.4, so it should ideally wholly contain 2 districts and intersect a 3rd.
    # whole_defect = |actual wholly contained - floor(seats_share)|
    # intersect_defect = |actual intersected - ceil(seats_share)|
    # defect = whole_defect + intersect_defect

    # Assume that: You have already identified Cnodes, they are passed
    # Dnodes = bipartite.set(B)  - Cnodes

    total_defect_whole = 0
    total_defect_intersect = 0
    for county_node in county_nodes:
        # Number of whole districts; i.e. Dnodes which are ONLY connected to this county
        number_whole_districts = sum(county_district_graph.degree[district_node] == 1
                                     for district_node in county_district_graph[county_node])
        defect_whole = abs(county_district_graph.nodes[county_node]['whole_target'] - number_whole_districts)
        number_districts_intersected = county_district_graph.degree[county_node]
        defect_intersect = abs(
            county_district_graph.nodes[county_node]['intersect_target'] - number_districts_intersected)

        if verbose:
            print([county_node, number_whole_districts, defect_whole,
                   number_districts_intersected, defect_intersect])

        total_defect_whole += defect_whole
        total_defect_intersect += defect_intersect

    return total_defect_whole + total_defect_intersect


def set_bipartitite_flag(county_district_graph):
    for node_id, node in county_district_graph.nodes.items():
        if isinstance(node_id, str):
            node['bipartite'] = 0
        else:
            node['bipartite'] = 1


def extract_counties(county_district_graph):
    return {node for node, data in county_district_graph.nodes(data=True) if data['bipartite'] == 0}


def extract_defect_targets(county_district_graph):
    counties = extract_counties(county_district_graph)

    whole_targets = {}
    intersect_targets = {}
    for county in counties:
        whole_targets[county] = county_district_graph.nodes[county]['whole_target']
        intersect_targets[county] = county_district_graph.nodes[county]['intersect_target']

    return whole_targets, intersect_targets


def build_county_district_graph(dual_graph, assignment, whole_targets, intersect_targets):
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


def build_accept_function(state, i):
    return lambda x: better_defect_accept(x, state, i)


def better_defect_accept(partition, state, i):
    """Measure defect. If defect is better, accept.
    If defect is worse, random prob of acceptance"""
    # This is the bipartite graph that encodes the "parent"
    county_district_graph = state['graph']

    if partition.parent is None:
        return 1

    # Now we must figure out how to change graph
    parent_assignment = partition.parent.assignment
    assignment = partition.assignment

    copied_graph = update_county_district_graph(partition.graph, county_district_graph, parent_assignment, assignment,
                                                True)

    county_nodes = {node for node, data in county_district_graph.nodes(data=True) if data['bipartite'] == 0}

    old_defect = calculate_defect(county_district_graph, county_nodes)
    new_defect = calculate_defect(copied_graph, county_nodes)

    bound = 1
    k = 3  # severity of penalizing defect
    # If new_bal<old_bal, leave prob(accept)=1
    if partition.parent is not None:
        if new_defect > old_defect:
            bound = np.exp(-k * (new_defect - old_defect))

    accept = np.random.random() < bound

    if i % 5 == 0:
        print([old_defect, new_defect, bound, accept])

    if accept:
        state['graph'] = copied_graph

    return accept


def update_county_district_graph(dual_graph, county_district_graph, old_assignment, new_assignment, copy_graph):
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


# utilities


def display_population_deviations(initial_partition):
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    print(len(initial_partition))
    print(ideal_population)
    mylist = initial_partition["population"]
    numpy.concatenate(mylist.values())
    print((numpy.max([x for x in mylist.values()]) - ideal_population) / ideal_population)
    print((numpy.min([x for x in mylist.values()]) - ideal_population) / ideal_population)


def calculate_size(canonical_plans):
    size = 0
    for plan in canonical_plans:
        for district_group in plan:
            size += sys.getsizeof(district_group)
    return size


if __name__ == '__main__':
    def main():
        directory = 'C:/Users/rob/projects/election/rob/'
        seeds_directory = f"{directory}seeds/"

        dual_graph = nx.read_gpickle(seeds_directory + "graph_TX_2020_cntyvtd_TXHD_2101.gpickle")
        county_district_graph = nx.read_gpickle(seeds_directory + "adj_TX_2020_cntyvtd_TXHD_2101.gpickle")
        set_bipartitite_flag(county_district_graph)

        whole_targets, intersect_targets = extract_defect_targets(county_district_graph)

        counties = extract_counties(county_district_graph)
        print(f"Counties: {len(counties)}")

        x = 0
        ensemble_description = 'ensemble_TXHD_2101_old_script_1'
        plans = cm.load_plans(directory, ensemble_description, x * 10)
        defects = []

        if True:
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
            previous_assignment = None
            for i, plan in enumerate(plans):
                if i % 1000 == 0:
                    print(i)
                assignment = cm.build_assignment(dual_graph, plan)
                if i == 0:
                    county_district_graph = build_county_district_graph(dual_graph, assignment, whole_targets,
                                                                        intersect_targets)
                else:
                    county_district_graph = update_county_district_graph(dual_graph, county_district_graph,
                                                                         previous_assignment, assignment, False)
                previous_assignment = assignment
                defects.append(calculate_defect(county_district_graph, counties, False))

        if False:
            for i, plan in enumerate(plans):
                if i % 1000 == 0:
                    print(i)
                assignment = cm.build_assignment(dual_graph, plan)
                county_district_graph = build_county_district_graph(dual_graph, assignment, whole_targets,
                                                                    intersect_targets)
                defects.append(calculate_defect(county_district_graph, counties, False))

        if False:
            run_chain(chain, None, None)


    main()
