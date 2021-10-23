import networkx as nx
import numpy as np
from typing import Callable, Iterable, Any
from itertools import chain, product
import pandas as pd
import random
import pickle
from datetime import datetime
import networkx.algorithms.bipartite.matching as ma
from scipy import stats
import random
import matplotlib.pyplot as plt

import common as cm
import simulation as si
import proposed_plans as pp
import data_transform as dt
from timer import Timer


def determine_isolated_edges(graph: nx.Graph, isolated_counties: Iterable[str]) -> list[tuple[str, str]]:
    edges = []
    for key, node in graph.nodes.items():
        if node["county"] in isolated_counties:
            for edge_x, edge_y in graph.edges(key):
                if graph.nodes[edge_x]['county'] != graph.nodes[edge_y]['county']:
                    edges.append((edge_x, edge_y))
    return edges


def remove_isolated_county_edges(networkX_graph: nx.Graph, county_district_graph: nx.Graph,
                                 minimum_whole_counties: int) -> None:
    counties = si.extract_counties(county_district_graph)

    isolated_counties = si.determine_isolated_counties(county_district_graph, counties, minimum_whole_counties)
    print(f"Isolated Counties: {sorted(isolated_counties)}")

    edges = determine_isolated_edges(networkX_graph, isolated_counties)
    # print(f'{len(edges)} of {len(networkX_graph.edges())}')
    networkX_graph.remove_edges_from(edges)
    # edges = determine_isolated_edges(networkX_graph, isolated_counties)
    # print(f'{len(edges)} of {len(networkX_graph.edges())}')


def build_region_plans_path(ensemble_directory: str, region: str) -> str:
    return f'{ensemble_directory}/{region}_plans.npz'


def verify_uniqueness(region_plans_lookup: dict[str, np.ndarray]) -> None:
    for region, region_plans in region_plans_lookup.items():
        print(region)
        previous_unique_districts = None
        for row in region_plans:
            unique_districts = set(row)
            if previous_unique_districts is None:
                print(f"Unique Districts: {len(unique_districts)}")
                previous_unique_districts = unique_districts
            elif not unique_districts == previous_unique_districts:
                print('Different Districts')


def build_region_indices_path(ensemble_directory: str) -> str:
    return f'{ensemble_directory}/region_indices_lookup.pickle'


def calculate_defects(dual_graph: nx.Graph, whole_targets: dict[str, int], intersect_targets: dict[str, int],
                      plans: np.ndarray):
    counties = si.extract_counties(county_district_graph)

    defects = []
    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(i)

        plan_defects = calculate_plan_defects(dual_graph, counties, whole_targets, intersect_targets, plan)

        defects.append(plan_defects)
    return defects


def calculate_plan_defects(dual_graph: nx.Graph, counties: set[str], whole_targets: dict[str, int], intersect_targets: dict[str, int], plan):
    assignment = cm.build_assignment(dual_graph, plan)
    county_district_graph = si.build_county_district_graph(dual_graph, assignment, whole_targets, intersect_targets)

    if counties is None:
        counties = si.extract_counties(county_district_graph)

    return si.calculate_defect(county_district_graph, counties)


def calculate_plans_defects(networkX_graph: nx.Graph, county_district_graph: nx.Graph, plans: np.ndarray) -> list[int]:
    whole_targets, intersect_targets = si.extract_defect_targets(county_district_graph)
    defects = calculate_defects(networkX_graph, whole_targets, intersect_targets, plans)
    print(f"Defect Groups: {cm.count_groups(defects)}")
    return defects


def save_unique_region_plans(directory: str, ensemble_description: str) -> None:
    nodes_by_region = cm.to_dict(
        cm.groupby_project([(y['county'], x)
                            for x, y in networkX_graph.nodes().items() if y['county'] in isolated_counties],
                           lambda x: x[0], lambda x: x[1]))
    nodes_by_region['Remaining'] = [x for x, y in networkX_graph.nodes().items() if
                                    y['county'] not in isolated_counties]
    nodes_by_region = {x: sorted(y) for x, y in nodes_by_region.items()}
    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
    for region, nodes in nodes_by_region.items():
        print(f'{region} {len(nodes)} {nodes}')
        cm.save_vector_csv(f'{ensemble_directory}{region}_nodes.csv', nodes)
    print(f"Total Number Nodes: {sum([len(x) for x in nodes_by_region.values()])}")

    node_ids_to_index = si.build_node_id_to_index_from_strings(networkX_graph.nodes())
    region_indices_lookup = {county: [node_ids_to_index[y] for y in node_ids] for county, node_ids in
                             nodes_by_region.items()}
    print(f"Indices in Components: {len(set(sum(region_indices_lookup.values(), [])))}")
    cm.save_pickle(build_region_indices_path(ensemble_directory), region_indices_lookup)

    plans = cm.load_plans_from_file(directory, ensemble_description, 'unique_plans.npz')

    print(f'{len(plans)} {max(plans[0])}')
    for region, region_indices in region_indices_lookup.items():
        region_plans = plans[:, region_indices]
        unique_region_plans = cm.determine_unique_plans(region_plans)
        print(f'{region} {np.shape(region_plans)} {np.shape(unique_region_plans)}')
        np.savez_compressed(build_region_plans_path(ensemble_directory, region), unique_region_plans)


def save_region_product_ensemble() -> None:
    whole_targets, intersect_targets = si.extract_defect_targets(county_district_graph)

    regions = ["Bexar", "Dallas", "Harris", "Tarrant", "Remaining"]
    region_plans_lookup = {x: cm.load_plans_from_path(build_region_plans_path(ensemble_directory, x))
                           for x in regions}
    verify_uniqueness(region_plans_lookup)

    region_indices_lookup = cm.load_pickle(build_region_indices_path(ensemble_directory))
    min_index = min([min(x) for x in region_indices_lookup.values()])
    max_index = max([max(x) for x in region_indices_lookup.values()])
    print(f"{min_index} {max_index}")

    verify = False
    number_districts = cm.get_number_districts(chamber)
    plan_hashes = set()
    number_plans = 1000000  # 100 #
    current_plan = 0
    number_collisions = 0
    plans = np.zeros((number_plans, max_index + 1), dtype='uint8')
    print(list(networkX_graph.nodes()))
    while current_plan < number_plans:
        if current_plan % 10000 == 0:
            print(
                f"{datetime.now().strftime('%H:%M:%S')} {current_plan} Number Collisions: {number_collisions}")

        for region, region_indices in region_indices_lookup.items():
            region_plan = random.choice(region_plans_lookup[region])
            plans[current_plan, region_indices] = region_plan

        plan_hash = cm.calculate_plan_hash(plans[current_plan])
        if plan_hash in plan_hashes:
            number_collisions += 1
            continue

        plan_hashes.add(plan_hash)

        if verify:
            plan = plans[current_plan]
            print(f"Defect: {calculate_plan_defect(networkX_graph, whole_targets, intersect_targets, plan)}")
            print(len(set(plan)))
            if np.any(plan == 0) or len(set(plan)) != number_districts:
                raise RuntimeError('')

        current_plan += 1
    np.savez_compressed(f'{product_ensemble_directory}/plans_0.npz', np.array(plans))


if __name__ == '__main__':
    def main():
        chamber = 'TXHD'
        directory = 'C:/Users/rob/projects/election/rob/'
        settings = cm.build_proposed_plan_simulation_settings(chamber, 2176)

        seeds_directory = cm.build_seeds_directory(directory)
        networkX_graph = nx.read_gpickle(seeds_directory + settings.networkX_graph_filename)
        county_district_graph = si.load_county_district_graph(directory, settings.country_district_graph_filename)
        counties = si.extract_counties(county_district_graph)

        ensemble_description = 'TXHD_2176_Reduced_3'
        ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)

        product_ensemble_description = 'TXHD_2176_product_1'
        product_ensemble_directory = cm.build_ensemble_directory(directory, product_ensemble_description)
        cm.ensure_directory_exists(product_ensemble_directory)

        if False:
            county_defects = si.calculate_county_defects(county_district_graph, counties)
            remove_isolated_county_edges(networkX_graph, county_district_graph, 4)

        if False:
            display_unique_plans_defects(ensemble_directory)

        if False:
            save_unique_region_plans(directory, ensemble_description)

        if False:
            save_region_product_ensemble()

        if False:
            with Timer(name='load_plans'):
                plans = cm.load_plans(directory, product_ensemble_description, 0)

            defects = calculate_plans_defects(networkX_graph, county_district_graph, plans)
            # print(defects)

            np.savez_compressed(f'{product_ensemble_directory}defects.npz', np.array(defects))

        if False:
            dt.save_ensemble_matrices(chamber, directory, settings.redistricting_data_filename, networkX_graph,
                                      product_ensemble_description)


    main()

# TODO Translate graph making code
