import networkx as nx
import networkx.algorithms.bipartite.matching as ma
import numpy as np
from typing import Callable, Iterable, Any
from itertools import chain, product
import pandas as pd
import random
import pickle
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import shapely as sh
import gzip
import csv

import common as cm
import simulation as si
import proposed_plans as pp
import data_transform as dt
from timer import Timer


def load_redistricting_data(directory: str) -> pd.DataFrame:
    redistricting_data_directory = pp.build_redistricting_data_directory(directory)
    data = pd.read_parquet(
        f'{redistricting_data_directory}redistricting_data_nodes_TX_nodes_TX_2020_raw.parquet')
    data['cnty'] = [x[-3:] for x in data['cnty']]
    data['geoid'] = [x[2:] for x in data['geoid']]
    data['polygon'] = data['polygon'].apply(sh.wkt.loads)
    data.set_index('geoid', inplace=True, drop=False)
    return data


def assign_2010_districts(chamber: str, data: pd.DataFrame) -> None:
    census_chamber_name = dt.get_census_chamber_name(chamber)
    data['district'] = [int(x) for x in data[census_chamber_name]]


def assign_plan_districts(chamber: str, directory: str, data: pd.DataFrame, plan: int,
                          plan_geoid_column: str, plan_district_column: str) -> pd.DataFrame:
    plan_assignments = pd.read_csv(f'{pp.build_plan_path(chamber, directory, plan)}')
    plan_assignments[plan_geoid_column] = [str(x)[2:] for x in plan_assignments[plan_geoid_column]]
    plan_assignments.rename(columns={plan_district_column: 'district'}, inplace=True)
    plan_assignments.set_index(plan_geoid_column, inplace=True)
    return data.join(plan_assignments, how='inner')


def build_node_data(data: pd.DataFrame, geographic_unit: str) -> pd.DataFrame:
    def determine_node_ids(data: pd.DataFrame, geographic_unit: str) -> list[str]:
        def determine_whole_counties(data: pd.DataFrame):
            county_districts = set(zip(data['cnty'], data['district']))
            county_groups = cm.groupby_project(list(county_districts), lambda x: x[0], lambda x: x[1])
            return {x for x, y in county_groups if len(y) == 1}

        def determine_non_whole_geographic_units(data: pd.DataFrame, geographic_unit: str) -> set[str]:
            level_districts = set(zip(data[geographic_unit], data['district']))
            level_groups = cm.groupby_project(list(level_districts), lambda x: x[0], lambda x: x[1])
            return {x for x, y in level_groups if len(y) > 1}

        def build_node_id(county: str, geographic_unit: str, district: int, whole_counties: set[str],
                          non_whole_levels: set[str]) -> str:
            if county in whole_counties:
                return county
            elif geographic_unit in non_whole_levels:
                return f'{geographic_unit}_{district}'
            else:
                return geographic_unit

        whole_counties = determine_whole_counties(data)
        non_whole_levels = determine_non_whole_geographic_units(data, geographic_unit)
        return [build_node_id(x, y, z, whole_counties, non_whole_levels) for x, y, z in
                zip(data['cnty'], data[geographic_unit], data['district'])]

    def handle_non_numeric_columns(group, state: dict[str, Any]):
        group_number = state['group_number']
        if group_number % 1000 == 0:
            print(f"Building Node: {group_number}")
        state['group_number'] = group_number + 1

        results = {}
        column_types = group.dtypes
        for column in group.columns:
            column_type = column_types[column]
            if column == 'geoid':
                results['node_geoids'] = ','.join([str(x) for x in group[column]])
                results['geoid'] = group['node_id'].iloc[0]
            elif column == 'district':
                results['node_districts'] = ','.join([str(x) for x in group[column].unique()])
                results['district'] = group['district'].iloc[0]
            elif column == 'polygon':
                results['polygon'] = sh.ops.unary_union(group['polygon'])
            elif column_type == 'object':
                results[column] = group[column].iloc[0]
        return results

    data['node_id'] = determine_node_ids(data, geographic_unit)
    grouped_by_node_ids = data.groupby('node_id')
    state = {'group_number': 0}
    grouped_data = pd.DataFrame(
        list(grouped_by_node_ids.apply(lambda x: handle_non_numeric_columns(x, state))))
    grouped_data.set_index('geoid', drop=False, inplace=True)
    grouped_data_sum = grouped_by_node_ids.sum()
    node_data = grouped_data.join(grouped_data_sum, how='inner', rsuffix='_right')
    node_data.drop(columns=['district_right'], inplace=True)
    return node_data


def build_seeds_working_directory(directory: str) -> str:
    return f'{cm.build_seeds_directory(directory)}working/'


def build_node_data_path(seeds_working_directory: str, seed_filename_prefix: str) -> str:
    return f'{seeds_working_directory}{seed_filename_prefix}.parquet'


def save_node_data(directory: str, seed_filename_prefix: str, node_data: pd.DataFrame) -> None:
    seeds_working_directory = build_seeds_working_directory(directory)

    node_data['polygon'] = [x.wkt for x in node_data['polygon']]
    node_data_path = build_node_data_path(seeds_working_directory, seed_filename_prefix)
    node_data.to_parquet(node_data_path, index=False)
    # node_data.to_csv(f'{seeds_working_directory}{stem}.csv', index=False, quoting=csv.QUOTE_ALL)


def build_seed_filename_prefix(chamber: str, geographic_unit: str, suffix: int) -> str:
    return f'TX_2020_{geographic_unit}_{chamber}_{suffix}'


def save_2010_node_data(chamber: str, directory: str, geographic_unit: str) -> None:
    data = load_redistricting_data(directory)
    assign_2010_districts(chamber, data)
    node_data = build_node_data(data, geographic_unit)
    seed_filename_prefix = build_seed_filename_prefix(chamber, geographic_unit, 2010)
    save_node_data(directory, seed_filename_prefix, node_data)


def save_plan_node_data(chamber: str, directory: str, geographic_unit: str, plan: int) -> None:
    data = load_redistricting_data(directory)
    plan_geoid_column = 'SCTBKEY'
    plan_district_column = 'DISTRICT'
    data = assign_plan_districts(chamber, directory, data, plan, plan_geoid_column, plan_district_column)
    node_data = build_node_data(data, geographic_unit)
    seed_filename_prefix = build_seed_filename_prefix(chamber, geographic_unit, plan)
    save_node_data(directory, seed_filename_prefix, node_data)


def load_node_data(directory: str, seed_filename_prefix: str) -> pd.DataFrame:
    seeds_working_directory = build_seeds_working_directory(directory)
    node_data_path = build_node_data_path(seeds_working_directory, seed_filename_prefix)
    node_data = pd.read_parquet(node_data_path)
    node_data['polygon'] = node_data['polygon'].apply(sh.wkt.loads)
    node_data.set_index('geoid', drop=False, inplace=True)
    return node_data


def build_dual_graph(chamber: str, node_data: pd.DataFrame) -> nx.Graph:
    intersections = set()
    for i, (x_geoid, x_polygon, y_geoid, y_polygon) in \
            enumerate((x_geoid, x_polygon, y_geoid, y_polygon)
                      for x_geoid, x_polygon in zip(node_data['geoid'], node_data['polygon'])
                      for y_geoid, y_polygon in zip(node_data['geoid'], node_data['polygon']) if x_geoid < y_geoid):
        if i % 1000000 == 0:
            print("Determining Intersections: {i}")

        if x_polygon.intersects(y_polygon):
            intersection = x_polygon.intersection(y_polygon)
            perimeter = intersection.boundary.length
            if perimeter > .01:
                intersections.add((x_geoid, y_geoid))

    seats_column = f'seats_{dt.get_census_chamber_name(chamber)}'
    node_attr = ['geoid', 'county', 'district', 'total_pop', seats_column, 'aland', 'perim', 'node_geoids',
                 'node_districts']

    graph = nx.from_edgelist(intersections)
    selected_node_data = node_data.filter(items=node_attr, axis='columns')
    nx.set_node_attributes(graph, selected_node_data.to_dict('index'))

    # TODO: check for number of connected components and raise error
    return graph


def add_new_districts(graph: nx.Graph, current_number_districts: int, new_districts: int, node_data: pd.DataFrame) \
        -> None:
    def get_sorted_connected_components(graph: nx.Graph) -> list[tuple[Any, ...]]:
        return sorted([tuple(x) for x in nx.connected_components(graph)], key=lambda x: len(x), reverse=True)

    def get_district_subgraph(graph: nx.Graph, district: int) -> nx.Graph:
        return nx.subgraph_view(graph, lambda x: graph.nodes[x]['district'] == district)

    def get_district_sorted_connected_components(graph: nx.Graph, district: int) -> list[tuple[Any, ...]]:
        return get_sorted_connected_components(get_district_subgraph(graph, district))

    # Create new districts starting at nodes with high population
    new_district_starts = node_data.nlargest(10 * new_districts, 'total_pop').index.tolist()
    next_district = current_number_districts + 1
    while new_districts > 0:
        # get most populous remaining node, make it a new district
        # check if this disconnected its old district.  If so, undo and try next node.
        n = new_district_starts.pop(0)
        D_old = graph.nodes[n]['district']
        graph.nodes[n]['district'] = next_district
        comp = get_district_sorted_connected_components(graph, D_old)
        if len(comp) == 1:
            # success
            next_district += 1
            new_districts -= 1
        else:
            # fail - disconnected old district - undo and try again
            graph.nodes[n]['district'] = D_old


def build_country_district_graph(chamber: str, dual_graph: nx.Graph) -> nx.Graph:
    # Create the county-district bi-partite adjacency graph.
    # This graph has 1 node for each county and district &
    # an edge for all (county, district) that intersect (share land).
    # It is an efficient tool to track map defect and other properties.
    graph = nx.Graph()

    seats_column = f'seats_{dt.get_census_chamber_name(chamber)}'

    counties = set()
    for n, data in dual_graph.nodes(data=True):
        district = data['district']
        graph.add_node(district)  # adds district node if not already present
        # A.nodes[D]['polsby_popper'] = 0
        for k in ['total_pop']:  # 'aland', 'perim']:
            try:
                graph.nodes[district][k] += data[k]  # add to attribute if exists
            except BaseException:
                graph.nodes[district][k] = data[k]  # else create attribute

        county = data['county']
        counties.add(county)
        graph.add_node(county)  # adds county node if not already present
        for k in ['total_pop', seats_column]:
            try:
                graph.nodes[county][k] += data[k]  # add to attribute if exists
            except BaseException:
                graph.nodes[county][k] = data[k]  # else create attribute

        graph.add_edge(county, district)  # create edge

    # get defect targets
    for county in counties:
        graph.nodes[county]['whole_target'] = int(np.floor(graph.nodes[county][seats_column]))
        graph.nodes[county]['intersect_target'] = int(np.ceil(graph.nodes[county][seats_column]))

    return graph


def save_graph_compressed_json(graph: nx.Graph, output_path: str) -> None:
    cm.save_all_bytes(gzip.compress(bytes(nx.jit_data(graph, indent=4), 'utf-8')), output_path)


def determine_isolated_edges(graph: nx.Graph, isolated_counties: Iterable[str]) -> list[tuple[str, str]]:
    edges = []
    for key, node in graph.nodes.items():
        if node["county"] in isolated_counties:
            for edge_x, edge_y in graph.edges(key):
                if graph.nodes[edge_x]['county'] != graph.nodes[edge_y]['county']:
                    edges.append((edge_x, edge_y))
    return edges


def remove_isolated_county_edges(dual_graph: nx.Graph, county_district_graph: nx.Graph,
                                 minimum_whole_counties: int) -> None:
    counties = si.extract_counties(county_district_graph)

    isolated_counties = si.determine_isolated_counties(county_district_graph, counties, minimum_whole_counties)
    print(f"Isolated Counties: {sorted(isolated_counties)}")

    edges = determine_isolated_edges(dual_graph, isolated_counties)
    dual_graph.remove_edges_from(edges)


def save_reduced_dual_graph(directory: str, reduced_graph_filename_prefix: str, dual_graph: nx.Graph,
                            county_district_graph: nx.Graph, minimum_whole_counties: int):
    remove_isolated_county_edges(dual_graph, county_district_graph, minimum_whole_counties)
    seeds_working_directory = build_seeds_working_directory(directory)
    nx.write_gpickle(dual_graph, f'{seeds_working_directory}{reduced_graph_filename_prefix}.gpickle')


def build_region_plans_path(ensemble_directory: str, region: str) -> str:
    return f'{ensemble_directory}/{region}_plans.npz'


def verify_region_district_uniqueness(region_plans_lookup: dict[str, np.ndarray]) -> None:
    for region, region_plans in region_plans_lookup.items():
        print(region)
        previous_unique_districts = None
        for plan in region_plans:
            unique_districts = set(plan)
            if previous_unique_districts is None:
                print(f"Unique Districts: {len(unique_districts)}")
                previous_unique_districts = unique_districts
            elif not unique_districts == previous_unique_districts:
                print('Different Districts')


def build_region_indices_path(ensemble_directory: str) -> str:
    return f'{ensemble_directory}/region_indices_lookup.pickle'


def calculate_defects(dual_graph: nx.Graph, counties: set[str], whole_targets: dict[str, int],
                      intersect_targets: dict[str, int], plans: np.ndarray):
    defects = []
    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(i)

        plan_defects = calculate_plan_defect(dual_graph, counties, whole_targets, intersect_targets, plan)

        defects.append(plan_defects)
    return defects


def calculate_plan_defect(dual_graph: nx.Graph, counties: set[str], whole_targets: dict[str, int],
                          intersect_targets: dict[str, int], plan):
    assignment = cm.build_assignment(dual_graph, plan)
    county_district_graph = si.build_county_district_graph(dual_graph, assignment, whole_targets, intersect_targets)

    if counties is None:
        counties = si.extract_counties(county_district_graph)

    return si.calculate_defect(county_district_graph, counties)


def calculate_plans_defects(dual_graph: nx.Graph, county_district_graph: nx.Graph, plans: np.ndarray) -> list[int]:
    whole_targets, intersect_targets = si.extract_defect_targets(county_district_graph)
    counties = si.extract_counties(county_district_graph)
    defects = calculate_defects(dual_graph, counties, whole_targets, intersect_targets, plans)
    print(f"Defect Groups: {cm.count_groups(defects)}")
    return defects


def save_unique_region_plans(directory: str, ensemble_description: str, dual_graph: nx.Graph,
                             isolated_counties: set[str]) -> None:
    nodes_by_region = cm.to_dict(
        cm.groupby_project([(y['county'], x)
                            for x, y in dual_graph.nodes().items() if y['county'] in isolated_counties],
                           lambda x: x[0], lambda x: x[1]))
    nodes_by_region['Remaining'] = [x for x, y in dual_graph.nodes().items() if
                                    y['county'] not in isolated_counties]
    nodes_by_region = {x: sorted(y) for x, y in nodes_by_region.items()}
    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
    for region, nodes in nodes_by_region.items():
        print(f'{region} {len(nodes)} {nodes}')
        cm.save_vector_csv(f'{ensemble_directory}{region}_nodes.csv', nodes)
    print(f"Total Number Nodes: {sum([len(x) for x in nodes_by_region.values()])}")

    node_ids_to_index = si.build_node_id_to_index_from_strings(dual_graph.nodes())
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


def save_region_product_ensemble(chamber: str, reduced_ensemble_directory: str, product_ensemble_directory: str,
                                 dual_graph: nx.Graph, county_district_graph: nx.Graph) -> None:
    whole_targets, intersect_targets = si.extract_defect_targets(county_district_graph)
    counties = si.extract_counties(county_district_graph)

    regions = ["Bexar", "Dallas", "Harris", "Tarrant", "Remaining"]
    region_plans_lookup = {x: cm.load_plans_from_path(build_region_plans_path(reduced_ensemble_directory, x))
                           for x in regions}
    verify_region_district_uniqueness(region_plans_lookup)

    region_indices_lookup = cm.load_pickle(build_region_indices_path(reduced_ensemble_directory))
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

    while current_plan < number_plans:
        if current_plan % 10000 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} {current_plan} Number Collisions: {number_collisions}")

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
            print(f"Defect: {calculate_plan_defect(dual_graph, counties, whole_targets, intersect_targets, plan)}")
            print(len(set(plan)))
            if np.any(plan == 0) or len(set(plan)) != number_districts:
                raise RuntimeError('')

        current_plan += 1

    np.savez_compressed(f'{product_ensemble_directory}/plans_0.npz', np.array(plans))


if __name__ == '__main__':
    def main():
        directory = 'C:/Users/rob/projects/election/rob/'

        chamber = 'TXHD'
        plan = 2176
        settings = cm.build_proposed_plan_simulation_settings(chamber, plan)

        seeds_directory = cm.build_seeds_directory(directory)
        dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)
        county_district_graph = si.load_county_district_graph(directory, settings.country_district_graph_filename)

        minimum_whole_counties = 4
        reduced_ensemble_description = 'TXHD_2176_Reduced_3'

        product_ensemble_description = f'{chamber}_{plan}_product_2'
        product_ensemble_directory = cm.build_ensemble_directory(directory, product_ensemble_description)
        cm.ensure_directory_exists(product_ensemble_directory)

        if True:
            # Build the reduced graph without edges between isolated counties to speed up the simulation
            reduced_graph_filename_prefix = settings.dual_graph_filename.replace('.gpickle', '_Reduced')
            save_reduced_dual_graph(directory, reduced_graph_filename_prefix, dual_graph, county_district_graph,
                                    minimum_whole_counties)

        if False:
            display_unique_plans_defects(ensemble_directory)

        if False:
            # It is assumed at this point that the reduced ensemble has been generated
            # Break up the generated plans into regional plans
            isolated_counties = si.determine_isolated_counties(county_district_graph, counties, minimum_whole_counties)
            save_unique_region_plans(directory, reduced_ensemble_description, dual_graph, isolated_counties)

        if False:
            # Randomly join together regional plans to form product plans
            reduced_ensemble_directory = cm.build_ensemble_directory(directory, reduced_ensemble_description)
            save_region_product_ensemble(chamber, reduced_ensemble_directory, product_ensemble_directory, dual_graph,
                                         county_district_graph)

        if False:
            with Timer(name='load_plans'):
                plans = cm.load_plans(directory, product_ensemble_description, 0)

            defects = calculate_plans_defects(dual_graph, county_district_graph, plans)
            # print(defects)

            np.savez_compressed(f'{product_ensemble_directory}defects.npz', np.array(defects))

        if True:
            # Build the desired ensemble matrices by joining product plans to data
            dt.save_ensemble_matrices(chamber, directory, settings.redistricting_data_filename, dual_graph,
                                      product_ensemble_description)

        if False:
            # Code for generating a seed from 2010 districts or from a plan file
            cm.ensure_directory_exists(build_seeds_working_directory(directory))

            chamber = 'TXHD'
            geographic_unit = 'cntyvtd'

            create_plan_graph = False
            if create_plan_graph:
                plan = 2176
                save_plan_node_data(chamber, directory, geographic_unit, plan)
                seed_filename_prefix = build_seed_filename_prefix(chamber, geographic_unit, plan)
            else:
                save_2010_node_data(chamber, directory, geographic_unit)
                seed_filename_prefix = build_seed_filename_prefix(chamber, geographic_unit, 2010)

            node_data = load_node_data(directory, seed_filename_prefix)
            dual_graph = build_dual_graph(chamber, node_data)

            if not create_plan_graph and chamber == 'USCD':
                add_new_districts(dual_graph, 36, 2, node_data)

            seeds_working_directory = build_seeds_working_directory(directory)

            dual_graph_path_prefix = f'{seeds_working_directory}graph_{seed_filename_prefix}'
            nx.write_gpickle(dual_graph, dual_graph_path_prefix + '.gpickle')
            save_graph_compressed_json(dual_graph, dual_graph_path_prefix + '.json.gz')

            county_district_graph = build_country_district_graph(chamber, dual_graph)

            county_district_graph_path_prefix = f'{seeds_working_directory}adj_{seed_filename_prefix}'
            nx.write_gpickle(county_district_graph, county_district_graph_path_prefix + '.gpickle')
            save_graph_compressed_json(county_district_graph, county_district_graph_path_prefix + '.json.gz')


    main()

# TODO Translate graph making code
