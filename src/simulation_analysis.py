import networkx as nx
import numpy as np
from typing import Iterable, Any
from itertools import chain, product
import pandas as pd
import random
import pickle
from datetime import datetime

import simulation as si
import common as cm
import proposed_plans as pp
import data_transform as dt
import utilities as ut
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


def build_region_plans_path(ensemble_directory, region):
    return f'{ensemble_directory}/{region}_plans.npz'


def load_filtered_redistricting_data(directory: str, redistricting_data_filename: str) -> pd.DataFrame:
    redistricting_data_directory = pp.build_redistricting_data_directory(directory)
    node_data = pd.read_parquet(f'{redistricting_data_directory}{redistricting_data_filename}')
    si.fix_election_columns_text(node_data)
    # print(len(node_data))
    node_data['black_o17_sum'] = dt.black_sum(node_data)
    node_data['black_hisp_o17_sum'] = dt.black_hisp_sum(node_data)
    filtered_node_data = node_data.filter(items=[
        'geoid',
        'President_2020_general_D_Biden', 'President_2020_general_R_Trump',
        'USSen_2020_general_D_Hegar', 'USSen_2020_general_R_Cornyn',
        'o17_hisp_pop', 'black_o17_sum', 'black_hisp_o17_sum', 'o17_pop'
    ])
    filtered_node_data.set_index('geoid', inplace=True)
    return filtered_node_data


def verify_uniqueness(region_plans_lookup):
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


if __name__ == '__main__':
    def main():
        chamber = 'TXHD'
        directory = 'C:/Users/rob/projects/election/rob/'
        settings = si.build_proposed_plan_settings(chamber, 2176)

        seeds_directory = cm.build_seeds_directory(directory)
        networkX_graph = nx.read_gpickle(seeds_directory + settings.networkX_graph_filename)
        county_district_graph = si.load_county_district_graph(directory, settings.country_district_graph_filename)
        counties = si.extract_counties(county_district_graph)
        isolated_counties = si.determine_isolated_counties(county_district_graph, counties, 4)

        if False:
            county_defects = si.calculate_county_defects(county_district_graph, counties)
            remove_isolated_county_edges(networkX_graph, county_district_graph, 4)

        if False:
            ensemble_description = 'TXHD_2176_Reduced_1'
            plans = cm.load_plans_from_files(directory, ensemble_description, range(0, 50))
            print(len(plans))

            unique_plans = cm.determine_unique(plans)
            print(f"Number Unique Plans: {len(unique_plans)}")

            ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
            np.savez_compressed(f'{ensemble_directory}/unique_plans.npz', np.array(unique_plans))

        if False:
            nodes_by_region = cm.to_dict(
                cm.groupby_project([(y['county'], x)
                                    for x, y in networkX_graph.nodes().items() if y['county'] in isolated_counties],
                                   lambda x: x[0], lambda x: x[1]))
            nodes_by_region['Remaining'] = sorted([x for x, y in networkX_graph.nodes().items() if
                                                   y['county'] not in isolated_counties])
            nodes_by_region = {x: sorted(y) for x, y in nodes_by_region.items()}

            ensemble_description = 'TXHD_2176_Reduced_1'
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

            # for x, y in components_lookup.items():
            #     print(f"{x} {y}")

            plans = cm.load_plans_from_file(directory, ensemble_description, 'unique_plans.npz')
            print(f'{len(plans)} {max(plans[0])}')
            for region, region_indices in region_indices_lookup.items():
                region_plans = plans[:, region_indices]
                unique_region_plans = cm.determine_unique(region_plans)
                print(f'{region} {np.shape(region_plans)} {np.shape(unique_region_plans)}')
                np.savez_compressed(build_region_plans_path(ensemble_directory, region), unique_region_plans)

        if False:
            regions = ["Bexar", "Dallas", "Harris", "Tarrant", "Remaining"]
            ensemble_description = 'TXHD_2176_Reduced_1'
            ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
            region_plans_lookup = {x: cm.load_plans_from_path(build_region_plans_path(ensemble_directory, x))
                                   for x in regions}

            # verify_uniqueness(region_plans_lookup)
            # for region, plans in region_plans_lookup.items():
            #    districts = sorted(list({x for x in plans[0]}))
            #    print(f"Region: {region} {districts}")

            region_indices_lookup = cm.load_pickle(build_region_indices_path(ensemble_directory))
            min_index = min([min(x) for x in region_indices_lookup.values()])
            max_index = max([max(x) for x in region_indices_lookup.values()])
            print(f"{min_index} {max_index}")

            output_ensemble_description = 'TXHD_2176_product_1'
            output_ensemble_directory = cm.build_ensemble_directory(directory, output_ensemble_description)
            ut.ensure_directory_exists(output_ensemble_directory)

            verify = False
            number_districts = cm.get_number_districts(chamber)
            plan_hashes = set()
            number_plans = 1000000 # 100  #
            current_plan = 0
            number_collisions = 0
            plans = np.zeros((number_plans, max_index + 1), dtype='uint8')
            while current_plan < number_plans:
                if current_plan % 10000 == 0:
                    print(f"{datetime.now().strftime('%H:%M:%S')} {current_plan} Number Collisions: {number_collisions}")

                for region, region_indices in region_indices_lookup.items():
                    region_plan = random.choice(region_plans_lookup[region])
                    plans[current_plan, region_indices] = region_plan
                    #plan.put(region_indices, region_plan)
                    #plan = plans[current_plan]
                    #print(f'{region} {plan} {np.count_nonzero(plan != 0)} {len(region_plan)}')

                plan_hash = cm.calculate_plan_hash(plans[current_plan])
                if plan_hash in plan_hashes:
                    number_collisions += 1
                    continue

                plan_hashes.add(plan_hash)

                if verify:
                    plan = plans[current_plan]
                    if np.any(plan == 0) or len(set(plan)) != number_districts:
                        raise RuntimeError('')

                current_plan += 1

            np.savez_compressed(f'{output_ensemble_directory}/plans_0.npz', np.array(plans))

        if True:
            ensemble_description = 'TXHD_2176_product_1'
            with Timer(name='load_plans'):
                plans = cm.load_plans(directory, ensemble_description, 0)
            print(plans[0:5])
            print(len(cm.count_groups(plans[0])))
            print(len(cm.count_groups(plans[1])))
            print(len(plans))

            defects = si.calculate_defects(networkX_graph, county_district_graph, plans[0:10000])
            print(defects)

            defect_groups = cm.count_groups(defects)
            print(defect_groups)

        if False:
            node_data = load_filtered_redistricting_data(directory, settings.redistricting_data_filename)


    main()
