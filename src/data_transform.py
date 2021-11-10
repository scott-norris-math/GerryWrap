from os import listdir
from os.path import isfile, exists
from collections import defaultdict
import pandas as pd
from typing import Callable, Iterable
import networkx as nx
import numpy as np
from datetime import datetime

import common as cm
import proposed_plans as pp
import simulation as si
from timer import Timer


def get_census_chamber_name(chamber: str) -> str:
    return {
        'USCD': 'cd',
        'TXSN': 'sldu',
        'TXHD': 'sldl'
    }[chamber]


def merge_plan_data(input_directories: Iterable[str], output_directory: str) -> None:
    masks = ['_perc', 'votes_']
    data_files = [(x, y) for x in input_directories for y in listdir(x)
                  if isfile(x + y) and any(mask in y for mask in masks)]

    file_groups: dict[str, list[str]] = defaultdict(list)

    print(data_files)
    print(file_groups)

    for directory, filename in data_files:
        file_groups[filename].append(directory)

    for filename, directories in file_groups.items():
        print('Merging:' + filename)
        merged_array = cm.load_merged_numpy_csv(filename, directories)
        cm.save_numpy_csv(output_directory + filename, merged_array)


def determine_data_columns(df: pd.DataFrame) -> list[str]:
    race_columns = [col for col in df.columns if (col.startswith('o17'))]
    election_columns = [
        'President_2020_general_D_Biden',
        'President_2020_general_R_Trump',
        'USSen_2020_general_D_Hegar',
        'USSen_2020_general_R_Cornyn'
    ]
    return cm.union(race_columns, election_columns)


def o17_pop(df: pd.DataFrame) -> pd.Series:
    return df['o17_pop']


def total_pop(df: pd.DataFrame) -> pd.Series:
    return df['total_pop']


def hisp_percent(df: pd.DataFrame) -> pd.Series:
    return df['o17_hisp_pop'] / df['o17_pop']


def white_percent(df: pd.DataFrame) -> pd.Series:
    return df['o17_nonhisp_white'] / df['o17_pop']


def non_white_percent(df: pd.DataFrame) -> pd.Series:
    return 1 - white_percent(df)


def black_sum_percent(df: pd.DataFrame) -> pd.Series:
    race_sum = black_sum(df)
    return race_sum / df['o17_pop']


def black_percent(df: pd.DataFrame) -> pd.Series:
    return df['black_o17_sum'] / df['o17_pop']


def black_sum(df: pd.DataFrame) -> pd.Series:
    entire_population_o17_columns = [col for col in df.columns
                                     if (col.startswith('o17') and not col.startswith('o17_nonhisp'))]
    race_pop_columns = [col for col in entire_population_o17_columns if 'black' in col]
    return df[race_pop_columns].sum(axis=1)


def black_hisp_sum_percent(df: pd.DataFrame) -> pd.Series:
    race_sum = black_hisp_sum(df)
    return race_sum / df['o17_pop']


def black_hisp_percent(df: pd.DataFrame) -> pd.Series:
    return df['black_hisp_o17_sum'] / df['o17_pop']


def black_hisp_sum(df: pd.DataFrame) -> pd.Series:
    nonhisp_o17_columns = [col for col in df.columns if col.startswith('o17_nonhisp')]
    race_nonhisp_o17_columns = [col for col in nonhisp_o17_columns if 'black' in col]
    race_pop_columns = ['o17_hisp_pop'] + race_nonhisp_o17_columns
    race_sum = df[race_pop_columns].sum(axis=1)
    return race_sum


def pres20_percent(df: pd.DataFrame) -> pd.Series:
    dem_votes = df['President_2020_general_D_Biden']
    rep_votes = df['President_2020_general_R_Trump']
    return dem_votes / (dem_votes + rep_votes)


def sen20_percent(df: pd.DataFrame) -> pd.Series:
    dem_votes = df['USSen_2020_general_D_Hegar']
    rep_votes = df['USSen_2020_general_R_Cornyn']
    return dem_votes / (dem_votes + rep_votes)


def transform_racial_group_file_prefix(racial_group: str) -> str:
    return {
        'HVAP': "hisp",
        'BVAP': "black",
        'BHVAP': "black_hisp",
        'WVAP': "white",
        'NWVAP': "non_white"
    }[racial_group]


def build_race_filename_prefix(race: str) -> str:
    return race + '_perc'


def build_race_filename_csv(race: str, suffix: str = '') -> str:
    return build_race_filename_prefix(race) + cm.build_suffix(suffix) + '.csv'


def build_election_filename_prefix(election: str) -> str:
    return 'votes_' + election


def build_election_filename_csv(election: str, suffix: str = '') -> str:
    return build_election_filename_prefix(election) + cm.build_suffix(suffix) + '.csv'


def build_statistics_settings() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (build_race_filename_prefix('hisp'), hisp_percent),
        (build_race_filename_prefix('black_hisp'), black_hisp_percent),
        (build_race_filename_prefix('black'), black_percent),
        (build_race_filename_prefix('white'), white_percent),
        (build_race_filename_prefix('non_white'), non_white_percent),
        (build_election_filename_prefix('PRES20'), pres20_percent),
        (build_election_filename_prefix('SEN20'), sen20_percent),
        ('o17_pop', o17_pop),
        ('total_pop', total_pop)
    ]


def build_canonical_assignments_list(assignments: list[tuple[int, int]]) -> list[list[int]]:
    partition_dictionary = defaultdict(list)
    for unit, district in assignments:
        partition_dictionary[district].append(unit)
    partition_list = [sorted(x) for x in partition_dictionary.values()]
    partition_list.sort(key=lambda x: x[0])
    return partition_list


def load_filtered_redistricting_data(directory: str, redistricting_data_filename: str,
                                     additional_columns: list[str] = []) -> pd.DataFrame:
    redistricting_data_directory = pp.build_redistricting_data_directory(directory)
    redistricting_data_path = f'{redistricting_data_directory}{redistricting_data_filename}'
    if redistricting_data_path.endswith('.parquet'):
        node_data = pd.read_parquet(redistricting_data_path)
    elif redistricting_data_path.endswith('.csv'):
        node_data = pd.read_csv(redistricting_data_path)
    else:
        raise RuntimeError('Unknown file extension')

    # redistricting data stored at the census block level has some different column names
    si.fix_election_columns_text(node_data)

    node_data['black_o17_sum'] = black_sum(node_data)
    node_data['black_hisp_o17_sum'] = black_hisp_sum(node_data)
    filtered_node_data = node_data.filter(items=[
        'geoid',
        'President_2020_general_D_Biden', 'President_2020_general_R_Trump',
        'USSen_2020_general_D_Hegar', 'USSen_2020_general_R_Cornyn',
        'o17_hisp_pop', 'black_o17_sum', 'black_hisp_o17_sum', 'o17_pop',
        'o17_nonhisp_white', 'total_pop'
    ] + additional_columns)
    filtered_node_data.set_index('geoid', drop=False, inplace=True)
    filtered_node_data.sort_index(inplace=True)
    return filtered_node_data


def combine_and_fix_redistricting_data_file(directory: str) -> None:
    county_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_sldu.csv')
    county_vtd_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cntyvtd_sldu.csv')

    county_data['geoid'] = county_data['geoid'].astype('str').apply(lambda p: p.zfill(3))
    county_vtd_data['geoid'] = county_vtd_data['geoid'].astype('str').apply(lambda p: p.zfill(9))

    combined = pd.concat([county_data, county_vtd_data])
    output_prefix = f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_cntyvtd_sldu'
    combined.to_parquet(output_prefix + '.parquet', index=False)
    combined.to_csv(output_prefix + '.csv', index=False)


def save_ensemble_matrices(chamber: str, directory: str, redistricting_data_filename: str, graph: nx.Graph,
                           ensemble_description: str, file_numbers: Iterable[int]) -> None:
    node_data = load_filtered_redistricting_data(directory, redistricting_data_filename)
    plans = cm.load_plans_from_files(directory, ensemble_description, file_numbers)

    statistics_settings = build_statistics_settings()
    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
    statistics_paths = {x: f'{ensemble_directory}{x}.npz' for x, _ in statistics_settings}
    statistics_settings = [(x, y) for x, y in statistics_settings if not exists(statistics_paths[x])]
    if len(statistics_settings) == 0:
        print("All ensemble matrices already exist")
        return

    print("Saving ensemble matrices for: ")
    for statistic, _ in statistics_settings:
        print(statistic)

    number_districts = cm.get_number_districts(chamber)
    ensemble_matrices = {statistic: np.zeros((len(plans), number_districts)) for statistic, _ in
                         statistics_settings}

    print(f"Original length of Redistricting Data: {len(node_data)}")
    geoids = list(sorted(graph.nodes()))
    geoids_lookup = set(geoids)
    node_data = node_data[node_data['geoid'].isin(geoids_lookup)]
    for current_plan, plan in enumerate(plans):
        if current_plan % 1000 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} {current_plan}")

        if current_plan == 0:
            print(f"Plan - Min District: {min(plan)} Max District: {max(plan)}")
            number_redistricting_data = len(node_data)
            plan_length = len(plan)
            print(f"Redistricting Data: {number_redistricting_data} Plan: {plan_length}")
            if number_redistricting_data != plan_length:
                raise RuntimeError("Length of redistricting data and first plan do not match.")

        node_data['assignment'] = plan
        grouped = node_data.groupby('assignment')
        district_data = grouped.sum()

        district_data.sort_index(inplace=True)
        for statistic, statistic_func in statistics_settings:
            statistic_series = statistic_func(district_data)
            ensemble_matrices[statistic][current_plan] = statistic_series.to_numpy()

    for statistic, ensemble_matrix in ensemble_matrices.items():
        print(f"Saving: {statistic}")
        np.savez_compressed(statistics_paths[statistic], ensemble_matrix)


def save_unique_plans(ensemble_directory, plans: np.ndarray) -> None:
    unique_plans = cm.determine_unique_plans(plans)
    print(f"Number Unique Plans: {len(unique_plans)}")
    np.savez_compressed(f'{ensemble_directory}/unique_plans.npz', np.array(unique_plans))


def compare_statistics(matrix_1, matrix_2, sort) -> None:
    if sort:
        matrix_1.sort(axis=1)
        matrix_2.sort(axis=1)
    print(np.shape(matrix_1))
    print(np.shape(matrix_2))
    if not np.allclose(matrix_1, matrix_2):
        print(matrix_1[0])
        print(matrix_2[0])
        print(f"different arrays")


def convert_to_csv(ensemble_directory, filename_prefix):
    array = cm.load_numpy_compressed(f'{ensemble_directory}{filename_prefix}.npz')
    cm.save_numpy_csv(f'{ensemble_directory}{filename_prefix}.csv', array)


if __name__ == '__main__':
    def main() -> None:
        print('start')

        t = Timer()
        t.start()

        directory = 'G:/rob/projects/election/rob/'

        if False:
            combine_and_fix_redistricting_data_file(directory)

        if False:
            ensemble_description = 'TXHD_2176_Reduced_3'
            plans = cm.load_plans_from_files(directory, ensemble_description, [12])
            ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
            save_unique_plans(ensemble_directory, plans)

        if True:
            chamber = 'TXHD'  # 'TXSN'  # 'USCD' #
            plan = 2176
            settings = cm.build_proposed_plan_simulation_settings(chamber, plan) # cm.build_TXSN_random_seed_simulation_settings()  # cm.build_USCD_random_seed_simulation_settings()  #

            seeds_directory = cm.build_seeds_directory(directory)
            dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)

            ensemble_description = f'{chamber}_{plan}_product_1' # f'{chamber}_random_seed_2' #
            ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)

            convert_to_csv(ensemble_directory, 'total_pop')
            convert_to_csv(ensemble_directory, 'o17_pop')

            # save_ensemble_matrices(chamber, directory, 'nodes_TX_2020_cnty_cntyvtd_sldu.parquet', dual_graph,  # nodes_TX_2020_cntyvtd_cd.csv
            #                        ensemble_description, range(0, 7))  # 15)) #

        if False:
            chamber = 'USCD'
            ensemble_description_1 = f'{chamber}_random_seed_2'
            ensemble_description_2 = f'{chamber}_random_seed_2Copy'
            ensemble_directory_1 = cm.build_ensemble_directory(directory, ensemble_description_1)
            ensemble_directory_2 = cm.build_ensemble_directory(directory, ensemble_description_2)
            sort = True
            for statistic, _ in build_statistics_settings():
                print(statistic)
                statistics_path_1 = f'{ensemble_directory_1}{statistic}.npz'
                statistics_path_2 = f'{ensemble_directory_2}{statistic}.npz'
                compare_statistics(cm.load_numpy_compressed(statistics_path_1),
                                   cm.load_numpy_compressed(statistics_path_2), sort)

        if False:
            chamber = 'TXSN'  # 'USCD'  #
            source = ''
            raw_input_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/RawInput' + source + '/'
            root_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/Input' + source + '/'
            data_directories = [raw_input_directory + x for x in
                                ['Outputs_USCD_2020Census/', 'Outputs_USCD_2020Census_A/']]

            merged_data_directory = root_directory + chamber + '/'
            cm.ensure_directory_exists(merged_data_directory)

            merge_data(data_directories, merged_data_directory)

            # df = pd.read_parquet(merged_data_meeting_path)
            # save_numpy_arrays(chamber, df, merged_data_directory)

        if False:
            assignments = [(3, 1), (2, 1), (1, 3), (17, 4), (27, 1)]
            canonical_list = build_canonical_assignments_list(assignments)
            print(canonical_list)

        t.stop()

        print('done')


    main()

# Notes
# t = Timer(name='class')
# t.start()
# # Do something
# t.stop()

# As a context manager:
#
# with Timer(name='context manager'):
#     # Do something

# As a decorator:
#
# @Timer(name='decorator')
# def stuff():
#     # Do something
