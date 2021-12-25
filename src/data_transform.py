from collections import defaultdict
from datetime import datetime
import networkx as nx
import numpy as np
from os import listdir
from os.path import isfile, exists
import pandas as pd
from typing import Callable, Iterable, Optional

import common as cm
import data_transform as dt
import proposed_plans as pp
import simulation as si
import simulation_analysis as sa
from timer import Timer


O17 = 'o17'
TOTAL = 'total'
CVAP = 'CVAP'
C = 'C'


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
    race_columns = [col for col in df.columns if (col.startswith(O17))]
    election_columns = [
        'President_2020_general_D_Biden',
        'President_2020_general_R_Trump',
        'USSen_2020_general_D_Hegar',
        'USSen_2020_general_R_Cornyn'
    ]
    return cm.union(race_columns, election_columns)


def pop(df: pd.DataFrame, group) -> pd.Series:
    return df[f'{group}_pop']


def hisp_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('hisp_pop', group)] / df[f'{group}_pop']


def white_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_white', group)] / df[f'{group}_pop']


def non_white_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return 1 - white_percent(df, group)


def black_sum_percent(df: pd.DataFrame, group: str) -> pd.Series:
    if group in build_census_population_groups():
        race_sum = black_sum(df, group)
        return race_sum / df[f'{group}_pop']
    elif group in build_cvap_population_groups():
        return black_percent(df, group)
    else:
        raise NotImplementedError("Unknown population group")


def black_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[f'calculated_{group}_black'] / df[f'{group}_pop']


def black_sum(df: pd.DataFrame, group: str) -> pd.Series:
    entire_population_group_columns = [col for col in df.columns if (col.startswith(group) and not col.startswith(hispanic_column('nonhisp', group)))]
    race_pop_columns = [col for col in entire_population_group_columns if 'black' in col]
    return df[race_pop_columns].sum(axis=1)


# ToDo: move logic for groups back to calling method
def black_hisp_sum_percent(df: pd.DataFrame, group: str) -> pd.Series:
    if group in build_census_population_groups():
        race_sum = black_hisp_sum(df, group)
        return race_sum / df[f'{group}_pop']
    elif group in build_cvap_population_groups():
        return black_hisp_percent(df, group)
    else:
        raise NotImplementedError("Unknown population group")


def black_hisp_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[f'calculated_{group}_black_hisp'] / df[f'{group}_pop']


def hispanic_column(suffix: str, group: str) -> str:
    return suffix if group == TOTAL else f'{group}_{suffix}'


def black_hisp_sum(df: pd.DataFrame, group: str) -> pd.Series:
    nonhisp_columns = [col for col in df.columns if col.startswith(hispanic_column('nonhisp', group))]
    race_nonhisp_columns = [col for col in nonhisp_columns if 'black' in col]
    race_pop_columns = [hispanic_column('hisp_pop', group)] + race_nonhisp_columns
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


def build_census_population_groups() -> list[str]:
    return [O17, TOTAL]


def build_cvap_population_groups() -> list[str]:
    return [CVAP, C]


def build_population_groups() -> list[str]:
    return dt.build_census_population_groups() + dt.build_cvap_population_groups()


def transform_racial_group_file_prefix(racial_group: str, population_group: str) -> tuple[str, str]:
    return ({
        'H': 'hisp',
        'B': 'black',
        'BH': 'black_hisp',
        'W': 'white',
        'NW': 'non_white'
    }[racial_group], {
        'VAP': 'o17',
        'T': 'total',
        'CVAP': 'cvap',
        'C': 'c'
    }[population_group])


def build_race_filename_prefix(race: str, group: str) -> str:
    return f'{race}_{group}_perc'


def build_race_filename_csv(race: str, group: str, suffix: str = '') -> str:
    return f'{build_race_filename_prefix(race, group)}{cm.build_suffix(suffix)}.csv'


def build_general_filename_csv(prefix: str, suffix: str = '') -> str:
    return f'{prefix}{cm.build_suffix(suffix)}.csv'


def build_election_filename_prefix(election: str) -> str:
    return 'votes_' + election


def build_election_filename_csv(election: str, suffix: str = '') -> str:
    return build_election_filename_prefix(election) + cm.build_suffix(suffix) + '.csv'


def build_statistics_settings() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [(build_election_filename_prefix('PRES20'), pres20_percent),
            (build_election_filename_prefix('SEN20'), sen20_percent)] + \
            build_statistics_group_settings(O17) + build_statistics_group_settings(TOTAL) + \
            build_statistics_group_settings(CVAP) + build_statistics_group_settings(C)


def build_statistics_group_settings(group: str) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (build_race_filename_prefix('hisp', group), lambda x: hisp_percent(x, group)),
        (build_race_filename_prefix('black', group), lambda x: black_percent(x, group)),
        (build_race_filename_prefix('black_hisp', group), lambda x: black_hisp_percent(x, group)),
        (build_race_filename_prefix('white', group), lambda x: white_percent(x, group)),
        (build_race_filename_prefix('non_white', group), lambda x: non_white_percent(x, group)),
        (f'pop_{group}', lambda x: pop(x, group))]


def build_canonical_assignments_list(assignments: list[tuple[int, int]]) -> list[list[int]]:
    partition_dictionary = defaultdict(list)
    for unit, district in assignments:
        partition_dictionary[district].append(unit)
    partition_list = [sorted(x) for x in partition_dictionary.values()]
    partition_list.sort(key=lambda x: x[0])
    return partition_list


def load_filtered_redistricting_data(directory: str, redistricting_data_filename: str, cvap_filename: str,
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

    calculated_columns = []
    for group in build_census_population_groups():
        column = f'calculated_{group}_black'
        node_data[column] = black_sum(node_data, group)
        calculated_columns.append(column)
        column = f'calculated_{group}_black_hisp'
        node_data[column] = black_hisp_sum(node_data, group)
        calculated_columns.append(column)

    filtered_node_data = node_data.filter(items=[
                                                    'geoid',
                                                    'President_2020_general_D_Biden', 'President_2020_general_R_Trump',
                                                    'USSen_2020_general_D_Hegar', 'USSen_2020_general_R_Cornyn',
                                                    'total_pop', 'o17_pop', 'nonhisp_pop',
                                                    'o17_hisp_pop', 'o17_nonhisp_white',
                                                    'hisp_pop', 'nonhisp_white'
                                                ] + calculated_columns + additional_columns)
    filtered_node_data.set_index('geoid', drop=False, inplace=True)

    cvap_data = pd.read_parquet(f'{redistricting_data_directory}{cvap_filename}')
    filtered_node_data = filtered_node_data.join(cvap_data, how='inner')

    filtered_node_data.sort_index(inplace=True)

    return filtered_node_data


def build_cvap_county_countyvtd_data_path_prefix(directory: str) -> str:
    return f'{pp.build_redistricting_data_directory(directory)}{build_cvap_county_countyvtd_data_filename_prefix()}'


def build_cvap_county_countyvtd_data_filename_prefix() -> str:
    return 'cvap_cty_cntyvtd'


def build_cvap_data_path_prefix(directory: str, chamber: str, plan: int) -> str:
    return f'{pp.build_redistricting_data_directory(directory)}{build_cvap_data_filename_prefix(chamber, plan)}'


def build_cvap_data_filename_prefix(chamber: str, plan: int) -> str:
    return f'cvap_{chamber}_{plan}'


def save_cvap_county_countyvtd_data(directory: str) -> None:
    cvap_data = pp.load_cvap_data(directory)
    block_ids = list(cvap_data.index.values)

    chamber = 'USCD'
    geoid_to_node_ids = match_block_ids(directory, chamber, block_ids)

    save_cvap_data_impl(cvap_data, geoid_to_node_ids, build_cvap_county_countyvtd_data_path_prefix(directory))


def save_cvap_data(directory: str, chamber: str, plan: int) -> None:
    cvap_data = pp.load_cvap_data(directory)
    geoid_to_node_assignments = build_geoid_to_node_ids(directory, chamber, plan)
    cvap_data = cvap_data[cvap_data.index.isin(geoid_to_node_assignments)]
    print(f"Matched: {len(cvap_data)}")
    save_cvap_data_impl(cvap_data, geoid_to_node_assignments, build_cvap_data_path_prefix(directory, chamber, plan))


def build_geoid_to_node_ids(directory: str, chamber: str, plan: int) -> dict[str, str]:
    seeds_directory = cm.build_seeds_directory(directory)
    seed_filename_prefix = sa.build_seed_filename_prefix(chamber, 'cntyvtd', plan)
    dual_graph_path_prefix = f'{seeds_directory}graph_{seed_filename_prefix}'
    dual_graph = nx.read_gpickle(dual_graph_path_prefix + '.gpickle')
    return build_geoid_to_node_ids_from_graph(dual_graph)


def build_geoid_to_node_ids_from_graph(graph: nx.Graph):
    return {geoid: x for x, y in graph.nodes.items() for geoid in y['node_geoids'].split(",")}


def save_cvap_data_impl(cvap_data: pd.DataFrame, geoid_to_node_ids: dict[str, str], output_path_prefix: str) -> None:
    cvap_data['node_id'] = [geoid_to_node_ids[x] for x in list(cvap_data.index.values)]
    cvap_groups = cvap_data.groupby('node_id')
    cvap_grouped_data = cvap_groups.sum()
    cvap_grouped_data.index.rename('geoid', inplace=True)
    cvap_grouped_data.sort_values(by=['geoid'], inplace=True)

    cvap_grouped_data.to_csv(f'{output_path_prefix}.csv')
    cvap_grouped_data.to_parquet(f'{output_path_prefix}.parquet')


def match_block_ids(directory:str, chamber: str, block_ids: list[str]):
    settings = cm.build_settings(chamber)
    seeds_directory = cm.build_seeds_directory(directory)
    dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)

    keys = list(dual_graph.nodes.keys())
    counties = {x for x in keys if len(x) == 3}
    vtds = {x for x in keys if len(x) == 9}
    print(f"{len(keys)} {len(counties)} {len(vtds)}")

    geoid_components = [(x, x[0:3]) for x in block_ids]

    redistricting_data = sa.load_redistricting_data(directory)

    county_vtd_lookup = {x: y for x, y in zip(redistricting_data['geoid'], redistricting_data['cntyvtd'])}
    geoid_county_matches = [(x, y) for x, y in geoid_components if y in counties]
    geoid_vtd_matches = [(x, county_vtd_lookup[x]) for x, _ in geoid_components if county_vtd_lookup[x] in vtds]
    non_matches = [x for x, y in geoid_components if not (y in counties or county_vtd_lookup[x] in vtds)]
    matches = {x: y for x, y in geoid_county_matches + geoid_vtd_matches}
    print(f"{len(block_ids)} {len(geoid_county_matches)} {len(geoid_vtd_matches)} {len(matches)} {len(non_matches)}")

    if any(non_matches):
        raise RuntimeError("Not all blocks matched")

    return matches


def combine_and_fix_redistricting_data_file(directory: str) -> None:
    county_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_sldu.csv')
    county_vtd_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cntyvtd_sldu.csv')

    county_data['geoid'] = county_data['geoid'].astype('str').apply(lambda p: p.zfill(3))
    county_vtd_data['geoid'] = county_vtd_data['geoid'].astype('str').apply(lambda p: p.zfill(9))

    combined = pd.concat([county_data, county_vtd_data])
    output_prefix = f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_cntyvtd_sldu'
    combined.to_parquet(output_prefix + '.parquet', index=False)
    combined.to_csv(output_prefix + '.csv', index=False)


def save_ensemble_matrices(chamber: str, directory: str, redistricting_data_filename: str, cvap_filename,
                           graph: nx.Graph, ensemble_description: str, use_unique_plans,
                           file_numbers: Optional[Iterable[int]], force: bool) -> None:
    node_data = load_filtered_redistricting_data(directory, redistricting_data_filename, cvap_filename)
    statistics_settings = build_statistics_settings()
    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)

    if use_unique_plans:
        plans = cm.load_plans_from_path(f'{ensemble_directory}unique_plans.npz')
    else:
        assert isinstance(file_numbers, Iterable)
        plans = cm.load_plans_from_files(directory, ensemble_description, file_numbers)

    statistics_paths = {x: f'{ensemble_directory}{x}.npz' for x, _ in statistics_settings}
    statistics_settings = [(x, y) for x, y in statistics_settings if force or not exists(statistics_paths[x])]
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


def save_unique_plans(ensemble_directory: str, plans: np.ndarray) -> None:
    unique_plans = cm.determine_unique_plans(plans)
    print(f"Number Unique Plans: {len(unique_plans)}")
    np.savez_compressed(f'{ensemble_directory}/unique_plans.npz', np.array(unique_plans))


def compare_statistics(matrix_1: np.ndarray, matrix_2: np.ndarray, sort: bool) -> None:
    if sort:
        matrix_1.sort(axis=1)
        matrix_2.sort(axis=1)
    print(np.shape(matrix_1))
    print(np.shape(matrix_2))
    if not np.allclose(matrix_1, matrix_2):
        print(matrix_1[0])
        print(matrix_2[0])
        print(f"different arrays")


def convert_to_csv(ensemble_directory: str, filename_prefix: str) -> None:
    array = cm.load_numpy_compressed(f'{ensemble_directory}{filename_prefix}.npz')
    cm.save_numpy_csv(f'{ensemble_directory}{filename_prefix}.csv', array)


if __name__ == '__main__':
    def main() -> None:
        print('start')

        t = Timer()
        t.start()

        directory = 'G:/rob/projects/election/rob/'

        chamber = 'DCN'  # 'USCD'  # 'TXHD'  #

        settings = cm.build_settings(chamber)

        if False:
            combine_and_fix_redistricting_data_file(directory)

        if False:
            plans = cm.load_plans_from_files(directory, settings.ensemble_description, range(0, settings.number_files))
            ensemble_directory = cm.build_ensemble_directory(directory, settings.ensemble_description)
            save_unique_plans(ensemble_directory, plans)

        if False:
            save_cvap_data(directory, 'DCN', 93173)

        if True:
            seeds_directory = cm.build_seeds_directory(directory)
            dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)

            save_ensemble_matrices(chamber, directory, settings.redistricting_data_filename,
                                   settings.cvap_data_filename, dual_graph, settings.ensemble_description,
                                   False, range(0, settings.number_files),  # 1), #
                                   True)  # False, None)  #
            ensemble_directory = cm.build_ensemble_directory(directory, settings.ensemble_description)
            for population_group in build_population_groups():
                convert_to_csv(ensemble_directory, f'pop_{population_group}')

        if False:
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
            source = ''
            raw_input_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/RawInput' + source + '/'
            root_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/Input' + source + '/'
            data_directories = [raw_input_directory + x for x in
                                ['Outputs_USCD_2020Census/', 'Outputs_USCD_2020Census_A/']]

            merged_data_directory = root_directory + chamber + '/'
            cm.ensure_directory_exists(merged_data_directory)

            merge_plan_data(data_directories, merged_data_directory)

            # df = pd.read_parquet(merged_data_meeting_path)
            # save_numpy_arrays(chamber, df, merged_data_directory)

        if False:
            assignments = [(3, 1), (2, 1), (1, 3), (17, 4), (27, 1)]
            canonical_list = build_canonical_assignments_list(assignments)
            print(canonical_list)

        t.stop()

        print('done')


    main()
