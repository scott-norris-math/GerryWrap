from collections import defaultdict
from datetime import datetime
import gerrychain
import geopandas as gpd
import itertools
import networkx as nx
import numpy as np
from os import listdir
from os.path import isfile, exists
import pandas as pd
import shapely as sh
from typing import Callable, Iterable, Optional

import common as cm
import data_transform as dt
import proposed_plans as pp
import simulation as si
import simulation_analysis as sa
from timer import Timer
import vra

O17 = 'o17'
TOTAL = 'total'
CVAP = 'CVAP'
C = 'C'
VRA_PREFIX = 'mggg_eff_'


def get_census_chamber_name(chamber: str) -> str:
    return {
        'USCD': 'cd',
        'TXSN': 'sldu',
        'TXHD': 'sldl'
    }[chamber]


def merge_ensemble_matrices(input_directories: Iterable[str], output_directory: str) -> None:
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


def pop_density(df: pd.DataFrame, group) -> pd.Series:
    return pop(df, group) / area(df)


def hisp_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('hisp_pop', group)] / df[f'{group}_pop']


def white_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_white', group)] / df[f'{group}_pop']


def asian_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_asian', group)] / df[f'{group}_pop']


def native_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_native', group)] / df[f'{group}_pop']


def pacific_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_pacific', group)] / df[f'{group}_pop']


def other_percent(df: pd.DataFrame, group: str) -> pd.Series:
    return df[hispanic_column('nonhisp_other', group)] / df[f'{group}_pop']


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
    entire_population_group_columns = [col for col in df.columns if (
            col.startswith(group) and not col.startswith(hispanic_column('nonhisp', group)))]
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


def area(df: pd.DataFrame) -> pd.Series:
    return df['aland']


def pres20_D_total(df: pd.DataFrame) -> pd.Series:
    return df['President_2020_general_D_Biden']


def pres20_R_total(df: pd.DataFrame) -> pd.Series:
    return df['President_2020_general_R_Trump']


def pres20_percent(df: pd.DataFrame) -> pd.Series:
    dem_votes = pres20_D_total(df)
    return dem_votes / (dem_votes + pres20_R_total(df))


def sen20_D_total(df: pd.DataFrame) -> pd.Series:
    return df['USSen_2020_general_D_Hegar']


def sen20_R_total(df: pd.DataFrame) -> pd.Series:
    return df['USSen_2020_general_R_Cornyn']


def sen20_percent(df: pd.DataFrame) -> pd.Series:
    dem_votes = sen20_D_total(df)
    return dem_votes / (dem_votes + sen20_R_total(df))


def build_census_population_groups() -> list[str]:
    return [O17, TOTAL]


def build_cvap_population_groups() -> list[str]:
    return [CVAP, C]


def build_population_groups(chamber: str) -> list[str]:
    return build_census_population_groups() + ([] if chamber == 'TXHD' else build_cvap_population_groups())


def transform_racial_population_group_file_prefix(racial_group: str, population_group: str) -> tuple[str, str]:
    return ({
                'H': 'hisp',
                'B': 'black',
                'BH': 'black_hisp',
                'W': 'white',
                'NW': 'non_white',
                'A': 'asian',
                'P': 'pacific',
                'N': 'native',
                'O': 'other'
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


def build_election_filename_prefix(election: str, statistic: str) -> str:
    return f'{election}_{statistic}'


def build_election_filename_csv(election: str, statistic: str, suffix: str = '') -> str:
    return build_election_filename_prefix(election, statistic) + cm.build_suffix(suffix) + '.csv'


def build_matrix_statistics_settings(chamber: str) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    election_statistics = [(build_election_filename_prefix('PRES20', 'votes'), pres20_percent),
                           (build_election_filename_prefix('PRES20', 'D_total'), pres20_D_total),
                           (build_election_filename_prefix('PRES20', 'R_total'), pres20_R_total),
                           (build_election_filename_prefix('SEN20', 'votes'), sen20_percent),
                           (build_election_filename_prefix('SEN20', 'D_total'), sen20_D_total),
                           (build_election_filename_prefix('SEN20', 'R_total'), sen20_R_total)]
    return election_statistics + list(
        itertools.chain(*(build_statistics_group_settings(x) for x in build_population_groups(chamber))))


def build_statistics_group_settings(group: str) -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (build_race_filename_prefix('hisp', group), lambda x: hisp_percent(x, group)),
        (build_race_filename_prefix('black', group), lambda x: black_percent(x, group)),
        (build_race_filename_prefix('black_hisp', group), lambda x: black_hisp_percent(x, group)),
        (build_race_filename_prefix('white', group), lambda x: white_percent(x, group)),
        (build_race_filename_prefix('asian', group), lambda x: asian_percent(x, group)),
        (build_race_filename_prefix('native', group), lambda x: native_percent(x, group)),
        (build_race_filename_prefix('pacific', group), lambda x: pacific_percent(x, group)),
        (build_race_filename_prefix('other', group), lambda x: other_percent(x, group)),
        (build_race_filename_prefix('non_white', group), lambda x: non_white_percent(x, group)),
        (f'pop_{group}', lambda x: pop(x, group))
    ]


def build_vector_statistics_settings() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    statistics = [
        (build_election_filename_prefix(election, statistic), lambda x: statistic_func(np.array(election_func(x))))
        for election, election_func in [('PRES20', pres20_percent), ('SEN20', sen20_percent)]
        for statistic, statistic_func in [('mean_median', calculate_mean_median),
                                          ('partisan_bias', calculate_partisan_bias),
                                          ('partisan_gini', calculate_partisan_gini)]]
    return statistics


def calculate_mean_median(plan_vector: np.ndarray) -> float:
    # mean-median score, denominated in terms of "percentage needed for
    #  majority" (R - D). Effectively this is:
    #  2 (because MM measures diff. from midpoint), and
    #   x 100 (to make it a %)
    return 200 * (np.median(plan_vector) - np.mean(plan_vector))


def calculate_partisan_bias(plan_vector: np.ndarray) -> float:
    mean_voteshare = np.mean(plan_vector)
    # DOES NOT matter if sorted, because we will sum up how many above 50%
    seats_votes1 = mean_voteshare - np.array(plan_vector) + 0.5
    seats_votes2 = np.flip(1 - seats_votes1)

    number_seats1 = np.count_nonzero(seats_votes1 <= 0.5)
    number_seats2 = np.count_nonzero(seats_votes2 <= 0.5)

    # TODO: Replace calculation with
    # 2 * np.count_nonzero(np.array(plan_vector) >= mean_voteshare) - number districts

    return number_seats1 - number_seats2


def calculate_partisan_gini(plan_vector: np.ndarray) -> float:
    # Code taken from gerrychain/metrics/partisan.py
    race_results = sorted(plan_vector, reverse=True)
    seats_votes = [plan_vector - r + 0.5 for r in race_results]

    # Apply reflection of seats-votes curve about (.5, .5)
    reflected_sv = reversed([1 - s for s in seats_votes])
    # Calculate the unscaled, unsigned area between the seats-votes curve
    # and its reflection. For each possible number of seats attained, we find
    # the area of a rectangle of unit height, with a width determined by the
    # horizontal distance between the curves at that number of seats.
    unscaled_area = sum(abs(s - r) for s, r in zip(seats_votes, reflected_sv))
    # We divide by area by the number of seats to obtain a partisan Gini score
    # between 0 and 1.
    return unscaled_area / len(race_results)


def calculate_ensemble_wasted_votes(party1_votes, party2_votes):
    wasted_votes = np.zeros(np.shape(party1_votes))
    total = np.sum(party1_votes[0] + party2_votes[0])
    for i, (party1_row, party2_row) in enumerate(zip(party1_votes, party2_votes)):
        wasted_votes[i] = calculate_wasted_votes(party1_row, party2_row, total)
    return wasted_votes


def calculate_wasted_votes(party1_votes: np.ndarray, party2_votes: np.ndarray, total: float) -> np.ndarray:
    return np.array([(x - y) / 2 if x > y else x for x, y in zip(party1_votes, party2_votes)]) / total


def calculate_efficiency_gap(ensemble_directory: str, election: str) -> np.ndarray:
    votes_D = cm.load_ensemble_matrix(ensemble_directory, f'{election}_D_total')
    votes_R = cm.load_ensemble_matrix(ensemble_directory, f'{election}_R_total')
    wasted_votes_D = calculate_ensemble_wasted_votes(votes_D, votes_R)
    wasted_votes_R = calculate_ensemble_wasted_votes(votes_R, votes_D)
    efficiency_gap = np.sum(wasted_votes_R - wasted_votes_D, axis = 1)
    return efficiency_gap


def load_filtered_redistricting_data(chamber: str, directory: str, redistricting_data_filename: str, cvap_filename: str,
                                     additional_columns: list[str] = []) -> pd.DataFrame:
    redistricting_data_path = f'{pp.build_redistricting_data_directory(directory)}{redistricting_data_filename}'
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
                                                    'geoid', 'aland',
                                                    'President_2020_general_D_Biden', 'President_2020_general_R_Trump',
                                                    'USSen_2020_general_D_Hegar', 'USSen_2020_general_R_Cornyn',
                                                    'total_pop', 'o17_pop', 'nonhisp_pop',
                                                    'hisp_pop', 'o17_hisp_pop',
                                                    'nonhisp_white', 'o17_nonhisp_white',
                                                    'nonhisp_asian', 'o17_nonhisp_asian',
                                                    'nonhisp_pacific', 'o17_nonhisp_pacific',
                                                    'nonhisp_native', 'o17_nonhisp_native',
                                                    'nonhisp_other', 'o17_nonhisp_other'
                                                ] + calculated_columns + additional_columns)
    filtered_node_data.set_index('geoid', drop=False, inplace=True)

    if chamber != 'TXHD':
        cvap_data = pd.read_parquet(f'{pp.build_redistricting_data_calculated_directory(directory)}{cvap_filename}')
        print(f'CVAP Data: {len(cvap_data)}  Filtered Data Before: {len(filtered_node_data)}')
        filtered_node_data = filtered_node_data.join(cvap_data, how='inner')
        print(f'Filtered Data After: {len(filtered_node_data)}')

    filtered_node_data.sort_index(inplace=True)

    return filtered_node_data


def load_cvap_county_countyvtd_data(chamber: str, directory: str) -> pd.DataFrame:
    return pd.read_parquet(f'{build_cvap_county_countyvtd_data_path_prefix(chamber, directory)}.parquet')


def build_cvap_county_countyvtd_data_path_prefix(chamber: str, directory: str) -> str:
    return f'{pp.build_redistricting_data_calculated_directory(directory)}{build_cvap_county_countyvtd_data_filename_prefix(chamber)}'


def build_cvap_county_countyvtd_data_filename_prefix(chamber: str) -> str:
    return f'cvap_cty_cntyvtd_{chamber}'


def build_cvap_data_path_prefix(chamber: str, directory: str, plan: int) -> str:
    return f'{pp.build_redistricting_data_calculated_directory(directory)}{build_cvap_data_filename_prefix(chamber, plan)}'


def build_cvap_data_filename_prefix(chamber: str, plan: int) -> str:
    return f'cvap_{chamber}_{plan}'


def build_redistricting_county_vtd_data_path_prefix(directory: str) -> str:
    return f'{pp.build_redistricting_data_calculated_directory(directory)}{build_redistricting_county_vtd_filename_prefix()}'


def build_redistricting_county_vtd_filename_prefix() -> str:
    return f'redistricting_county_vtd_data'


def save_cvap_county_countyvtd_data(chamber: str, directory: str) -> None:
    cvap_data = pp.load_cvap_data(directory)
    block_ids = list(cvap_data.index.values)

    county_vtd_lookup = load_county_vtd_lookup(directory)

    dual_graph = sa.load_settings_dual_graph(chamber, directory)
    geoid_to_node_ids = match_census_block_ids(dual_graph, block_ids, county_vtd_lookup)

    save_data_frame_grouped_by_ids(cvap_data, geoid_to_node_ids,
                                   build_cvap_county_countyvtd_data_path_prefix(chamber, directory))


def save_cvap_data(chamber: str, directory: str, plan: Optional[int]) -> None:
    cvap_data = pp.load_cvap_data(directory)
    dual_graph = sa.load_settings_dual_graph(chamber, directory) \
        if plan is None else sa.load_plan_dual_graph(chamber, directory, plan)
    geoid_to_node_assignments = build_geoid_to_node_ids_from_graph(dual_graph)
    missing_geoids = [x for x in geoid_to_node_assignments if x not in cvap_data.index]
    cvap_data = cvap_data[cvap_data.index.isin(geoid_to_node_assignments)]
    print(f"Matched: {len(cvap_data)} Missing: {len(missing_geoids)}")
    save_data_frame_grouped_by_ids(cvap_data, geoid_to_node_assignments,
                                   build_cvap_data_path_prefix(chamber, directory, plan))


def save_redistricting_county_vtd_data(directory: str) -> None:
    def union_geometry(group: pd.DataFrame) -> dict[str, str]:
        return {
            'cntyvtd': group['cntyvtd'].iloc[0],
            'geometry': sh.ops.unary_union(group['polygon']).wkt
        }

    redistricting_data = sa.load_redistricting_data(directory)
    si.fix_election_columns_text(redistricting_data)
    geometry_df = redistricting_data.filter(items=['cntyvtd', 'polygon']).copy()
    redistricting_data.drop(
        columns=['county', 'tabblock', 'bg', 'tract', 'cnty', 'cd', 'sldu', 'sldl', 'polygon'],
        inplace=True)
    county_vtd_lookup = {x: y for x, y in zip(redistricting_data['geoid'], redistricting_data['cntyvtd'])}
    redistricting_data.set_index('geoid', inplace=True, drop=True)
    data_frame_group_sums = dt.group_data_frame_by_ids(redistricting_data, county_vtd_lookup)
    geometry_grouped = pd.DataFrame(list(geometry_df.groupby('cntyvtd').apply(union_geometry)))
    geometry_grouped.set_index('cntyvtd', drop=True, inplace=True)
    geometry_grouped.index.rename('geoid', inplace=True)

    data_frame_group_sums_joined = data_frame_group_sums.join(geometry_grouped, how='inner')
    data_frame_group_sums_joined.to_parquet(f'{build_redistricting_county_vtd_data_path_prefix(directory)}.parquet')


def load_redistricting_county_vtd_data(directory: str) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(pd.read_parquet(build_redistricting_county_vtd_data_path_prefix(directory) + '.parquet'))
    gdf['geometry'] = gdf['geometry'].apply(sh.wkt.loads)
    return gdf


def save_vra_county_countyvtd_data(chamber: str, directory: str) -> None:
    vra_data = pp.load_vra_data(directory)
    county_vtd_ids = list(vra_data.index.values)

    county_vtd_to_node_ids, unmatched_vtd_ids = match_county_vtd_ids_with_overrides(chamber, directory, county_vtd_ids)

    data_frame_group_sums = group_data_frame_by_ids(vra_data, county_vtd_to_node_ids)
    data_frame_group_sums.drop(columns=['CNTY_x', 'COLOR_x', 'VTDKEY_x', 'Unnamed_ 0'], inplace=True)

    unmatched_vtds = pd.DataFrame(columns=data_frame_group_sums.columns, index=unmatched_vtd_ids).fillna(0)
    data_frame_group_sums = data_frame_group_sums.append(unmatched_vtds)

    data_frame_group_sums['CNTYVTD'] = data_frame_group_sums.index
    data_frame_group_sums.sort_values(by='CNTYVTD', inplace=True)

    save_data_frame(data_frame_group_sums, build_vra_county_countyvtd_data_path_prefix(chamber, directory))


def match_county_vtd_ids_with_overrides(chamber: str, directory: str,
                                        county_vtd_ids: list[str]) -> tuple[dict[str, str], set[str]]:
    graph = sa.load_settings_dual_graph(chamber, directory)
    overrides = dict()
    if chamber == 'USCD':
        overrides['113004650'] = '113004649'
        overrides['201000922'] = '201000921'
    county_vtd_to_node_ids, unmatched_vtd_ids = match_county_vtd_ids(graph, county_vtd_ids, overrides)
    return county_vtd_to_node_ids, unmatched_vtd_ids


def load_vra_county_countyvtd_data(chamber: str, directory: str) -> pd.DataFrame:
    return pd.read_parquet(f'{build_vra_county_countyvtd_data_path_prefix(chamber, directory)}.parquet')


def build_vra_county_countyvtd_data_path_prefix(chamber: str, directory: str) -> str:
    return f'{pp.build_redistricting_data_calculated_directory(directory)}{build_vra_county_countyvtd_data_filename_prefix(chamber)}'


def build_vra_county_countyvtd_data_filename_prefix(chamber) -> str:
    return f'cvap_vra_cty_cntyvtd_{chamber}'


def build_geoid_to_node_ids_from_graph(graph: nx.Graph) -> dict[str, str]:
    def normalize(geoid: str) -> str:
        return geoid.removeprefix('48') if len(geoid) == 15 else geoid

    return {normalize(geoid): x for x, y in graph.nodes.items() for geoid in y['node_geoids'].split(",")}


def save_data_frame_grouped_by_ids(data_frame: pd.DataFrame, data_frame_id_to_group_ids: dict[str, str],
                                   output_path_prefix: str) -> None:
    data_frame_group_sums = group_data_frame_by_ids(data_frame, data_frame_id_to_group_ids)
    save_data_frame(data_frame_group_sums, output_path_prefix)


def save_data_frame(data_frame: pd.DataFrame, output_path_prefix: str) -> None:
    data_frame.to_csv(f'{output_path_prefix}.csv')
    data_frame.to_parquet(f'{output_path_prefix}.parquet')


def group_data_frame_by_ids(data_frame: pd.DataFrame, data_frame_id_to_group_ids: dict[str, str]) -> pd.DataFrame:
    data_frame['group_id'] = [data_frame_id_to_group_ids[x] for x in list(data_frame.index.values)]
    sums_df = data_frame.groupby('group_id').sum()
    sums_df.index.rename('geoid', inplace=True)
    sums_df.sort_values(by=['geoid'], inplace=True)
    return sums_df


def match_census_block_ids(graph: nx.Graph, block_ids: list[str], county_vtd_lookup: dict[str, str]) -> dict[str, str]:
    counties, vtds = determine_county_county_vtd_node_ids(graph)
    geoid_components = [(x, x[0:3]) for x in block_ids]

    geoid_county_matches = [(x, y) for x, y in geoid_components if y in counties]
    geoid_vtd_matches = [(x, county_vtd_lookup[x]) for x, _ in geoid_components if county_vtd_lookup[x] in vtds]
    matches = {x: y for x, y in geoid_county_matches + geoid_vtd_matches}
    non_matches = [x for x, y in geoid_components if x not in matches]

    print(f"{len(block_ids)} {len(geoid_county_matches)} {len(geoid_vtd_matches)} {len(matches)} {len(non_matches)}")

    if any(non_matches):
        print(f"Unmatched Blocks {len(non_matches)}:")
        for non_match in sorted(non_matches):
            print(non_match)
        raise RuntimeError("Not all blocks matched")

    return matches


def load_county_vtd_lookup(directory: str) -> dict[str, str]:
    lookup_df = pd.read_csv(build_county_vtd_lookup_path(directory))
    return {str(x).zfill(13): y for x, y in zip(lookup_df['geoid'], lookup_df['cntyvtd'])}


def save_county_vtd_lookup(directory: str) -> None:
    redistricting_data = sa.load_redistricting_data(directory)
    lookup_df = redistricting_data.filter(items=['geoid', 'cntyvtd'])
    lookup_df.to_csv(build_county_vtd_lookup_path(directory))


def build_county_vtd_lookup_path(directory: str) -> str:
    return f'{pp.build_redistricting_data_calculated_directory(directory)}county_vtd_lookup.csv'


def match_county_vtd_ids(graph: nx.Graph, county_vtd_ids: list[str], overrides: dict[str, str]) -> tuple[
    dict, set[str]]:
    def map_vtd(s: str) -> str:
        return s if s[8].isdigit() else f'{s[0:3]}0{s[3:8]}'.lower()

    counties, vtds = determine_county_county_vtd_node_ids(graph)

    vtd_components = [(x, x[0:3]) for x in county_vtd_ids]

    vtd_matches = [(x, map_vtd(x)) for x, _ in vtd_components if map_vtd(x) in vtds]
    county_matches = [(x, y) for x, y in vtd_components if y in counties]
    matches = {x: y for x, y in county_matches + vtd_matches}
    for x, y in overrides.items():
        matches[x] = y
    non_matches = [x for x, y in vtd_components if x not in matches]

    print(f"{len(county_vtd_ids)} {len(county_matches)} {len(vtd_matches)} {len(matches)} {len(non_matches)}")

    if any(non_matches):
        raise RuntimeError("Not all vtds matched")

    unmatched_vtds = vtds - {y for x, y in vtd_matches}
    return matches, unmatched_vtds


def determine_county_county_vtd_node_ids(dual_graph: nx.Graph) -> tuple[set[str], set[str]]:
    keys = list(dual_graph.nodes.keys())
    counties = {x for x in keys if len(x) == 3}
    vtds = {x for x in keys if len(x) == 9}

    if len(keys) != len(counties) + len(vtds):
        print(f"Unexpected Node Types {len(keys)} {len(counties)} {len(vtds)}")
        unknown_node_types = set(keys) - counties - vtds
        for x in unknown_node_types:
            print(x)
        raise RuntimeError("Unexpected Node Types")

    return counties, vtds


def combine_and_fix_redistricting_data_file(directory: str) -> None:
    county_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_sldu.csv')
    county_vtd_data = pd.read_csv(f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cntyvtd_sldu.csv')

    county_data['geoid'] = county_data['geoid'].astype('str').apply(lambda p: p.zfill(3))
    county_vtd_data['geoid'] = county_vtd_data['geoid'].astype('str').apply(lambda p: p.zfill(9))

    combined = pd.concat([county_data, county_vtd_data])
    output_prefix = f'{pp.build_redistricting_data_directory(directory)}nodes_TX_2020_cnty_cntyvtd_sldu'
    combined.to_parquet(output_prefix + '.parquet')
    combined.to_csv(output_prefix + '.csv', index=False)


def save_ensemble_matrices(chamber: str, directory: str, redistricting_data_filename: str, cvap_filename: str,
                           graph: nx.Graph, ensemble_description: str, use_unique_plans: bool,
                           file_numbers: Optional[Iterable[int]], district_offset: int, force: bool) -> None:
    plans = load_plans(directory, ensemble_description, use_unique_plans, file_numbers, district_offset)
    node_data = load_filtered_redistricting_data(chamber, directory, redistricting_data_filename, cvap_filename)

    statistics_settings = build_matrix_statistics_settings(chamber)
    statistics_paths = {x: f'{cm.build_ensemble_directory(directory, ensemble_description)}{x}.npz'
                        for x, _ in statistics_settings}
    statistics_settings = [(x, y) for x, y in statistics_settings if force or not exists(statistics_paths[x])]
    if len(statistics_settings) == 0:
        raise RuntimeError("All ensemble matrices already exist")

    print("Saving ensemble matrices for: ")
    for statistic, _ in statistics_settings:
        print(statistic)

    number_districts = cm.get_number_districts(chamber)
    ensemble_matrices = {statistic: np.zeros((len(plans), number_districts)) for statistic, _ in statistics_settings}

    for plan_id, district_data in enumerate(iterate_plans_district_data(graph, plans, node_data)):
        for statistic, statistic_func in statistics_settings:
            statistic_series = statistic_func(district_data)
            ensemble_matrices[statistic][plan_id] = statistic_series.to_numpy()

    for statistic, ensemble_matrix in ensemble_matrices.items():
        print(f"Saving: {statistic}")
        np.savez_compressed(statistics_paths[statistic], ensemble_matrix)


def build_ensemble_matrix_saver(chamber: str, directory: str, ensemble_description: str, force: bool,
                                plans: np.ndarray) -> tuple[list, Callable, Callable]:
    def build() -> tuple[list, Callable, Callable]:
        statistics_settings = build_matrix_statistics_settings(chamber)
        statistics_paths = {x: f'{cm.build_ensemble_directory(directory, ensemble_description)}{x}.npz'
                            for x, _ in statistics_settings}
        statistics_settings = [(x, y) for x, y in statistics_settings if force or not exists(statistics_paths[x])]

        number_districts = cm.get_number_districts(chamber)
        ensemble_matrices = {statistic: np.zeros((len(plans), number_districts)) for statistic, _ in
                             statistics_settings}

        def process(plan_id: int, district_data: pd.DataFrame) -> None:
            for statistic, statistic_func in statistics_settings:
                statistic_series = statistic_func(district_data)
                ensemble_matrices[statistic][plan_id] = statistic_series.to_numpy()

        def finish() -> None:
            for statistic, ensemble_matrix in ensemble_matrices.items():
                print(f"Saving: {statistic}")
                np.savez_compressed(statistics_paths[statistic], ensemble_matrix)

        return [x for x, _ in statistics_settings], process, finish

    return build()


def build_ensemble_vector_saver(directory: str, ensemble_description: str, force: bool,
                                plans: np.ndarray) -> tuple[list, Callable, Callable]:
    def build() -> tuple[list, Callable, Callable]:
        statistics_settings = build_vector_statistics_settings()
        statistics_paths = {x: f'{cm.build_ensemble_directory(directory, ensemble_description)}{x}.npz'
                            for x, _ in statistics_settings}
        statistics_settings = [(x, y) for x, y in statistics_settings if force or not exists(statistics_paths[x])]

        ensemble_vectors = {statistic: np.zeros((len(plans))) for statistic, _ in statistics_settings}

        def process(plan_id: int, district_data: pd.DataFrame) -> None:
            for statistic, statistic_func in statistics_settings:
                ensemble_vectors[statistic][plan_id] = statistic_func(district_data)

        def finish() -> None:
            for statistic, ensemble_vector in ensemble_vectors.items():
                print(f"Saving: {statistic}")
                np.savez_compressed(statistics_paths[statistic], ensemble_vector)

        return [x for x, _ in statistics_settings], process, finish

    return build()


def save_ensemble_data(chamber: str, directory: str, redistricting_data_filename: str, cvap_filename: str,
                       graph: nx.Graph, ensemble_description: str, use_unique_plans: bool,
                       file_numbers: Optional[Iterable[int]], district_offset: int, force: bool):
    plans = load_plans(directory, ensemble_description, use_unique_plans, file_numbers, district_offset)
    node_data = load_filtered_redistricting_data(chamber, directory, redistricting_data_filename, cvap_filename)

    computers = [build_ensemble_matrix_saver(chamber, directory, ensemble_description, force, plans),
                 build_ensemble_vector_saver(directory, ensemble_description, force, plans)]

    print("Saving ensemble data for: ")
    statistics = [y for x in computers for y in x[0]]
    for statistic in statistics:
        print(statistic)

    if len(statistics) == 0:
        raise RuntimeError("All ensemble matrices already exist")

    for plan_id, district_data in enumerate(iterate_plans_district_data(graph, plans, node_data)):
        for computer in computers:
            computer[1](plan_id, district_data)

    for computer in computers:
        computer[2]()


def calculate_flips(node_ids: list[str], previous_assignment: dict[str, int],
                    assignment: dict[str, int]) -> dict[str, int]:
    return {x: assignment[x] for x in node_ids if previous_assignment[x] != assignment[x]}


def save_ensemble_vra_matrices(chamber: str, directory: str, graph: nx.Graph, ensemble_description: str,
                               use_unique_plans: bool, file_numbers: Optional[Iterable[int]],
                               district_offset: int) -> None:
    print(f"Loading Plans")
    plans = load_plans(directory, ensemble_description, use_unique_plans, file_numbers, district_offset)

    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
    for i, ensemble_matrices in build_ensemble_vra_matrices(chamber, directory, graph, plans):
        for model, model_dict in ensemble_matrices.items():
            for racial_group, ensemble_matrix in model_dict.items():
                print(f"Saving: {model} {racial_group}")
                path = build_vra_effectiveness_path(ensemble_directory, model, racial_group, 'npz')
                np.savez_compressed(path, ensemble_matrix[0:(i + 1)])


def build_ensemble_vra_matrices(chamber: str, directory: str, graph: nx.Graph,
                                plans: np.ndarray) -> Iterable[tuple[int, dict[str, dict[str, np.ndarray]]]]:
    def store_values(plan_number: int, model: str, probabilities: dict[int, tuple],
                     ensemble_matrices: dict[str, dict[str, np.ndarray]], indices: dict[str, int]) -> None:
        def store_changed_values(racial_group: str, index: int) -> None:
            for district, district_probabilities in probabilities.items():
                ensemble_matrices[model][racial_group][plan_number][district] = district_probabilities[index]

        if plan_number == 0:
            districts_range = range(0, len(probabilities))
            for racial_group, index in indices.items():
                ensemble_matrices[model][racial_group][plan_number] = [probabilities[x][index] for x in districts_range]
        else:
            for racial_group, index in indices.items():
                ensemble_matrices[model][racial_group][plan_number] = ensemble_matrices[model][racial_group][
                    plan_number - 1]
                store_changed_values(racial_group, index)

    number_plans = len(plans)
    min_district = np.min(plans)
    max_district = np.max(plans)
    print(f'Min District: {min_district} Max District: {max_district}')
    if min_district != 0:
        raise RuntimeError("VRA code expects minimum district to be zero")

    node_data = load_vra_county_countyvtd_data(chamber, directory)
    node_data['geoid'] = node_data.index

    graph = gerrychain.Graph(graph)

    print(f"Processing Plans: {number_plans}")
    node_ids = list(sorted(graph.nodes().keys()))
    node_data = node_data[node_data['geoid'].isin(set(node_ids))]
    print(f"{len(node_ids)} {len(node_data)}")

    candidates, elections, final_elec_model = vra.load_elec_model(max_district + 1, node_data)
    updaters = vra.build_updaters(candidates, elections, final_elec_model)

    graph.join(node_data)

    ensemble_matrices = {x: {y: np.zeros((number_plans, max_district + 1)) for y in build_vra_racial_groups()}
                         for x in build_vra_models()}

    # important for the final_elec_model routine to have an integer based index (just for dist?)
    node_data.reset_index(inplace=True)

    # Each dictionary maps district -> (Latino, Black, Neither, Overlap)
    indices = {x: y for x, y in zip(build_vra_racial_groups(), [1, 0, 3])}

    for i, partition in enumerate(iterate_partitions(graph, updaters, plans)):
        final_state_prob, final_equal_prob, final_dist_prob = final_elec_model(partition)
        for model, probabilities in zip(build_vra_models(), [final_state_prob, final_equal_prob, final_dist_prob]):
            store_values(i, model, probabilities, ensemble_matrices, indices)

        if ((i + 1) % 1000) == 0 or (i + 1) == number_plans:
            yield i, ensemble_matrices


def iterate_partitions(graph: nx.Graph, updaters, plans: np.ndarray) -> Iterable[gerrychain.Partition]:
    node_ids = list(sorted(graph.nodes().keys()))

    previous_assignment = None
    previous_partition = None
    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} {i}")

        assignment = {x: y for x, y in zip(node_ids, plan)}
        if previous_assignment is None:
            partition = gerrychain.Partition(graph, assignment=assignment, updaters=updaters)
        else:
            flips = None if previous_assignment is None else calculate_flips(node_ids, previous_assignment, assignment)
            partition = gerrychain.Partition(parent=previous_partition, flips=flips)

        yield partition

        previous_assignment = assignment
        if previous_partition is not None:
            previous_partition.parent = None
        previous_partition = partition


def build_vra_effectiveness_path(directory: str, model: str, racial_group: str, file_extension: str,
                                 suffix: str = '') -> str:
    return f'{directory}{build_vra_effectiveness_filename(model, racial_group, file_extension, suffix)}'


def build_vra_effectiveness_filename(model: str, racial_group: str, file_extension: str, suffix: str = '') -> str:
    return f'{build_vra_effectiveness_filename_prefix(model, racial_group)}{f"_{suffix}" if suffix != "" else ""}.{file_extension}'


def build_vra_effectiveness_filename_prefix(model: str, racial_group: str) -> str:
    return f'{racial_group}_CVAP_{VRA_PREFIX}{model}'


def build_vra_aggregate_model_racial_groups() -> list[tuple[str, str]]:
    aggregate_racial_groups = build_vra_aggregate_racial_group_definitions()
    racial_groups = build_vra_racial_groups() + list(aggregate_racial_groups.keys())
    models = build_vra_models()
    return [(x, y) for x in models for y in racial_groups]


def build_vra_models() -> list[str]:
    return ['state', 'equal']  # , 'dist']


def build_vra_racial_groups() -> list[str]:
    return ['BA', 'HA', 'BHO']


def build_vra_aggregate_racial_group_definitions() -> dict[str, list[str]]:
    return {
        'B': ['BA', 'BHO'],
        'H': ['HA', 'BHO'],
        'BH': ['BA', 'HA', 'BHO']
    }


def iterate_plans_district_data(graph: nx.Graph, plans: np.ndarray, node_data: pd.DataFrame) -> Iterable[pd.DataFrame]:
    node_ids = list(sorted(graph.nodes()))
    print(f"Original length of Redistricting Data: {len(node_data)}  Number Nodes: {len(node_ids)}")
    node_ids_lookup = set(node_ids)

    missing_node_data = [x for x in node_ids if x not in node_data['geoid']]
    if any(missing_node_data):
        print(f"Missing data for nodes {len(missing_node_data)}:")
        for geoid in sorted(missing_node_data):
            print(geoid)

    node_data = node_data[node_data['geoid'].isin(node_ids_lookup)]

    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} {i}")

        if i == 0:
            print(f"Plan - Min District: {min(plan)} Max District: {max(plan)}")
            number_redistricting_data = len(node_data)
            plan_length = len(plan)
            print(f"Redistricting Data: {number_redistricting_data} Plan: {plan_length}")
            if number_redistricting_data != plan_length:
                raise RuntimeError("Length of redistricting data and first plan do not match.")

        node_data['assignment'] = plan
        district_data = node_data.groupby('assignment').sum()
        district_data.sort_index(inplace=True)

        yield district_data


def load_plans(directory: str, ensemble_description: str, use_unique_plans: bool,
               file_numbers: Optional[Iterable[int]], district_offset: int) -> np.ndarray:
    ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
    if use_unique_plans:
        plans = district_offset + cm.load_plans_from_path(f'{ensemble_directory}unique_plans.npz')
    else:
        assert isinstance(file_numbers, Iterable)
        plans = cm.load_plans_from_files(directory, ensemble_description, file_numbers, district_offset)
    return plans


def save_unique_plans(ensemble_directory: str, plans: np.ndarray) -> None:
    unique_plans = cm.determine_unique_plans(plans)
    print(f"Number Unique Plans: {len(unique_plans)}")
    np.savez_compressed(f'{ensemble_directory}/unique_plans.npz', np.array(unique_plans))


def convert_to_csv(ensemble_directory: str, filename_prefix: str) -> None:
    array = cm.load_numpy_compressed(f'{ensemble_directory}{filename_prefix}.npz')
    cm.save_numpy_csv(f'{ensemble_directory}{filename_prefix}.csv', array)


if __name__ == '__main__':
    def main() -> None:
        print('start')

        directory = 'G:/rob/projects/election/rob/'

        chamber = 'USCD'  # 'TXSN'  # 'TXHD'  # 'DCN'  #

        settings = cm.build_settings(chamber)

        if False:
            combine_and_fix_redistricting_data_file(directory)

        if False:
            plans = cm.load_plans_from_files(directory, settings.ensemble_description, range(0, settings.number_files),
                                             settings.district_offset)
            ensemble_directory = cm.build_ensemble_directory(directory, settings.ensemble_description)
            save_unique_plans(ensemble_directory, plans)

        if False:
            save_county_vtd_lookup(directory)

        if False:
            # for chamber in ['TXSN', 'USCD']:
            #    save_cvap_county_countyvtd_data(chamber, directory)
            save_cvap_data('TXHD', directory, None)
            # save_cvap_data('DCN', directory, 93173)

        if False:
            save_redistricting_county_vtd_data(directory)

        if True:
            for chamber in cm.CHAMBERS + ['DCN']:
                settings = cm.build_settings(chamber)

                seeds_directory = cm.build_seeds_directory(directory)
                dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)

                save_ensemble_data(chamber, directory, settings.redistricting_data_filename,
                                   settings.cvap_data_filename, dual_graph, settings.ensemble_description,
                                   False, range(0, settings.number_files),
                                   settings.district_offset, False)

                if False:
                    ensemble_directory = cm.build_ensemble_directory(directory, settings.ensemble_description)
                    for population_group in build_population_groups(chamber):
                        convert_to_csv(ensemble_directory, f'pop_{population_group}')

        if False:
            save_vra_county_countyvtd_data(chamber, directory)

        if False:
            seeds_directory = cm.build_seeds_directory(directory)
            dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)

            save_ensemble_vra_matrices('USCD', directory, dual_graph, settings.ensemble_description,
                                       False, range(0, settings.number_files), settings.district_offset)

        if False:
            source = ''
            raw_input_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/RawInput' + source + '/'
            root_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/Input' + source + '/'
            data_directories = [raw_input_directory + x for x in
                                ['Outputs_USCD_2020Census/', 'Outputs_USCD_2020Census_A/']]

            merged_data_directory = root_directory + chamber + '/'
            cm.ensure_directory_exists(merged_data_directory)

            merge_ensemble_matrices(data_directories, merged_data_directory)

            # df = pd.read_parquet(merged_data_meeting_path)
            # save_numpy_arrays(chamber, df, merged_data_directory)

        print('done')


    main()
