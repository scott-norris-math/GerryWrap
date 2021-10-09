from os import listdir
from os.path import isfile
from collections import defaultdict
import pandas as pd
from typing import Callable, Iterable

import common as cm
from timer import Timer
import utilities as ut


def save_array_csv(chamber: str, df: pd.DataFrame, path: str, statistic_func) -> None:
    statistic = statistic_func(df)
    pivoted_df = df.pivot(['seed', 'plan'], get_bq_chamber_name(chamber), statistic)
    # print(pivoted_df.head())
    numpy_array = pivoted_df.to_numpy(copy=True)
    cm.save_numpy_csv(path, numpy_array)


def get_bq_chamber_name(chamber: str) -> str:
    return {
        'USCD': 'cd',
        'TXSN': 'sldu',
        'TXHD': 'sldl'
    }[chamber]


def merge_plan_data(input_directories: Iterable[str], output_directory: str) -> None:
    masks = ['_perc', 'votes_']
    data_files = [(x, y) for x in input_directories for y in listdir(x)
                  if isfile(x + y) and any(mask in y for mask in masks)]

    file_groups = defaultdict(list)

    print(data_files)
    print(file_groups)

    for directory, filename in data_files:
        file_groups[filename].append(directory)

    for filename, directories in file_groups.items():
        print('Merging:' + filename)
        merged_array = load_merged_numpy_csv(filename, directories)
        cm.save_numpy_csv(output_directory + filename, merged_array)


def determine_data_columns(df: pd.DataFrame) -> list[str]:
    race_columns = [col for col in df.columns if (col.startswith('o17'))]
    election_columns = [
        'President_2020_general_D_Biden',
        'President_2020_general_R_Trump',
        'USSen_2020_general_D_Hegar',
        'USSen_2020_general_R_Cornyn'
    ]
    return ut.union(race_columns, election_columns)


def get_common_columns(chamber: str) -> list[str]:
    return ['seed', 'plan', get_bq_chamber_name(chamber), 'total_pop']


def o17_pop(df: pd.DataFrame) -> pd.Series:
    return df['o17_pop']


def total_pop(df: pd.DataFrame) -> pd.Series:
    return df['total_pop']


def hisp_percent(df: pd.DataFrame) -> pd.Series:
    return df['o17_hisp_pop'] / df['o17_pop']


def black_percent(df: pd.DataFrame) -> pd.Series:
    race_sum = black_sum(df)
    return race_sum / df['o17_pop']


def black_sum(df):
    entire_population_o17_columns = [col for col in df.columns
                                     if (col.startswith('o17') and not col.startswith('o17_nonhisp'))]
    race_pop_columns = [col for col in entire_population_o17_columns if 'black' in col]
    return df[race_pop_columns].sum(axis=1)


def black_hisp_percent(df: pd.DataFrame) -> pd.Series:
    race_sum = black_hisp_sum(df)
    return race_sum / df['o17_pop']


def black_hisp_sum(df):
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
        'BHVAP': "black_hisp"
    }[racial_group]


def build_race_filename_prefix(race: str) -> str:
    return race + '_perc'


def build_race_filename_csv(race: str, suffix: str = '') -> str:
    suffix = '' if suffix == '' else '_' + suffix
    return build_race_filename_prefix(race) + suffix + '.csv'


def build_election_filename_prefix(election: str) -> str:
    return 'votes_' + election


def build_election_filename_csv(election: str, suffix: str = '') -> str:
    suffix = '' if suffix == '' else '_' + suffix
    return build_election_filename_prefix(election) + suffix + '.csv'


def build_statistics_settings() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (build_race_filename_csv('hisp'), hisp_percent),
        (build_race_filename_csv('black_hisp'), black_hisp_percent),
        (build_race_filename_csv('black'), black_percent),
        (build_election_filename_csv('PRES20'), pres20_percent),
        (build_election_filename_csv('SEN20'), sen20_percent),
    ]


def save_numpy_array_csvs(chamber: str, df: pd.DataFrame, output_directory: str) -> None:
    print(len(df) / cm.get_number_districts(chamber))
    array_settings = build_statistics_settings()

    for filename, statistic_func in array_settings:
        path = output_directory + filename
        print('Saving: ' + path)
        save_array_csv(chamber, df, path, statistic_func)


def build_canonical_assignments_list(assignments: list[tuple[int, int]]) -> list[list[int]]:
    print(assignments)
    partition_dictionary = defaultdict(list)
    for unit, district in assignments:
        partition_dictionary[district].append(unit)
    partition_list = [sorted(x) for x in partition_dictionary.values()]
    print(partition_list)
    partition_list.sort(key=lambda x: x[0])
    return partition_list


if __name__ == '__main__':
    def main():
        print('start')

        t = Timer()
        t.start()

        if True:
            # build_plan_data()
            pass

        if False:
            chamber = 'TXSN'  # 'USCD'  #
            source = ''
            raw_input_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/RawInput' + source + '/'
            root_directory = '/home/user/election/MeetingPrep/PostMortem/Plots/Input' + source + '/'
            data_directories = [raw_input_directory + x for x in
                                  ['Outputs_USCD_2020Census/', 'Outputs_USCD_2020Census_A/']]

            merged_data_directory = root_directory + chamber + '/'
            ensure_directory_exists(merged_data_directory)

            merge_data(data_directories, merged_data_directory)

        # df = pd.read_parquet(merged_data_meeting_path)
        # save_numpy_arrays(chamber, df, merged_data_directory)

        if False:
            assignments = [('c', 1), ('b', 1), ('a', 3), ('q', 4), ('aa', 1)]
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
