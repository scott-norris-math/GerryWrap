import os
import pandas as pd
from typing import Callable
import glob
from shutil import copy2
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from operator import itemgetter

import data_transform as dt
import utilities as ut
from timer import Timer
import common as cm

SCTB_COLUMN_NAME = 'SCTBKEY'


def build_columns_path(chamber: str, directory: str) -> str:
    return build_redistricting_data_calculated_directory(directory) + chamber + "_columns.csv"


def determine_data_tablock_columns(df: pd.DataFrame) -> list[str]:
    race_columns = [col for col in df.columns if (col.startswith('o17'))]
    election_columns = [
        'President_2020_D_Biden_general',
        'President_2020_R_Trump_general',
        'USSen_2020_D_Hegar_general',
        'USSen_2020_R_Cornyn_general'
    ]
    return ut.union(race_columns, election_columns)


def save_data_tablock_columns(chamber: str, directory: str, data_path: str) -> None:
    data_df = pd.read_parquet(data_path)
    print("Data: " + str(len(data_df)))
    data_columns = ['geoid', 'county', dt.get_census_chamber_name(chamber), 'total_pop'] + \
                   determine_data_tablock_columns(data_df)

    print("Data Columns: " + str(len(data_columns)))
    print(data_columns)
    columns_df = pd.DataFrame(data_columns, columns=["column"])
    columns_df.to_csv(build_columns_path(chamber, directory))


def save_filtered_data(chamber: str, directory: str, data_path: str) -> None:
    columns = list(pd.read_csv(build_columns_path(chamber, directory))['column'])
    print(str(len(columns)) + " " + str(columns))

    data_df = pd.read_parquet(data_path, columns=columns)
    data_df.sort_values(by=['geoid'], inplace=True)
    print("Filtered Data: " + str(len(data_df)))

    data_df.to_parquet(build_data_filtered_path(chamber, directory, 'parquet'))
    data_df.to_csv(build_data_filtered_path(chamber, directory, 'csv'))


def build_data_filtered_path(chamber: str, directory: str, suffix: str) -> str:
    data_filtered_path_prefix = \
        build_redistricting_data_calculated_directory(directory) + \
        'redistricting_data_nodes_TX_nodes_TX_2020_tabblock_' + \
        chamber + '_filtered'
    return data_filtered_path_prefix + "." + suffix


def process_tabblock_data(chamber: str, directory: str) -> None:
    bq_chamber_name = dt.get_census_chamber_name(chamber)
    data_path = \
        build_redistricting_data_directory(directory) + \
        'redistricting_data_nodes_TX_nodes_TX_2020_tabblock_' + \
        bq_chamber_name + '_contract0.parquet'
    save_data_tablock_columns(chamber, directory, data_path)
    save_filtered_data(chamber, directory, data_path)


def load_block_equivalency_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={SCTB_COLUMN_NAME: str, 'DISTRICT': int})


def build_plans_raw_directory(directory: str) -> str:
    return f'{directory}plans_raw/'


def build_plan_raw_path(chamber: str, directory: str, plan: int) -> str:
    return f'{build_plans_raw_directory(directory)}PLAN{cm.encode_chamber_character(chamber)}{plan}.csv'


def build_plans_directory(directory: str) -> str:
    return f'{directory}plans/'


def build_plan_metadata_path(chamber: str, plan_directory: str) -> str:
    return f'{plan_directory}plans_metadata_{chamber}.csv'


def build_plan_path(chamber: str, directory: str, plan: int) -> str:
    return f'{build_plans_directory(directory)}PLAN{cm.encode_chamber_character(chamber)}{plan}.csv'


def normalize_block_equivalency_df(bef_df: pd.DataFrame) -> None:
    if SCTB_COLUMN_NAME in bef_df.columns:
        bef_df['geoid'] = [str(x)[2:] for x in bef_df[SCTB_COLUMN_NAME]]
        bef_df.drop(columns=[SCTB_COLUMN_NAME], inplace=True)


def merge_block_equivalence_files(chamber: str, directory: str, source_plan: int, diff_plan: int) -> (
        pd.DataFrame, int):
    source_bef_df = load_block_equivalency_file(build_plan_path(chamber, directory, source_plan))
    diff_bef_df = load_block_equivalency_file(build_plan_raw_path(chamber, directory, diff_plan))
    source_bef_df.set_index([SCTB_COLUMN_NAME], inplace=True)
    diff_bef_df.set_index([SCTB_COLUMN_NAME], inplace=True)
    joined_source_bef_df = source_bef_df.join(diff_bef_df, how='inner', on=SCTB_COLUMN_NAME, lsuffix='_source',
                                              rsuffix='_diff')
    joined_source_bef_df['DISTRICT'] = [x if y == 0 else y for x, y in
                                        zip(joined_source_bef_df['DISTRICT_source'],
                                            joined_source_bef_df['DISTRICT_diff'])]
    changed_rows = [not x == y for x, y in
                    zip(joined_source_bef_df['DISTRICT'], joined_source_bef_df['DISTRICT_source'])]
    number_changed_rows = sum(changed_rows)
    print(f"Merged: {source_plan} {diff_plan}  Changed Rows: {number_changed_rows}")
    joined_source_bef_df.drop(columns=['DISTRICT_source', 'DISTRICT_diff'], inplace=True)
    return joined_source_bef_df, number_changed_rows


def compare_block_equivalence_files(chamber: str, directory: str, source_plan: int, target_plan: int) -> (
        pd.DataFrame, int):
    source_bef_df = load_block_equivalency_file(build_plan_path(chamber, directory, source_plan))
    target_bef_df = load_block_equivalency_file(build_plan_path(chamber, directory, target_plan))
    source_bef_df.set_index([SCTB_COLUMN_NAME], inplace=True)
    target_bef_df.set_index([SCTB_COLUMN_NAME], inplace=True)
    joined_source_bef_df = source_bef_df.join(target_bef_df, how='inner', on=SCTB_COLUMN_NAME, lsuffix='_source',
                                              rsuffix='_target')
    joined_source_bef_df['DISTRICT'] = [x if y == 0 else y for x, y in
                                        zip(joined_source_bef_df['DISTRICT_source'],
                                            joined_source_bef_df['DISTRICT_target'])]
    changed_rows = [not x == y for x, y in
                    zip(joined_source_bef_df['DISTRICT'], joined_source_bef_df['DISTRICT_source'])]

    return sum(changed_rows)


def save_merged_plans(chamber: str, directory: str, source_plan: int, diff_plan: int, plans_metadata) -> None:
    merged_plan, number_changed_rows = merge_block_equivalence_files(chamber, directory, source_plan, diff_plan)
    output_path = build_plan_path(chamber, directory, diff_plan)
    merged_plan.to_csv(output_path)

    # plans_metadata = load_plans_metadata(chamber, build_plans_directory(directory))
    plans_metadata.at[diff_plan, 'changed_rows'] = number_changed_rows
    save_plans_metadata(chamber, build_plans_directory(directory), plans_metadata)


def save_plan_data(chamber: str, directory: str, plan: int) -> None:
    bef_df = load_block_equivalency_file(build_plan_path(chamber, directory, plan))
    normalize_block_equivalency_df(bef_df)

    data_filtered_path = build_data_filtered_path(chamber, directory, 'parquet')
    data_df = pd.read_parquet(data_filtered_path)
    print("Filtered Data: " + str(len(data_df)))
    print(data_df.head())

    bef_df.set_index(['geoid'], inplace=True)
    data_df.set_index(['geoid'], inplace=True)
    joined_df = bef_df.join(data_df, how='inner', on='geoid')
    if not len(joined_df) == len(bef_df):
        raise RuntimeError("Invalid Length")

    joined_path_prefix = \
        build_redistricting_data_calculated_directory(directory) + \
        'redistricting_data_nodes_TX_nodes_TX_2020_tabblock_' + \
        chamber + '_filtered_' + str(plan)
    joined_path_parquet = joined_path_prefix + '.parquet'
    joined_df.to_parquet(joined_path_parquet)
    joined_df.to_csv(joined_path_prefix + ".csv")

    bq_chamber_name = dt.get_census_chamber_name(chamber)
    joined_df.drop(columns=['county', bq_chamber_name], inplace=True)
    joined_df['DISTRICT'] = joined_df['DISTRICT'].astype(str)
    grouped_by_district = joined_df.groupby('DISTRICT')

    plan_df = grouped_by_district.sum()
    plan_df.sort_values('DISTRICT', ascending=True, inplace=True)
    print(f"{len(plan_df)}")

    plan_df.to_parquet(build_plan_data_path(chamber, directory, plan, 'parquet'))
    plan_df.to_csv(build_plan_data_csv_path(directory, chamber, plan))


def build_plan_data_csv_path(directory: str, chamber: str, plan: int) -> str:
    return build_plan_data_path(chamber, directory, plan, 'csv')


def build_plan_data_path(chamber: str, directory: str, plan: int, suffix: str) -> str:
    aggregated_path_prefix = \
        build_redistricting_data_calculated_directory(directory) + \
        'redistricting_data_nodes_TX_nodes_TX_2020_' + \
        chamber + '_' + str(plan)
    return aggregated_path_prefix + '.' + suffix


def build_redistricting_data_directory(directory: str) -> str:
    return directory + 'redistricting_data/'


def build_redistricting_data_calculated_directory(directory: str) -> str:
    return directory + 'redistricting_data_calculated/'


def pres20_percent_vector(df: pd.DataFrame) -> pd.Series:
    dem_votes = df['President_2020_D_Biden_general']
    rep_votes = df['President_2020_R_Trump_general']
    return dem_votes / (dem_votes + rep_votes)


def sen20_percent_vector(df: pd.DataFrame) -> pd.Series:
    dem_votes = df['USSen_2020_D_Hegar_general']
    rep_votes = df['USSen_2020_R_Cornyn_general']
    return dem_votes / (dem_votes + rep_votes)


def build_statistics_vector_settings() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (dt.build_race_filename_csv('hisp', 'vector'), dt.hisp_percent),
        (dt.build_race_filename_csv('black_hisp', 'vector'), dt.black_hisp_sum_percent),
        (dt.build_race_filename_csv('black', 'vector'), dt.black_sum_percent),
        (dt.build_election_filename_csv('PRES20', 'vector'), pres20_percent_vector),
        (dt.build_election_filename_csv('SEN20', 'vector'), sen20_percent_vector),
        ('o17_pop_vector.csv', dt.o17_pop)
    ]


def save_vector_file(chamber: str, df: pd.DataFrame, path: str,
                     statistic_func: Callable[[pd.DataFrame], pd.Series]) -> None:
    df['district'] = df.index
    districts = df['district'].astype(int)
    number_districts = len(districts)
    max_district = max(districts)
    print(f"Districts: {number_districts} {max_district}")
    if not number_districts == max_district:
        raise RuntimeError(f'max district does not match number of districts in: {path}')

    if max_district not in cm.get_allowed_number_districts(chamber):
        raise RuntimeError(f'max district is not an allowed number of districts: {path}')

    statistics_lookup = {x: y for x, y in zip(df['district'], statistic_func(df))}
    statistics_list = [statistics_lookup[str(x)] for x in range(1, max_district + 1)]
    # print(f"Statistics: {len(statistics_list)} {statistics_list}")

    cm.save_vector_csv(path, statistics_list)


def save_plan_vectors(chamber: str, directory: str, plan: int):
    plan_df = pd.read_parquet(build_plan_data_path(chamber, directory, plan, 'parquet'))
    output_directory = build_plan_vectors_directory(chamber, directory, plan)
    ut.ensure_directory_exists(output_directory)
    vector_settings = build_statistics_vector_settings()

    for filename, statistic_func in vector_settings:
        path = output_directory + filename
        if not os.path.exists(path):
            print("Saving: " + path)
            save_vector_file(chamber, plan_df, path, statistic_func)


def build_plan_vectors_directory(chamber: str, directory: str, plan: int) -> str:
    return f'{directory}plan_vectors/vectors_PLAN{cm.encode_chamber_character(chamber)}{plan}/'


def get_known_plans(chamber: str, plans_raw_directory: str) -> set[int]:
    plans = set()
    for path in glob.glob(f'{plans_raw_directory}plan{cm.encode_chamber_character(chamber)}*_blk.zip'):
        path = os.path.normpath(path)
        plan_string = str(path.replace(os.path.dirname(path), '').removesuffix('_blk.zip')[1:]). \
            removeprefix(f'plan{cm.encode_chamber_character(chamber).lower()}')
        plans.add(int(plan_string))
    return plans


def save_current_merged_plans(chamber: str, directory: str, plans_metadata: pd.DataFrame, force=False) -> None:
    plans_directory = build_plans_directory(directory)
    ut.ensure_directory_exists(plans_directory)

    for plan_metadata in plans_metadata.itertuples():
        plan = plan_metadata.plan
        previous_plan = plan_metadata.previous_plan
        plan_path = build_plan_path(chamber, directory, plan)
        if force or not os.path.exists(plan_path):
            print(f'Merging Plan: {plan}')
            if previous_plan == 0:
                diff_bef_df = load_block_equivalency_file(build_plan_raw_path(chamber, directory, plan))
                if any(diff_bef_df['DISTRICT'] == 0):
                    print(f"INVALID PLAN {plan}")
                    plans_metadata.at[plan, 'invalid'] = 1
                    save_plans_metadata(chamber, build_plans_directory(directory), plans_metadata)

                copy2(build_plan_raw_path(chamber, directory, plan), plan_path)
            else:
                save_merged_plans(chamber, directory, previous_plan, plan, plans_metadata)


def save_current_plan_vectors(chamber: str, directory: str) -> None:
    ut.ensure_directory_exists(directory + 'plan_vectors/')
    plans_directory = build_plans_directory(directory)
    plans_metadata = load_plans_metadata(chamber, plans_directory)

    for plan_metadata in plans_metadata.itertuples():
        if plan_metadata.invalid:
            continue

        plan = plan_metadata.plan
        # TODO: check the existence of the vectors
        if not os.path.exists(build_plan_data_csv_path(directory, chamber, plan)):
            print(f"Processing Plan Data/Vector: {plan}")
            save_plan_data(chamber, directory, plan)
            save_plan_vectors(chamber, directory, plan)


def save_plan_vectors_summaries(chamber: str, directory: str) -> None:
    plans_directory = build_plans_directory(directory)
    plans_metadata = load_plans_metadata(chamber, plans_directory)

    df = pd.DataFrame(columns=['plan', 'description', 'submitter', 'previous_plan', 'statistic', 'district', 'value'])
    districts_range = range(1, cm.get_number_districts(chamber) + 1)
    df_pivoted = pd.DataFrame(columns=['plan', 'statistic'] + list(map(lambda x: str(x), districts_range)))
    for plan_metadata in plans_metadata.itertuples():
        if plan_metadata.invalid:
            continue

        vector_settings = build_statistics_vector_settings()
        plan = plan_metadata.plan
        if chamber == 'USCD' and plan == 2100:
            continue

        plan_directory = build_plan_vectors_directory(chamber, directory, plan)
        for statistic, _ in vector_settings:
            vector = cm.load_numpy_csv(plan_directory + statistic)
            row = {'plan': plan, 'description': plan_metadata.description,
                   'submitter': plan_metadata.submitter, 'previous_plan': plan_metadata.previous_plan,
                   'statistic': statistic}
            row_pivoted = {'plan': plan, 'description': plan_metadata.description,
                           'submitter': plan_metadata.submitter, 'previous_plan': plan_metadata.previous_plan,
                           'statistic': statistic}
            for district in districts_range:
                row['district'] = district
                row['value'] = vector[district - 1]
                row_pivoted[str(district)] = vector[district - 1]
                df = df.append(row, ignore_index=True)

            df_pivoted = df_pivoted.append(row_pivoted, ignore_index=True)
    df.sort_values(by=['statistic', 'plan', 'district'], inplace=True)
    df_pivoted.sort_values(by=['statistic', 'plan'], inplace=True)
    df.to_csv(f'{directory}plan_vectors/current_vectors_{chamber}.csv', index=False)
    df_pivoted.to_csv(f'{directory}plan_vectors/current_vectors_{chamber}_pivoted.csv', index=False)


def load_plans_metadata(chamber: str, plans_directory: str) -> pd.DataFrame:
    plans_metadata = pd.read_csv(build_plan_metadata_path(chamber, plans_directory))
    plans_metadata.set_index('plan', drop=False, inplace=True)
    return plans_metadata


def save_plans_metadata(chamber: str, plans_directory: str, plans_metadata: pd.DataFrame) -> None:
    plans_metadata.to_csv(build_plan_metadata_path(chamber, plans_directory), index=False)


def determine_valid_plans(plans_metadata_df: pd.DataFrame) -> set[int]:
    return {x.plan for x in plans_metadata_df.itertuples() if not x.invalid}


def get_valid_plans(chamber, plans_directory) -> set[int]:
    return determine_valid_plans(load_plans_metadata(chamber, plans_directory))


def update_plan_vectors(chamber: str, directory: str) -> None:
    save_current_plan_vectors(chamber, directory)
    save_plan_vectors_summaries(chamber, directory)


def build_graph_path(chamber: str, directory: str) -> str:
    return f'{directory}plans_graph_{chamber}.png'


def save_graph(chamber: str, directory: str, plans_metadata: pd.DataFrame) -> None:
    def parse_submitter(submitter: str) -> (bool, str):
        elements = submitter.split(" ")
        cleaned_title = elements[0].removesuffix(".")
        if len(elements) == 2:
            name = elements[1]
        elif len(elements) == 3:
            name = elements[1][0] + " " + elements[2]
        else:
            error = f"Unknown Name Format - {submitter}"
            raise RuntimeError(error)

        if cleaned_title in ["SEN", "REP"]:
            return cleaned_title, name
        else:
            return None, submitter

    def determine_node_color(submitter: (str, str), party_lookup: dict[str, dict[str, str]], is_max_plan: bool) -> str:
        title = submitter[0]
        last_name = submitter[1]
        party = None
        if title == 'SEN':
            party = party_lookup['TXSN'].get(last_name)
        elif title == 'REP':
            party = party_lookup['TXHD'].get(last_name)

        if party is None:
            return "grey"

        if is_max_plan:
            return 'cornflowerblue' if party == 'D' else 'tomato'
        else:
            return 'aliceblue' if party == 'D' else 'mistyrose'

    def build_multiline(s: str) -> str:
        s = s.replace("/", " ")
        elements = []
        for element in s.split(" "):
            if "," not in element:
                elements.append(element)
            else:
                subelements = element.split(",")
                chunk_size = 3
                chunks = [subelements[x:x + chunk_size] for x in range(0, len(subelements), chunk_size)]
                elements.append("\n".join(f"{','.join(chunk)}" for chunk in chunks))
        return "\n".join(elements)

    previous_plans = set(plans_metadata['previous_plan'])
    max_ammendments = plans_metadata.filter(['plan', 'submitter', 'previous_plan']). \
        query('previous_plan > 0').groupby(by=['submitter', 'previous_plan']).aggregate(max).to_dict()['plan']

    add_overridden_ammendments = False

    proposed_plans_graph = nx.DiGraph()
    for plan_metadata in plans_metadata.itertuples():
        plan = plan_metadata.plan
        previous_plan = plan_metadata.previous_plan

        overridden_ammendment = plan not in previous_plans and previous_plan != 0 and max_ammendments.get(
            (plan_metadata.submitter, plan_metadata.previous_plan)) != plan
        if not add_overridden_ammendments and overridden_ammendment:
            continue

        proposed_plans_graph.add_node(plan)
        if previous_plan != 0:
            proposed_plans_graph.add_edge(previous_plan, plan, weight=plan_metadata.changed_rows)

    party_lookup = build_party_lookup(directory)
    admissible_metadata = [x for x in plans_metadata.itertuples() if x.plan in proposed_plans_graph.nodes]
    submitters = {x.submitter: parse_submitter(x.submitter) for x in admissible_metadata}

    layout = 'fdp' if chamber == 'TXHD' else 'dot'
    pos = nx.nx_pydot.graphviz_layout(proposed_plans_graph, prog=layout)
    labels = {x.plan: f"{x.plan}\n{build_multiline(submitters[x.submitter][1])}" for x in admissible_metadata}
    node_colors = [determine_node_color(submitters[x.submitter], party_lookup,
                                        x.previous_plan == 0 or
                                        max_ammendments.get((x.submitter, x.previous_plan)) == x.plan)
                   for x in admissible_metadata]

    if chamber == 'TXHD':
        height = 16
        node_size = 4500
    else:
        height = 8
        node_size = 5500
    fig, ax = plt.subplots(figsize=(22, height))
    nx.draw_networkx(proposed_plans_graph, pos, node_size=node_size, labels=labels, node_color=node_colors)
    edge_labels = {x: "{:,}".format(y) for x, y in nx.get_edge_attributes(proposed_plans_graph, 'weight').items()}
    nx.draw_networkx_edge_labels(proposed_plans_graph, pos, edge_labels=edge_labels)

    fig.tight_layout()

    pylab.savefig(build_graph_path(chamber, directory))


def build_party_lookup(directory: str) -> dict[str, dict[str, str]]:
    def normalize(elements: list[str]) -> tuple[str, str]:
        name_parts = [x for x in elements if x.removesuffix(".") not in ["Jr", "III"]]
        return name_parts[0], name_parts[-1]

    def build_chamber_party_lookup(chamber: str, directory: str):
        legislator_lookup = pd.read_csv(f'{directory}{chamber}PartyLookup.csv')
        split_names = [normalize(x.split(' ')) for x in legislator_lookup['Name'] if x != 'Vacant']
        names = [(elements[0], elements[1]) for elements in split_names]
        names_grouped = cm.groupby_project(names,
                                           lambda x: x[1], lambda x: x[0])

        single_name_mapping = {(y[0] + ' ' + x): x.upper() for x, y in names_grouped if len(y) == 1}
        repeated_name_mapping = {(first_name + ' ' + x): (first_name[0] + ' ' + x).upper() for x, y in names_grouped if
                                 len(y) >= 2 for first_name in y}

        name_lookup = {**single_name_mapping, **repeated_name_mapping}
        return {name_lookup[' '.join(normalize(x.split(' ')))]: y[0] for x, y in
                zip(legislator_lookup['Name'], legislator_lookup['Party']) if x != 'Vacant'}

    return {chamber: build_chamber_party_lookup(chamber, directory) for chamber in cm.CHAMBERS}


def analyze_bef_assignments(chamber: str, directory: str, plan: int) -> None:
    bef_df = load_block_equivalency_file(build_plan_path(chamber, directory, plan))
    normalize_block_equivalency_df(bef_df)

    assignments_df = pd.read_parquet(build_redistricting_data_directory(directory) +
                                     'redistricting_data_assignments_TX_assignments_TX_2020.parquet')
    assignments_df['geoid'] = [str(x)[2:] for x in assignments_df['geoid']]

    bef_df.set_index(['geoid'], inplace=True)
    assignments_df.set_index(['geoid'], inplace=True)
    bef_assignments_df = bef_df.join(assignments_df, how='inner', on='geoid')

    if not len(bef_assignments_df) == len(bef_df):
        raise RuntimeError("Invalid Length")

    bef_assignments_df.to_csv(directory + 'bef_assignments_' + plan + '.csv')
    print(len(bef_df))
    print(len(assignments_df))
    print(bef_assignments_df.columns)
    print(bef_assignments_df.head())


def analyze_proposed_plan_seed_assignments(chamber: str, root_directory: str, plan: int) -> dict:
    networkXGraph = nx.read_gpickle(f'{root_directory}seeds/graph_TX_2020_cntyvtd_{chamber}_{plan}.gpickle')
    nodes = networkXGraph.nodes

    node_districts = [(x, y['district'], count_groups(y['node_districts'].split(','))) for x, y in nodes.items()]
    print(node_districts)

    nodes_with_multiple_districts = [(x, y, z) for x, y, z in node_districts if len(z) > 1]
    print(len(nodes_with_multiple_districts))
    print(nodes_with_multiple_districts)

    block_assignments = {a: y for (x, y, z) in node_districts for a in z}
    print(block_assignments)

    return block_assignments


def analyze_proposed_plan_seed_assignments_v2(chamber: str, directory: str, plan: int) -> None:
    graph = nx.read_gpickle(f'{directory}seeds/graph_TX_2020_cntyvtd_{chamber}_{plan}_Reduced.gpickle')

    block_assignments = extract_graph_block_assignments(graph)
    block_assignments_df = pd.DataFrame(block_assignments, columns=['geoid', 'district', 'node_key'])
    block_assignments_df.set_index('geoid', drop=False, inplace=True)
    print(len(block_assignments_df))

    node_data = load_nodes_raw_filtered(directory)
    block_data = block_assignments_df.join(node_data, rsuffix='_node_raw')
    block_data.drop(columns=['geoid_node_raw'], inplace=True)
    district_blocks = block_data.groupby(['district'])
    correct_district_populations = {x: sum(y['total_pop']) for x, y in district_blocks}
    average_population = sum(correct_district_populations.values()) / len(correct_district_populations)
    print(f"Avg Population: {average_population}")
    node_populations = [(node['district'], node['total_pop']) for node in graph.nodes.values()]
    node_populations_groups = groupby_project(node_populations, itemgetter(0), itemgetter(1))
    graph_district_populations = {x: sum(y) for x, y in node_populations_groups}

    # print(graph_district_populations)
    # print(len(graph_district_populations))

    # graph_district_populations = [district, sum() for district, nodes in node_populations.group]
    # print(list(district_blocks))
    # print(correct_district_populations)
    both_populations = join_dict(graph_district_populations, correct_district_populations)
    different_populations = [(x, y, z, y - z, (y - z) / average_population) for x, (y, z) in both_populations.items() if
                             y != z]
    print(sorted(different_populations, key=lambda x: x[4]))


def verify_graph(chamber, directory, plan):
    graph = nx.read_gpickle(f'{directory}seeds/graph_TX_2020_cntyvtd_{chamber}_{plan}_Reduced.gpickle')
    print(f"Connected Components: {nx.number_connected_components(graph)}")
    print([x for x, y in graph.degree() if y == 1])


def load_nodes_raw_filtered(directory):
    filtered_filename = 'nodes_raw_filtered.parquet'
    filtered_path = build_redistricting_data_calculated_directory(directory) + filtered_filename
    if not os.path.exists(filtered_path):
        with Timer(name='load_nodes_raw_filtered'):
            node_data = pd.read_parquet(build_redistricting_data_directory(directory) +
                                        'redistricting_data_nodes_TX_nodes_TX_2020_raw.parquet',
                                        columns=['geoid', 'county', 'cntyvtd', 'total_pop',
                                                 'o17_pop'])  # , 'polygon'])
        node_data.to_parquet(filtered_path)
    else:
        node_data = pd.read_parquet(filtered_path)
    node_data.set_index('geoid', drop=False, inplace=True)
    print(len(node_data))
    return node_data


def extract_graph_block_assignments(graph):
    block_assignments = [(geoid, int(district), key) for key, node in graph.nodes.items()
                         for geoid, district in
                         zip(node['node_geoids'].split(','), node['node_districts'].split(','))]
    # print(block_assignments)
    # reassigned_blocks = [geoid for geoid, district, key, node in block_assignments if district != node['district']]
    # print(len(reassigned_blocks) / len(block_assignments))
    return block_assignments


def save_graph_filtered(chamber, root_directory, proposed_plans_metadata):
    proposed_plans_metadata = proposed_plans_metadata[proposed_plans_metadata['plan'] != 2100]
    if chamber == 'TXHD':
        proposed_plans_metadata = proposed_plans_metadata[(proposed_plans_metadata['plan'] >= 2176) & (
                (proposed_plans_metadata['previous_plan'] >= 2176) | (
                proposed_plans_metadata['previous_plan'] == 0))]
    save_graph(chamber, root_directory, proposed_plans_metadata)


if __name__ == '__main__':
    def main():
        print("start")

        t = Timer()
        t.start()

        chamber = 'TXHD'  # 'TXSN'  # 'USCD'
        root_directory = 'C:/Users/rob/projects/election/rob/'
        plans_directory = build_plans_directory(root_directory)

        if False:
            # process_tabblock_data(chamber, root_directory)
            update_plan_vectors(chamber, root_directory)

        if False:
            pd.set_option('display.width', 500)
            pd.set_option('display.max_columns', 500)
            plan = 2176
            # analyze_bef_assignments(chamber, root_directory, plan)
            # analyze_proposed_plan_seed_assignments(chamber, root_directory, plan)
            # analyze_proposed_plan_seed_assignments_v2(chamber, root_directory, plan)
            verify_graph(chamber, root_directory, plan)

        if False:
            plans_metadata = load_plans_metadata(chamber, plans_directory)
            save_current_merged_plans(chamber, root_directory, plans_metadata, force=True)

        if True:
            for chamber in ['TXHD']:  # cm.CHAMBERS:
                proposed_plans_metadata = load_plans_metadata(chamber, plans_directory)
                save_graph_filtered(chamber, root_directory, proposed_plans_metadata)
            # pylab.show()

        if False:
            save_plan_vectors(chamber, root_directory, 2101)

        if False:
            plans = [x.plan for x in load_plans_metadata(chamber, plans_directory).itertuples()]
            for source_plan in plans:
                number_changed_rows = compare_block_equivalence_files(chamber, root_directory, source_plan, 2130)
                print(f"Compared {source_plan} {2130}  Changed Districts: {number_changed_rows:,}")

        t.stop()

        print("done")


    main()
