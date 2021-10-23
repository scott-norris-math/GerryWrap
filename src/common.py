import os
import csv
import zipfile
from collections import defaultdict
from itertools import groupby
from typing import Iterable, Callable, Any
import numpy as np
import gerrychain
import pickle

from addict import Dict


CHAMBERS = ['TXSN', 'USCD', 'TXHD']


def build_suffix(suffix: str) -> str:
    return '' if suffix == '' else f'_{suffix}'


def load_plans_from_path(path: str) -> np.ndarray:
    return np.load(path)['arr_0']


def load_plans(directory: str, ensemble_description: str, file_number: int) -> np.ndarray:
    return load_plans_from_path(f'{directory}ensembles/ensemble_{ensemble_description}/plans_{file_number}.npz')


def load_plans_from_file(directory: str, ensemble_description: str, plans_filename: str) -> np.ndarray:
    return load_plans_from_path(f'{directory}ensembles/ensemble_{ensemble_description}/{plans_filename}')


def load_plans_from_files(directory: str, ensemble_description: str, file_numbers: Iterable[int]) -> np.ndarray:
    return np.concatenate([load_plans(directory, ensemble_description, x) for x in file_numbers])


def build_assignment(graph: gerrychain.Graph, plan: np.ndarray) -> dict[str, int]:
    return {x: y for x, y in zip(sorted(graph.nodes), plan)}


def save_all_text(text: str, path: str) -> None:
    with open(path, 'w', encoding='utf8') as fp:
        fp.write(text)
    fp.close()


def read_all_text(path: str) -> str:
    file = open(path, mode='r')
    page_text = file.read()
    file.close()
    return page_text


def unzip_file(output_directory: str, zip_path: str) -> None:
    print(f"Unzipping: {zip_path} Start")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_directory)
    print(f"Unzipping: {zip_path} End")


def load_ensemble_matrix_sorted_transposed(input_directory: str, statistic_name: str) -> np.ndarray:
    path = input_directory + statistic_name + '.npz'
    return load_ensemble_matrix_sorted_transposed_from_path(path)


def load_ensemble_matrix_sorted_transposed_from_path(path: str) -> np.ndarray:
    ensemble_matrix = load_numpy_compressed(path)
    for row in ensemble_matrix:
        row.sort()
    ensemble_matrix = ensemble_matrix.transpose()
    districts, number_plans = np.shape(ensemble_matrix)
    print(districts, "districts")
    print(number_plans, "plans")
    return ensemble_matrix


def load_numpy_compressed(path: str) -> np.ndarray:
    return np.load(path)['arr_0']


def build_plan_name(chamber: str, plan: int) -> str:
    return f"{encode_chamber_character(chamber)}{plan}"


def load_plan_vectors(chamber: str, input_directory: str, statistic_name: str, plans: Iterable[int]) -> \
        dict[int, np.ndarray]:
    plan_vectors = {}
    for plan in plans:
        plan_vector_directory = f'{input_directory}plan_vectors/vectors_PLAN{build_plan_name(chamber, plan)}/'

        path = f'{plan_vector_directory}{statistic_name}_vector.csv'
        plan_vectors[plan] = load_numpy_csv(path)

    return plan_vectors


def build_seeds_directory(directory: str) -> str:
    return f'{directory}seeds/'


def build_ensemble_directory(directory: str, ensemble_description: str) -> str:
    return f'{directory}ensembles/ensemble_{ensemble_description}/'


def build_ensemble_description(chamber: str, seed_description: str, ensemble_number: int) -> str:
    return f'{chamber}_{seed_description}_{ensemble_number}'


def encode_chamber_character(chamber: str) -> str:
    return {
        'USCD': 'C',
        'TXSN': 'S',
        'TXHD': 'H'
    }[chamber]


def save_vector_csv(path: str, vector: list) -> None:
    with open(path, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows([vector])


def save_numpy_csv(path: str, array: np.ndarray) -> None:
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(array)


def load_numpy_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')


def load_merged_numpy_csv(filename: str, directories: list[str]) -> np.ndarray:
    data = [cm.load_numpy_csv(x + filename) for x in directories]
    return np.concatenate(data)


def get_number_districts(chamber: str) -> int:
    return {
        'USCD': 38,
        'TXSN': 31,
        'TXHD': 150
    }[chamber]


def get_allowed_number_districts(chamber: str) -> list[int]:
    return {
        'USCD': [36, 38],
        'TXSN': [31],
        'TXHD': [150]
    }[chamber]


def build_canonical_plan_set(plan: np.ndarray) -> frozenset[frozenset[int]]:
    district_plans_lookup = defaultdict(set[int])
    for i, district in enumerate(plan):
        district_plans_lookup[district].add(i)
    return frozenset(frozenset(x) for x in district_plans_lookup.values())


def calculate_plan_hash(plan: np.ndarray) -> int:
    return hash(build_canonical_plan_set(plan))


def build_plan_set_list(plan: np.ndarray) -> np.ndarray:
    district_plans_lookup = defaultdict(set[int])
    for i, district in enumerate(plan):
        district_plans_lookup[district].add(i)
    return np.array(frozenset(district_plans_lookup[x]) for x in sorted(district_plans_lookup))


def diff_plan_set_list(plan1: list[frozenset[int]], plan2: list[frozenset[int]], is_multi_step=False):
    data_type = np.uint16

    changed_indices = [i for i, (x, y) in enumerate(zip(plan1, plan2)) if not x == y]
    if len(changed_indices) == 0:
        return {
            'changed_indices': np.empty([0], data_type),
            'added_nodes': np.empty([0], data_type),
            'removed_nodes': np.empty([0], data_type)
        }
    elif is_multi_step or len(changed_indices) == 2:
        first_changed_index = changed_indices[0]
        return {
            'changed_indices': np.array(changed_indices, data_type),
            'added_nodes': np.array(list(plan2[first_changed_index].difference(plan1[first_changed_index])), data_type),
            'removed_nodes': np.array(list(plan1[first_changed_index].difference(plan2[first_changed_index])),
                                      data_type)
        }
    else:
        error_message = f"Each plan must differ by zero or two partitions.   Differences: {changed_indices}"
        raise RuntimeError(error_message)


def determine_unique_plans(plans: np.ndarray) -> np.ndarray:
    unique_plans = []
    plan_hashes = set()
    for i, plan in enumerate(plans):
        if i % 1000 == 0:
            print(f'{i} Number Unique: {len(unique_plans)}')

        plan_hash = calculate_plan_hash(plan)
        if plan_hash not in plan_hashes:
            plan_hashes.add(plan_hash)
            unique_plans.append(plan)

    return np.array(unique_plans)


def adjoin(elements: list, f: Callable) -> list:
    return [(x, f(x)) for x in elements]


def count_groups(elements: list) -> dict:
    counts = defaultdict(int)
    for x in elements:
        counts[x] += 1
    return counts


def groupby_project(iterable, key: Callable, projection: Callable) -> list:
    sorted_elements = iterable.copy()
    sorted_elements.sort(key=key)
    return [(x, [projection(z) for z in y]) for x, y in groupby(sorted_elements, key)]


def to_dict(elements: list) -> dict:
    return {x: y for x, y in elements}


def join_dict(d1: dict, d2: dict) -> dict:
    return {x: (y, d2[x]) for x, y in d1.items()}


def save_pickle(path: str, obj: Any) -> None:
    outfile = open(path, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load_pickle(path: str) -> Any:
    infile = open(path, 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj


def get_current_ensemble(chamber: str) -> tuple[str, int]:
    if chamber == 'USCD':
        seed_description = 'random_seed'
        ensemble_number = 2
    elif chamber == 'TXSN':
        seed_description = 'random_seed'
        ensemble_number = 2
    elif chamber == 'TXHD':
        seed_description = '2176_product'
        ensemble_number = 1
    else:
        raise RuntimeError("Unknown chamber")

    return seed_description, ensemble_number


def determine_original_plan(chamber: str) -> int:
    return {
        'USCD': None,
        'TXHD': 2100,
        'TXSN': 2100
    }[chamber]


def determine_population_limit(chamber: str) -> float:
    return {
        'USCD': .01,
        'TXSN': .02,
        'TXHD': .05
    }[chamber]


def build_proposed_plan_simulation_settings(chamber: str, plan: int) -> Dict:
    settings = Dict()
    settings.networkX_graph_filename = f'graph_TX_2020_cntyvtd_{chamber}_{plan}.gpickle'
    settings.redistricting_data_filename = f'nodes_TX_2020_cntyvtd_{chamber}_{plan}.parquet'
    settings.country_district_graph_filename = f'adj_TX_2020_cntyvtd_{chamber}_{plan}.gpickle'
    settings.epsilon = determine_population_limit(chamber)
    return settings


def build_TXSN_random_seed_simulation_settings() -> Dict:
    settings = Dict()
    settings.networkX_graph_filename = 'graph_TX_2020_cntyvtd_TXSN_seed_1000000.gpickle'
    settings.epsilon = determine_population_limit('TXSN')
    return settings


def ensure_directory_exists(output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)


def union(x: list, y: list) -> list:
    return list(set(x) | set(y))
