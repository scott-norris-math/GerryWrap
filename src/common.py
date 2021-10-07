import csv
import zipfile
from collections import defaultdict
from itertools import groupby
from typing import Callable
import numpy as np
import gerrychain


CHAMBERS = ['TXSN', 'USCD', 'TXHD']


def load_plans(directory: str, ensemble_description: str, file_number: int) -> np.ndarray:
    plans = np.load(f'{directory}ensembles/{ensemble_description}/plans_{file_number}.npz')['arr_0']
    return plans + 1


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


def load_ensemble_matrix(input_directory: str, statistic_name: str) -> np.ndarray:
    path = input_directory + statistic_name + '.npz'
    ensemble_matrix = load_numpy_compressed(path)

    for row in ensemble_matrix:
        row.sort()

    ensemble_matrix = ensemble_matrix.transpose()
    districts, chainlength = np.shape(ensemble_matrix)
    print(districts, "districts")
    print(chainlength, "plans")
    return ensemble_matrix


def load_numpy_compressed(path: str) -> np.ndarray:
    return np.load(path)['arr_0']


def load_plan_vectors(chamber: str, input_directory: str, statistic_name: str, plans: list[int]) -> dict[
    int, np.ndarray]:
    plan_vectors = {}
    for plan in plans:
        plan_vector_directory = f'{input_directory}plan_vectors/vectors_PLAN{encode_chamber_character(chamber)}{plan}/'

        path = f'{plan_vector_directory}{statistic_name}_vector.csv'
        plan_vectors[plan] = load_numpy_csv(path)

    return plan_vectors


def build_ensemble_directory(directory: str, ensemble_description: str) -> str:
    return f'{directory}ensembles/ensemble_{ensemble_description}/'


def build_ensemble_description(chamber: str, seed_description: str, ensemble_number: int):
    suffix = f'{chamber}_{seed_description}_{ensemble_number}'
    return suffix


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


def build_canonical_plan(plan: np.ndarray) -> frozenset[frozenset[np.int16]]:
    district_plans_lookup = defaultdict(set[np.uint16])
    for i, district in enumerate(plan):
        district_plans_lookup[district].add(np.int16(i))
    canonical_plan = frozenset(frozenset(x) for x in district_plans_lookup.values())
    max_district = max(district_plans_lookup.keys())
    if len(canonical_plan) != max_district:
        raise RuntimeError("Error in Canonical Plan formation")
    return canonical_plan


def calculate_plan_hash(plan: np.ndarray) -> int:
    return hash(build_canonical_plan(plan))


def adjoin(l: list, f: Callable) -> list:
    return [(x, f(x)) for x in l]


def count_groups(l: list) -> dict:
    counts = defaultdict(int)
    for x in l:
        counts[x] += 1
    return counts


def groupby_project(iterable, key, projection):
    sorted = iterable.copy()
    sorted.sort(key=key)
    return [(x, [projection(z) for z in y]) for x, y in groupby(sorted, key)]


def to_dict(l):
    return {x: y for x, y in l}


def join_dict(d1, d2):
    return {x: (y, d2[x]) for x, y in d1.items()}


def load_merged_numpy_csv(filename: str, directories: list[str]) -> np.ndarray:
    data = [cm.load_numpy_csv(x + filename) for x in directories]
    return np.concatenate(data)


