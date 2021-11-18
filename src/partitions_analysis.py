from datetime import datetime
from itertools import chain, product
import math
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.bipartite.matching as ma
import numpy as np
import pandas as pd
import pickle
import random
from scipy import stats
import scipy.sparse
from scipy.stats import norm, beta, skewnorm
from typing import Callable, Iterable, Any

import common as cm
import data_transform as dt
import proposed_plans as pp
import simulation as si
from timer import Timer


def calculate_intersection_nodes(plan1: np.ndarray, plan2: np.ndarray) -> dict[tuple[int, int], list[int]]:
    intersections = {}
    for i, x in enumerate(zip(plan1, plan2)):
        if x not in intersections:
            intersections[x] = [i]
        else:
            intersections[x].append(i)
    return intersections


def calculate_intersection_sizes_as_dictionary(plan1: np.ndarray, plan2: np.ndarray, node_weights: np.ndarray) \
        -> dict[tuple[int, int], int]:
    intersections = {}
    for i, x in enumerate(zip(plan1, plan2)):
        if x not in intersections:
            intersections[x] = node_weights[i]
        else:
            intersections[x] += node_weights[i]
    return intersections


def calculate_intersection_sizes(plan1: np.ndarray, plan2: np.ndarray, node_weights: np.ndarray, value_type: str) \
        -> scipy.sparse.csr_matrix:
    intersections_sizes = calculate_intersection_sizes_as_dictionary(plan1, plan2, node_weights)

    number_values = len(intersections_sizes)
    row = np.ndarray(number_values, dtype='uint8')
    col = np.ndarray(number_values, dtype='uint8')
    values = np.ndarray(number_values, dtype=value_type)
    for i, ((x, y), n) in enumerate(intersections_sizes.items()):
        row[i] = x - 1
        col[i] = y - 1
        values[i] = n
    number_parts = max(plan1)
    return scipy.sparse.csr_matrix((values, (row, col)), shape=(number_parts, number_parts))


def build_intersection_sizes_dictionary_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}intersection_sizes_dictionary{cm.build_suffix(suffix)}.npz'


def build_intersection_sizes_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}intersection_sizes{cm.build_suffix(suffix)}.npz'


def build_matchings_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}matchings{cm.build_suffix(suffix)}.npz'


def build_matched_intersection_sizes_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}matched_intersection_sizes{cm.build_suffix(suffix)}.npz'


def build_symmetric_difference_sizes_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}symmetric_difference_sizes{cm.build_suffix(suffix)}.npz'


def build_matched_plan_sizes_path(analysis_directory: str, suffix: str = '') -> str:
    return f'{analysis_directory}matched_plan_sizes{cm.build_suffix(suffix)}.npz'


def save_intersection_sizes(analysis_directory: str, initial_plan: np.ndarray, plans: np.ndarray,
                            node_weights: np.ndarray, value_type: str, suffix: str = '') -> None:
    number_plans = len(plans)
    intersection_sizes = np.ndarray(number_plans, dtype='object')
    for x in range(0, number_plans):
        if x % 1000 == 0:
            print(x)
        intersection_sizes[x] = calculate_intersection_sizes(initial_plan, plans[x], node_weights, value_type)
    np.savez_compressed(build_intersection_sizes_path(analysis_directory, suffix), intersection_sizes)


def load_intersection_sizes(analysis_directory: str, suffix: str = '') -> np.ndarray:
    path = build_intersection_sizes_path(analysis_directory, suffix)
    return np.load(path, allow_pickle=True)['arr_0']


def calculate_matching_from_dictionary(intersection_sizes: dict[tuple[int, int], int]) -> np.ndarray:
    def build_left_node(x: int) -> str:
        return f'l{x}'

    def build_right_node(x: int) -> str:
        return f'r{x}'

    def parse_edge(e: tuple[str, str]) -> tuple[int, int]:
        x, y = e
        if x[0] == "l":
            return int(x[1:]), int(y[1:])
        else:
            return int(y[1:]), int(x[1:])

    left_nodes = {build_left_node(x) for x, _ in intersection_sizes}
    right_nodes = {build_right_node(y) for _, y in intersection_sizes}

    distance_graph = nx.Graph()
    distance_graph.add_nodes_from(left_nodes, bipartite=0)
    distance_graph.add_nodes_from(right_nodes, bipartite=1)
    distance_graph.add_weighted_edges_from((build_left_node(x), build_right_node(y), -int(z))
                                           for (x, y), z in intersection_sizes.items())

    zero_edges = [(x, y) for x in left_nodes for y in right_nodes if (x, y) not in distance_graph.edges()]
    distance_graph.add_weighted_edges_from((x, y, 0) for x, y in zero_edges)

    matching = ma.minimum_weight_full_matching(distance_graph, left_nodes, weight='weight')
    matching_array = np.ndarray(len(left_nodes), dtype='uint8')

    for edge in matching.items():
        x, y = parse_edge(edge)
        matching_array[y - 1] = x
    return matching_array


def calculate_matching(intersection_sizes: np.ndarray) -> np.ndarray:
    intersection_sizes_dict = transform_sparse_matrix_to_dict(intersection_sizes)
    return calculate_matching_from_dictionary(intersection_sizes_dict)


def transform_sparse_matrix_to_dict(matrix: np.ndarray) -> dict[tuple[int, int], int]:
    transformed = {}
    rows, columns = np.shape(matrix)
    for x in range(0, rows):
        for y in range(0, columns):
            value = matrix[x, y]
            if value != 0:
                transformed[(x + 1, y + 1)] = value
    return transformed


def save_matchings(analysis_directory: str, intersection_sizes: np.ndarray, suffix: str = '') -> None:
    number_plans = len(intersection_sizes)
    number_parts, _ = np.shape(intersection_sizes[0])
    matchings = np.ndarray((number_plans, number_parts), dtype='uint8')
    for x in range(0, number_plans):
        if x % 1000 == 0:
            print(x)
        matchings[x] = calculate_matching(intersection_sizes[x].toarray())
    np.savez_compressed(build_matchings_path(analysis_directory, suffix), matchings)


def load_matchings(analysis_directory: str, suffix: str = '') -> np.ndarray:
    path = build_matchings_path(analysis_directory, suffix)
    return np.load(path)['arr_0']


def save_matched_intersection_sizes(analysis_directory: str, matchings: np.ndarray, intersection_sizes: np.ndarray,
                                    value_type: str, suffix: str = '') -> None:
    number_plans = len(matchings)
    number_parts = len(matchings[0])
    matched_intersections_sizes = np.ndarray(number_plans, dtype='object')
    for i, intersections in enumerate(intersection_sizes):
        if i % 1000 == 0:
            print(i)

        intersection_sizes_dict = transform_sparse_matrix_to_dict(intersections.todense())
        number_values = len(intersection_sizes_dict)
        matching = matchings[i]

        row = np.ndarray(number_values, dtype='uint8')
        col = np.ndarray(number_values, dtype='uint8')
        values = np.ndarray(number_values, dtype=value_type)
        for j, ((x, y), n) in enumerate(intersection_sizes_dict.items()):
            row[j] = x - 1
            col[j] = matching[y - 1] - 1
            values[j] = n
        matched_intersections_sizes[i] = scipy.sparse.csr_matrix((values, (row, col)),
                                                                 shape=(number_parts, number_parts))
    np.savez_compressed(build_matched_intersection_sizes_path(analysis_directory, suffix), matched_intersections_sizes)


def load_matched_intersection_sizes(analysis_directory: str, suffix: str = '') -> np.ndarray:
    path = build_matched_intersection_sizes_path(analysis_directory, suffix)
    return np.load(path, allow_pickle=True)['arr_0']


def calculate_number_intersections(intersection_sizes: np.ndarray) -> int:
    return np.trace(intersection_sizes)


def save_matched_plan_sizes(analysis_directory: str, matched_intersection_sizes: np.ndarray,
                            value_type: str, suffix: str = '') -> None:
    number_plans = len(matched_intersection_sizes)
    number_parts, _ = np.shape(matched_intersection_sizes[0].todense())
    matched_plan_sizes = np.ndarray((number_plans, number_parts), dtype=value_type)
    for i, sparse_matrix in enumerate(matched_intersection_sizes):
        if i % 1000 == 0:
            print(i)

        matrix = sparse_matrix.todense()
        matched_plan_sizes[i] = np.array([np.sum(x) for x in matrix], dtype=value_type)
    np.savez_compressed(build_matched_plan_sizes_path(analysis_directory, suffix), matched_plan_sizes)


def load_matched_plan_sizes(analysis_directory: str, suffix: str = '') -> np.ndarray:
    path = build_matched_plan_sizes_path(analysis_directory, suffix)
    return np.load(path)['arr_0']


def calculate_symmetric_difference_sizes(matched_intersections_sizes: np.ndarray) -> np.ndarray:
    number_plans = len(matched_intersections_sizes)
    total_weight = np.sum(matched_intersections_sizes[0].todense())
    symmetric_difference_sizes = np.ndarray(number_plans, dtype='int')
    for x in range(0, number_plans):
        if x % 100000 == 0:
            print(x)

        symmetric_difference_sizes[x] = total_weight - calculate_number_intersections(
            matched_intersections_sizes[x].todense())
    return symmetric_difference_sizes


def save_symmetric_difference_sizes(analysis_directory: str, symmetric_difference_sizes: np.ndarray,
                                    suffix: str = '') -> None:
    np.savez_compressed(build_symmetric_difference_sizes_path(analysis_directory, suffix), symmetric_difference_sizes)


def load_symmetric_difference_sizes(analysis_directory: str, suffix: str = '') -> np.ndarray:
    path = build_symmetric_difference_sizes_path(analysis_directory, suffix)
    return np.load(path)['arr_0']


def build_and_plot_random_distances(plans: np.ndarray) -> None:
    number_plans = len(plans)
    number_nodes = len(plans[0])

    random_intersection_sizes = np.ndarray(number_plans, dtype='object')
    for x in range(0, number_plans):
        plan1 = random.choice(plans)
        plan2 = random.choice(plans)
        if x % 1000 == 0:
            print(x)
        random_intersection_sizes[x] = calculate_intersection_sizes(plan1, plan2)
    random_distances = np.ndarray(number_plans, dtype='int')
    for x in range(0, number_plans):
        if x % 1000 == 0:
            print(x)
        random_distance[x] = number_nodes + calculate_distance_offsets(random_intersection_sizes[x])
    plt.plot(random_distances)

    data = random_distances
    a, loc, scale = stats.skewnorm.fit(data)
    print([a, loc, scale])
    number_bins = 200
    plt.figure(figsize=(16, 12))
    y, x, _ = plt.hist(data, bins=number_bins, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, number_bins)
    p = stats.skewnorm.pdf(x, a, loc, scale)
    plt.plot(x, p, 'k', linewidth=2)


def plot_distances_norm(total_weight: int, distances: np.ndarray) -> None:
    data = [np.arcsin(math.sqrt(x / total_weight)) for x in distances]
    plot_norm(data)


def plot_norm(data: Iterable[float]) -> None:
    a, b = norm.fit(data)
    print([a, b])
    number_bins = 200
    plt.figure(figsize=(16, 12))
    y, x, _ = plt.hist(data, bins=number_bins, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, number_bins)
    p = stats.norm.pdf(x, a, b)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()


def plot_distances_skewnormal(total_weight: int, distances: np.ndarray) -> None:
    data = [np.arcsin(math.sqrt(x / total_weight)) for x in distances]
    plot_skewnorm(data)


def plot_skewnorm(data: Iterable[float]) -> None:
    a, loc, scale = skewnorm.fit(data)
    print([a, loc, scale])
    number_bins = 200
    plt.figure(figsize=(16, 12))
    y, x, _ = plt.hist(data, bins=number_bins, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, number_bins)
    p = stats.skewnorm.pdf(x, a, loc, scale)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()


def plot_distances_beta(total_weight: int, distances: np.ndarray) -> None:
    data = [np.arcsin(math.sqrt(x / total_weight)) for x in distances]
    plot_beta(data)


def plot_beta(data: Iterable[float]) -> None:
    alpha, beta, xx, yy = stats.beta.fit(data)
    print([alpha, beta, xx, yy])
    number_bins = 200
    plt.figure(figsize=(16, 12))
    y, x, _ = plt.hist(data, bins=number_bins, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, number_bins)
    p = stats.beta.pdf(x, alpha, beta, xx, yy)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()


if __name__ == '__main__':
    def main() -> None:
        directory = 'G:/rob/projects/election/rob/'

        if True:
            ensemble_description = 'TXSN_random_seed_2'
            ensemble_directory = cm.build_ensemble_directory(directory, ensemble_description)
            analysis_directory = f'{ensemble_directory}analysis/'
            cm.ensure_directory_exists(analysis_directory)

            plans = cm.load_plans_from_path(f'{ensemble_directory}unique_plans.npz')

            reference_plan = plans[100000]

            seeds_directory = cm.build_seeds_directory(directory)
            settings = cm.build_TXSN_random_seed_simulation_settings()
            dual_graph = nx.read_gpickle(seeds_directory + settings.dual_graph_filename)
            total_populations = [(x, y['total_pop']) for x, y in dual_graph.nodes.items()]
            node_weights = np.array([y for _, y in sorted(total_populations, key=lambda x: x[0])])

            value_type = 'uint'
            suffix = 'total_pop'

            # save_intersection_sizes(analysis_directory, reference_plan, plans, node_weights, value_type, suffix)
            intersection_sizes = load_intersection_sizes(analysis_directory, suffix)

            # save_matchings(analysis_directory, intersection_sizes, suffix)
            # matchings = load_matchings(analysis_directory, suffix)

            # save_matched_intersection_sizes(analysis_directory, matchings, intersection_sizes, value_type, suffix)
            matched_intersections_sizes = load_matched_intersection_sizes(analysis_directory, suffix)

            # symmetric_difference_sizes = calculate_symmetric_difference_sizes(matched_intersections_sizes)
            # save_symmetric_difference_sizes(analysis_directory, symmetric_difference_sizes, suffix)

            # save_matched_plan_sizes(analysis_directory, matched_intersections_sizes, value_type, suffix)
            matched_plan_sizes = load_matched_plan_sizes(analysis_directory, suffix)

            return

            # plt.plot(distances)
            # plt.show()

            # plot_distances_skewnormal(number_nodes, distances)
            plot_distances_beta(number_nodes, symmetric_difference_sizes)

            # build_and_plot_random_distances(plans)


    main()
