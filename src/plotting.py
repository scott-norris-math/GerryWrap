from collections import defaultdict
import geopandas as gpd
from gerrychain import Partition, GeographicPartition, Graph
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import shapely as sh
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from typing import Iterable, Any, Optional

import common as cm
import plan_statistics as ps
import GerryWrap as gw
from timer import Timer
import proposed_plans as pp
import data_transform as dt
import simulation as si

BLACK = 'k'


def get_chamber_pretty_name(chamber: str) -> str:
    return {
        'USCD': "US Congress",
        'TXSN': "Texas Senate",
        'TXHD': "Texas House"
    }[chamber]


def get_elections() -> list[str]:
    return ['PRES20', 'SEN20']


def get_election_pretty_name(election: str) -> str:
    return {
        'SEN20': "2020 US Senate Election",
        'PRES20': "2020 US Presidential Election"
    }[election]


def get_racial_group_pretty_name(group: str) -> str:
    return {
        'HVAP': "Hispanic",
        'BVAP': "Black",
        'BHVAP': "Black+Hispanic",
        'WVAP': "White",
        'NWVAP': "Non-White"
    }[group]


def get_racial_groups() -> list[str]:
    return ['BVAP', 'HVAP', 'BHVAP', 'NWVAP']


def get_racial_group_fill_color(group: str) -> str:
    return {
        'HVAP': '#2222ff',
        'BHVAP': '#33CC00',
        'BVAP': '#ff9933',
        'WVAP': '#ff9933',
        'NWVAP': '#ff9933',
    }[group]


def build_plan_label(chamber, plan) -> str:
    return f"Proposed {cm.encode_chamber_character(chamber)}{plan}"


def save_election_plots(chamber: str, root_directory: str, ensemble_directory: str, plots_directory: str,
                        current_plan: Optional[int], comparison_plans: list[int], plan_pnums: list[bool],
                        plan_legend_names: list[str], plan_colors: list[str]) -> None:
    for election in get_elections():
        print(election)

        statistic_name = dt.build_election_filename_prefix(election)
        ensemble_matrix = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory, statistic_name)
        ensemble_matrix_transposed = ensemble_matrix.transpose()
        plans = ([] if current_plan is None else [current_plan]) + comparison_plans
        plan_vectors = cm.load_plan_vectors(chamber, root_directory, statistic_name, plans)

        partisan_metrics_figure_point = None
        for comparison_plan in comparison_plans:
            print(comparison_plan)

            if True:
                current_plan_override = None if chamber == 'USCD' else current_plan

                save_seats_voteshares_ensemble_comps_plot(chamber, plots_directory, election, ensemble_matrix,
                                                          current_plan_override, comparison_plan, plan_vectors,
                                                          plan_pnums, plan_legend_names, plan_colors)
                save_vote_vector_ensemble_plot(chamber, plots_directory, election, ensemble_matrix, comparison_plan,
                                               plan_vectors)
                save_seats_votes_ensemble_plot(chamber, plots_directory, election, ensemble_matrix, comparison_plan,
                                               plan_vectors)
                save_mean_median_partisan_bias_plot(chamber, plots_directory, election, ensemble_matrix,
                                                    comparison_plan, plan_vectors)

            if True:
                comparison_label = build_plan_label(chamber, comparison_plan)
                partisan_metrics_figure, partisan_metrics_point = \
                    gw.partisan_metrics_hist2D(ensemble_matrix_transposed,
                                               plan_vectors[comparison_plan], comparison_label,
                                               None, partisan_metrics_figure_point)
                partisan_metrics_figure_point = partisan_metrics_figure, partisan_metrics_point
                partisan_metrics_figure.savefig(
                    f'{plots_directory}partisan-metrics-2D-{chamber}-{comparison_plan}-{election}.pdf')

            clear_plots()


def hist_ensemble_comps(chamber: str, ensemble_matrix_transposed: np.ndarray, perc_thresh: float, title: str,
                        hist_x_axis_label: str, fill_color: str, do_small_histogram_pics: bool = True,
                        comp_plans: bool = False, comp_plans_vv: list[np.ndarray] = [],
                        comp_plans_names: list[str] = [], comp_plans_colors: list[str] = []) -> Any:
    if do_small_histogram_pics:
        plt.rc('font', size=100)  # controls default text sizes
        plt.rc('axes', titlesize=100)  # fontsize of the axes title
        plt.rc('axes', labelsize=50)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
        plt.rc('legend', fontsize=20)
        plt.rc('figure', titlesize=80)  # fontsize of the figure title
    else:
        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plot, ax = plt.subplots(1, 1, figsize=(14, 10))

    hist_y_axis_label = 'Plans'

    values = np.count_nonzero(ensemble_matrix_transposed > perc_thresh, axis=1)

    if do_small_histogram_pics:
        # This sets approximate range of histogram
        histogram_half_range = 6
        mode_result = stats.mode(values)
        if len(mode_result.mode) != 1:
            raise RuntimeError('More than one mode present in histogram')
        mode = mode_result.mode[0]
        histogram_range = range(max(mode - histogram_half_range, 0), mode + histogram_half_range)
        counts, bins, patches = plt.hist(x=values, bins=histogram_range,
                                         color=fill_color, alpha=.5, rwidth=0.85, align='left')
        ax.set_xticks(bins)
    else:
        dist = cm.get_number_districts(chamber)
        plt.hist(x=values, bins=dist + 1, range=(0, dist), color=fill_color, alpha=.5, rwidth=0.85)

    if comp_plans:
        for i in np.arange(len(comp_plans_vv)):
            comparison_x_value = sum(comp_plans_vv[i] > perc_thresh)
            plt.axvline(x=comparison_x_value, color=comp_plans_colors[i], ls='--', lw=2.5, ymax=0.75,
                        label=comp_plans_names[i])

    plt.xlabel(hist_x_axis_label)
    plt.ylabel(hist_y_axis_label)
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc=2)

    return plot


def save_seats_voteshares_ensemble_comps_plot(chamber: str, output_directory: str, election: str,
                                              ensemble_matrix: np.ndarray, current_plan: Optional[int],
                                              comparison_plan: int, plan_vectors: dict[int, np.ndarray],
                                              plan_pnums: list[bool], plan_legend_names: list[str],
                                              plan_colors: list[str]) -> None:
    plot_title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_election_pretty_name(election) + ')'

    plot = save_violin_comparison_plots(chamber, ensemble_matrix, current_plan, comparison_plan, plan_pnums,
                                        plan_legend_names, plan_colors, plan_vectors, plot_title,
                                        None, h_line_label="Needed to Win", y_axis_label="Democratic Vote Share")

    plot.savefig(
        f'{output_directory}seats-voteshares-ensemble-comps-{chamber}-{current_plan}-{comparison_plan}-{election}.pdf')

    plt.rcParams.update(plt.rcParamsDefault)


def save_violin_comparison_plots(chamber: str, ensemble_matrix: np.ndarray, current_plan: Optional[int],
                                 comparison_plan: int, plan_pnums: list[bool], plan_legend_names: list[str],
                                 plan_colors: list[str], plan_vectors: dict[int, np.ndarray], plot_title: str,
                                 fill_color: Any, h_line_label: str, y_axis_label: str) -> Any:
    plt.rc('xtick', labelsize=8)
    if current_plan is None:
        comparison_plans = [plan_vectors[comparison_plan]]
        plan_legend_names = plan_legend_names[1:]
        plan_colors = plan_colors[1:]
        plan_pnums = plan_pnums[1:]
    else:
        comparison_plans = [plan_vectors[current_plan], plan_vectors[comparison_plan]]

    if comparison_plan is not None:
        plan_legend_names = plan_legend_names.copy()
        plan_legend_names[-1] = plan_legend_names[-1] + f' {cm.encode_chamber_character(chamber)}{comparison_plan}'

    return gw.vote_vector_ensemble_comps(ensemble_matrix, plot_title, pc_thresh=.01, have_actual=False,
                                         comp_plans=True, comp_plans_vv=comparison_plans,
                                         comp_plans_names=plan_legend_names, comp_plans_colors=plan_colors,
                                         comp_plans_pnums=plan_pnums, fill_color=fill_color, h_line_label=h_line_label,
                                         y_axis_label=y_axis_label)


def save_vote_vector_ensemble_plot(chamber: str, output_directory: str, election: str, ensemble_matrix: np.ndarray,
                                   plan: int, plan_vectors: dict[int, np.ndarray]) -> None:
    plot_title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_election_pretty_name(
        election) + ')'
    plot = gw.vote_vector_ensemble(np.column_stack([plan_vectors[plan], ensemble_matrix]), plot_title, pc_thresh=.01,
                                   have_actual=True, comparison_label=build_plan_label(chamber, plan))
    plot.savefig(f'{output_directory}seats-voteshares-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_seats_votes_ensemble_plot(chamber: str, output_directory: str, election: str, ensemble_matrix: np.ndarray,
                                   plan: int, plan_vectors: dict[int, np.ndarray]) -> None:
    plot_title = get_chamber_pretty_name(chamber) + " Seats-Votes Curve (" + get_election_pretty_name(election) + ")"
    plot = gw.seats_votes_ensemble(np.column_stack([plan_vectors[plan], ensemble_matrix]), plot_title,
                                   have_actual=True)
    plot.savefig(f'{output_directory}seats-votes-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_mean_median_partisan_bias_plot(chamber: str, output_directory: str, election: str, ensemble_matrix: np.ndarray,
                                        plan: int, plan_vectors: dict[int, np.ndarray]) -> None:
    # OPTIONAL: pass in statewide=statewide_dem_share
    plot = gw.mean_median_partisan_bias(np.column_stack([plan_vectors[plan], ensemble_matrix]), have_actual=True,
                                        comparison_label=build_plan_label(chamber, plan))
    plot.savefig(f'{output_directory}mean-median-partisan-bias-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_racial_plots(chamber: str, root_directory: str, ensemble_directory: str, plots_directory: str,
                      current_plan: Optional[int], comparison_plans: list[int], plan_pnums: list[bool],
                      plan_legend_names: list[str], plan_colors: list[str]) -> None:
    for racial_group in get_racial_groups():
        print(racial_group)

        racial_group_file_prefix = dt.transform_racial_group_file_prefix(racial_group)
        statistic_name = dt.build_race_filename_prefix(racial_group_file_prefix)
        ensemble_matrix = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory, statistic_name)
        ensemble_matrix_transposed = ensemble_matrix.transpose()
        plans = ([] if current_plan is None else [current_plan]) + comparison_plans
        plan_vectors = cm.load_plan_vectors(chamber, root_directory, statistic_name, plans)

        for comparison_plan in comparison_plans:
            print(comparison_plan)

            if True:
                save_racial_ensemble_comps_plot(chamber, plots_directory, racial_group, ensemble_matrix,
                                                current_plan, comparison_plan, plan_vectors, plan_pnums,
                                                plan_legend_names, plan_colors)

            if True:
                save_racial_histograms(chamber, plots_directory, racial_group, ensemble_matrix_transposed,
                                       current_plan, comparison_plan, plan_vectors, plan_legend_names, plan_colors,
                                       True)

            clear_plots()


def save_racial_ensemble_comps_plot(chamber: str, output_directory: str, group: str, ensemble_matrix: np.ndarray,
                                    current_plan: Optional[int], comparison_plan: int,
                                    plan_vectors: dict[int, np.ndarray], plan_pnums: list[bool],
                                    plan_legend_names: list[str], plan_colors: list[str]) -> None:
    title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_racial_group_pretty_name(group) + ')'
    group_fill_color = get_racial_group_fill_color(group)
    plot = save_violin_comparison_plots(chamber, ensemble_matrix, current_plan, comparison_plan, plan_pnums,
                                        plan_legend_names,
                                        plan_colors, plan_vectors, title, fill_color=group_fill_color,
                                        h_line_label="Needed for Majority", y_axis_label="Voting Age Population")

    plot.savefig(f'{output_directory}violin-plot-{chamber}-{current_plan}-{comparison_plan}-{group}.pdf')
    plt.rcParams.update(plt.rcParamsDefault)


def save_racial_histograms(chamber: str, output_directory: str, group: str, ensemble_matrix_transposed: np.ndarray,
                           current_plan: Optional[int], comparison_plan: int, plan_vectors: dict[int, np.ndarray],
                           plan_legend_names: list[str], plan_colors: list[str],
                           do_small_histogram_pics: bool) -> None:
    chamber_pretty_name = get_chamber_pretty_name(chamber)

    if do_small_histogram_pics:
        plot_title = chamber + ' ' + group
    else:
        plot_title = chamber_pretty_name + ' District Results  (' + get_racial_group_pretty_name(group) + ')'

    if group == 'BVAP':
        perc_thresh = [0.3, 0.4, 0.5]
    elif group == 'HVAP':
        perc_thresh = [0.5, 0.55, 0.6, 0.65]
    elif group == 'BHVAP':
        perc_thresh = [0.5, 0.6, 0.7]
    elif group == 'WVAP':
        perc_thresh = [0.5, 0.6, 0.7]
    elif group == 'NWVAP':
        perc_thresh = [0.5, 0.6, 0.7]
    else:
        raise RuntimeError(f'Unknown group: {group}')

    if current_plan is None:
        comparison_plans = [plan_vectors[comparison_plan]]
        plan_legend_names = plan_legend_names[1:]
        plan_colors = plan_colors[1:]
    else:
        comparison_plans = [plan_vectors[current_plan], plan_vectors[comparison_plan]]

    if comparison_plan is not None:
        plan_legend_names = plan_legend_names.copy()
        plan_legend_names[-1] = plan_legend_names[-1] + f' {cm.encode_chamber_character(chamber)}{comparison_plan}'

    group_fill_color = get_racial_group_fill_color(group)
    for x in perc_thresh:
        hist_axis_label = f'Districts > {x * 100:.0f}%'
        plot = hist_ensemble_comps(chamber, ensemble_matrix_transposed, x, plot_title, hist_axis_label,
                                   group_fill_color, do_small_histogram_pics=do_small_histogram_pics,
                                   comp_plans=True, comp_plans_vv=comparison_plans,
                                   comp_plans_names=plan_legend_names, comp_plans_colors=plan_colors)
        size_string = 'small' if do_small_histogram_pics else 'large'
        plot.savefig(
            f'{output_directory}hist-{size_string}-{chamber}-{current_plan}-{comparison_plan}-{group}-{str(x)}.pdf')

    plt.rcParams.update(plt.rcParamsDefault)


def save_plots(chamber: str, root_directory: str, seed_description: str, ensemble_number: int,
               current_plan: Optional[int], comparison_plans: list[int]) -> None:
    ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)
    ensemble_directory = cm.build_ensemble_directory(root_directory, ensemble_description)
    plots_directory = build_plots_directory(root_directory, ensemble_description)
    cm.ensure_directory_exists(plots_directory)

    # For which should we plot district #, if any?
    plan_pnums = [] if chamber == 'TXHD' else [False, True]

    # Now we give each plan a name
    plan_legend_names = ['Current', 'Proposed']

    # Now we choose colors for each plan
    # Use RGB triplets OR strings
    plan_colors = ['green', 'red', 'darkviolet', 'turquoise', 'gold']

    if False:
        save_election_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan,
                            comparison_plans, plan_pnums, plan_legend_names, plan_colors)

    if False:
        save_racial_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan, comparison_plans,
                          plan_pnums, plan_legend_names, plan_colors)

    if True:
        save_election_racial_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan,
                                   comparison_plans, True, True)


def save_election_racial_plots(chamber: str, root_directory: str, ensemble_directory: str, plots_directory: str,
                               current_plan: Optional[int], comparison_plans: list[int], use_global_medians: bool,
                               display_ensemble: bool) -> None:
    for election in get_elections():
        print(f"Election: {election}")
        statistic_name = dt.build_election_filename_prefix(election)
        ensemble_matrix_election = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory,
                                                                             statistic_name).transpose()

        plans = ([] if current_plan is None else [current_plan]) + comparison_plans
        plan_vectors_election = cm.load_plan_vectors(chamber, root_directory, statistic_name, plans)
        for racial_group in get_racial_groups():
            print(f"Racial Group: {racial_group}")
            racial_group_file_prefix = dt.transform_racial_group_file_prefix(racial_group)
            statistic_name = dt.build_race_filename_prefix(racial_group_file_prefix)
            ensemble_matrix_racial = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory,
                                                                               statistic_name).transpose()
            plans = ([] if current_plan is None else [current_plan]) + comparison_plans
            plan_vectors_racial = cm.load_plan_vectors(chamber, root_directory, statistic_name, plans)

            previous_graphics = None
            for comparison_plan in comparison_plans:
                print(f"Plan: {comparison_plan}")
                title = get_chamber_pretty_name(chamber) + '  (' + get_racial_group_pretty_name(
                    racial_group) + ' - ' + get_election_pretty_name(election) + ')'
                number_points = 10000000
                with Timer(name='racial_vs_political'):
                    figure, previous_axes_points_annotations = \
                        gw.racial_vs_political_deviations(ensemble_matrix_election, plan_vectors_election[comparison_plan],
                                                          ensemble_matrix_racial, plan_vectors_racial[comparison_plan],
                                                          title, use_global_medians, display_ensemble, number_points,
                                                          previous_graphics)
                previous_graphics = figure, previous_axes_points_annotations
                infix = '' if use_global_medians else '-rank-medians'
                infix += '-with_ensemble' if display_ensemble else ''
                figure.savefig(
                    f'{plots_directory}racial-political-deviations{infix}-{chamber}-{comparison_plan}-{election}-{racial_group}.pdf')
            clear_plots()


def build_plots_directory(directory: str, ensemble_description: str) -> str:
    return f'{directory}plots/plots_{ensemble_description}/'


def draw_rainbow_map(partition: GeographicPartition) -> None:
    plt.rcParams['figure.figsize'] = [20, 10]
    partition.plot(cmap="gist_rainbow")
    reset_rcParams()


def reset_rcParams() -> None:
    plt.rcParams.update(plt.rcParamsDefault)


def register_colormap() -> None:
    mpl.cm.unregister_cmap('CustomStacked')
    pastel1 = mpl.cm.get_cmap('Pastel1')
    set3 = mpl.cm.get_cmap('Set3')
    stacked = np.vstack([set3(range(0, 12)), pastel1(range(0, 9))])
    mpl.cm.register_cmap('CustomStacked', ListedColormap(stacked))


def build_district_geometries(graph: nx.Graph, assignment: dict[str, int]) -> dict[int, BaseGeometry]:
    node_geometries = [(assignment[geoid], node['geometry']) for (geoid, node) in graph.nodes.items()]

    district_geometry_components = defaultdict(list)
    for district, geometry in node_geometries:
        district_geometry_components[district].append(geometry)

    district_geometries = {}
    for district, geometry_components in district_geometry_components.items():
        district_geometry = sh.ops.unary_union(geometry_components)
        district_geometries[district] = district_geometry
    return district_geometries


def plot_boundaries(geometries: Iterable[BaseGeometry]) -> None:
    for geometry in geometries:
        parts = list(geometry) if isinstance(geometry, MultiPolygon) else [geometry]
        for part in parts:
            plt.plot(*part.exterior.xy, BLACK, linewidth=.2, solid_capstyle='round', zorder=1)


def build_points(district_geometries: dict[int, BaseGeometry],
                 alternative_geometries: Optional[dict[int, BaseGeometry]]) -> dict[int, BaseGeometry]:
    points = {}
    for district, geometry in district_geometries.items():
        point_coordinates = geometry.representative_point().coords[:]
        if alternative_geometries is not None and not any(point_coordinates):
            print(f"Warning: Using alternative geometry for location of district {district} label")
            point_coordinates = alternative_geometries[district].representative_point().coords[:]
        points[district] = point_coordinates[0]
    return points


def save_plan_map(chamber: str, partition: GeographicPartition, coloring_assignment: dict[str, int],
                  district_text_colors: Optional[dict[int, int]], output_path_prefix: str) -> None:
    set_figure_size(chamber)

    colored_partition = GeographicPartition(partition.graph, assignment=coloring_assignment, updaters=[])
    colored_partition.plot(cmap='CustomStacked', linewidth=.1, edgecolor='0.8',
                           vmax=20)  # number colors in colmormap - 1

    district_geometries = build_district_geometries(partition.graph, partition.assignment)
    plot_boundaries(district_geometries.values())
    plot_district_numbers(district_geometries, district_text_colors, None)

    plt.tight_layout()
    plt.savefig(f'{output_path_prefix}.pdf')
    plt.savefig(f'{output_path_prefix}.png', dpi=150 if chamber == 'TXHD' else 200)

    reset_rcParams()


def set_figure_size(chamber) -> None:
    if chamber == 'TXHD':
        plt.rcParams['figure.figsize'] = [40, 40]
    else:
        plt.rcParams['figure.figsize'] = [20, 20]


def calculate_equitable_coloring(partition: Partition) -> dict[str, int]:
    partition_graph = build_partition_graph(partition)
    max_degree = max([y for x, y in partition_graph.degree()])
    coloring = nx.equitable_color(partition_graph, max(12, max_degree + 1))

    if False:
        print("Coloring")
        for district, color in coloring.items():
            print(f"{district}: {color}")

    return {x: coloring[partition.assignment[x]] for x in partition.graph.nodes()}


def plot_district_numbers(district_geometries: dict[int, BaseGeometry], district_text_colors: Optional[dict[int, int]],
                          alternative_geometries: Optional[dict[int, BaseGeometry]]) -> None:
    points = build_points(district_geometries, alternative_geometries)
    colors = list(mpl.cm.get_cmap("CustomStacked").colors)
    for district, point in sorted(points.items(), key=lambda x: x[0]):
        if isinstance(district_text_colors, dict):
            text_color = BLACK if district > len(district_text_colors) else colors[district_text_colors[district]]
            plt.annotate(str(district), xy=point, horizontalalignment='center', backgroundcolor='white',
                         bbox=dict(pad=0, facecolor='white', edgecolor='none', alpha=.6),
                         color=text_color, verticalalignment='center',
                         path_effects=[pe.withStroke(linewidth=.5, foreground='black')])
        else:
            plt.annotate(str(district), xy=point, horizontalalignment='center', verticalalignment='center')


def build_partition_graph(partition) -> nx.Graph:
    cut_edges = partition['cut_edges']
    cut_edges_districts = {(partition.assignment[x], partition.assignment[y]) for x, y in cut_edges}

    partition_graph = nx.Graph()
    vertices = {y for x in cut_edges_districts for y in x}

    partition_graph.add_nodes_from(vertices)
    partition_graph.add_edges_from(cut_edges_districts)

    return partition_graph


def save_proposed_plan_map(chamber: str, directory: str, plan: int) -> None:
    partition = si.load_geographic_partition(chamber, directory, plan)

    coloring_assignment = calculate_equitable_coloring(partition)

    reports_directory, report_filename_prefix = cm.build_reports_directory_and_filename(chamber, directory, plan)
    save_plan_map(chamber, partition, coloring_assignment, None, f'{reports_directory}{report_filename_prefix}_map.pdf')


def save_proposed_plan_diff_map_reversed(chamber: str, directory: str, plan: int) -> None:
    census_chamber = dt.get_census_chamber_name(chamber)
    partition = si.load_geographic_partition(chamber, directory, plan, columns=['geometry', census_chamber])

    assignments_2010 = {geoid: int(node[census_chamber]) for geoid, node in partition.graph.nodes.items()}
    partition_2010 = GeographicPartition(partition.graph, assignments_2010)

    coloring_assignment = calculate_equitable_coloring(partition_2010)
    district_coloring_array, district_coloring_dict = build_district_coloring(coloring_assignment, partition_2010)

    reports_directory, report_filename_prefix = cm.build_reports_directory_and_filename(chamber, directory, plan)
    diff_map_filename_prefix = f'{report_filename_prefix}_diff_map_reversed'
    cm.save_vector_csv(f'{reports_directory}{diff_map_filename_prefix}_colors.csv', district_coloring_array)

    save_plan_map(chamber, partition, coloring_assignment, district_coloring_dict,
                  f'{reports_directory}{diff_map_filename_prefix}')


def save_ensemble_map(chamber: str, directory: str, graph: Graph, plans: np.ndarray, ensemble_description: str,
                      plan_relative_number: int, plan_absolute_number: int) -> None:
    plots_directory = build_plots_directory(directory, ensemble_description)
    cm.ensure_directory_exists(plots_directory)

    plan = plans[plan_relative_number]
    assignment = cm.build_assignment(graph, plan)
    partition = GeographicPartition(graph, assignment=assignment, updaters=[])

    coloring_assignment = calculate_equitable_coloring(partition)
    save_plan_map(chamber, partition, coloring_assignment, None,
                  f'{plots_directory}{ensemble_description}_map_{plan_absolute_number}.pdf')


def save_proposed_plan_diff_map(chamber: str, directory: str, plan: int) -> None:
    census_chamber = dt.get_census_chamber_name(chamber)
    partition = si.load_geographic_partition(chamber, directory, plan, columns=['geometry', census_chamber])

    assignments_2010 = {geoid: int(node[census_chamber]) for geoid, node in partition.graph.nodes.items()}
    partition_2010 = GeographicPartition(partition.graph, assignments_2010)

    coloring_assignment = calculate_equitable_coloring(partition)
    district_coloring_array, district_coloring_dict = build_district_coloring(coloring_assignment, partition)

    reports_directory, report_filename_prefix = cm.build_reports_directory_and_filename(chamber, directory, plan)
    diff_map_filename_prefix = f'{report_filename_prefix}_diff_map'
    cm.save_vector_csv(f'{reports_directory}{diff_map_filename_prefix}_colors.csv', district_coloring_array)

    save_two_partition_map(chamber, partition, coloring_assignment, district_coloring_dict, partition_2010,
                           f'{reports_directory}{diff_map_filename_prefix}')


def save_two_partition_map(chamber: str, partition: GeographicPartition, coloring_assignment: dict[str, int],
                           partition_text_colors: Optional[dict[int, int]], boundary_partition: GeographicPartition,
                           output_path_prefix: str) -> None:
    set_figure_size(chamber)

    colored_partition = GeographicPartition(partition.graph, assignment=coloring_assignment, updaters=[])
    colored_partition.plot(cmap='CustomStacked', linewidth=.1, edgecolor='0.6',
                           vmax=20)  # number colors in colmormap - 1

    district_geometries = build_district_geometries(partition.graph, partition.assignment)
    boundary_district_geometries = build_district_geometries(boundary_partition.graph, boundary_partition.assignment)
    intersected_geometries = {
        x: y.intersection(boundary_district_geometries[x]) if x in boundary_district_geometries else y
        for x, y in district_geometries.items()}

    plot_boundaries(boundary_district_geometries.values())
    plot_district_numbers(intersected_geometries, partition_text_colors, district_geometries)

    plt.tight_layout()
    plt.savefig(f'{output_path_prefix}.pdf')
    plt.savefig(f'{output_path_prefix}.png', dpi=150 if chamber == 'TXHD' else 200)

    reset_rcParams()


def build_district_coloring(coloring_assignment: dict[str, int], partition: Partition) -> \
        tuple[list[int], dict[int, int]]:
    district_colorings = {(partition.assignment[geoid], coloring_assignment[geoid])
                          for geoid, node in partition.graph.nodes.items()}
    district_coloring_groups = cm.to_dict(cm.groupby_project(list(district_colorings), lambda x: x[0], lambda x: x[1]))
    if any([x for x, y in district_coloring_groups.items() if len(y) > 1]):
        raise RuntimeError("Districts with multiple colors")
    district_coloring_dict = {x: district_coloring_groups[x][0] for x in range(1, len(district_coloring_groups) + 1)}
    district_coloring_array = [district_coloring_dict[x] for x in range(1, len(district_coloring_groups) + 1)]
    return district_coloring_array, district_coloring_dict


def clear_plots() -> None:
    plt.figure().clear()
    plt.close()


if __name__ == '__main__':
    def main() -> None:
        directory = 'G:/rob/projects/election/rob/'

        if True:
            for chamber in cm.CHAMBERS:  # ['USCD']:  #
                print(f"Saving plots for {chamber}")
                current_plan = cm.determine_original_plan(chamber)

                if False:
                    comparison_plans = [pp.build_final_plan(chamber)]
                else:
                    comparison_plans = sorted(pp.get_valid_plans(chamber, pp.build_plans_directory(directory)) - {2100},
                        reverse=True)

                seed_description, ensemble_number = cm.get_current_ensemble(chamber)
                save_plots(chamber, directory, seed_description, ensemble_number, current_plan, comparison_plans)

        if False:
            register_colormap()

            chamber = 'TXHD'  # 'TXSN'  # 'USCD'  #
            settings = cm.build_proposed_plan_simulation_settings(chamber, 2176)
            geodata = si.load_geodataframe(directory, settings.redistricting_data_filename)
            # settings = cm.build_TXSN_random_seed_simulation_settings()
            # geodata = si.load_geodataframe(directory, None)
            graph = si.load_graph_with_geometry(directory, settings.dual_graph_filename, geodata)

            seed_description, ensemble_number = cm.get_current_ensemble(chamber)
            ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)

            print(f"Creating maps for {ensemble_description}")
            plans = np.concatenate([cm.load_plans(directory, ensemble_description, x) for x in range(0, 1)])

            # plans = cm.determine_unique_plans(plans)

            print(f"Number Plans: {len(plans)}")
            for x in range(0, 10):
                plan_number = x * 100000
                print(plan_number)

                save_ensemble_map(chamber, directory, graph, plans, ensemble_description, plan_number, plan_number)
                # plt.show()

        if False:
            register_colormap()

            for chamber in cm.CHAMBERS:  # ['USCD', 'TXHD']:  #
                print(chamber)

                if False:
                    plan = pp.build_final_plan(chamber)
                    save_proposed_plan_diff_map(chamber, directory, plan)
                    save_proposed_plan_diff_map_reversed(chamber, directory, plan)
                    break

                for plan in sorted(pp.get_valid_plans(chamber, pp.build_plans_directory(directory)) - {2100}, reverse=True):
                    print(plan)
                    save_proposed_plan_diff_map(chamber, directory, plan)
                    save_proposed_plan_diff_map_reversed(chamber, directory, plan)


    main()

# TODO transpose load_ensemble_matrix_sorted_transposed
# TODO implement colormap from bokeh for use in pyplot
