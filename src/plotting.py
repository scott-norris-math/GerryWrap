import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import networkx as nx
from gerrychain import GeographicPartition, Graph
from collections import defaultdict
import shapely as sh
from shapely import wkt
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import common as cm
import utilities as ut
import plan_statistics as ps
import GerryWrap as gw
from timer import Timer
import proposed_plans as pp
import data_transform as dt
import simulation as si


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
        'BHVAP': "Black+Hispanic"
    }[group]


def get_racial_groups() -> list[str]:
    return ['BVAP', 'HVAP', 'BHVAP']


def get_racial_group_fill_color(group: str) -> str:
    return {
        'HVAP': '#2222ff',
        'BHVAP': '#33CC00',
        'BVAP': '#ff9933'
    }[group]


def build_plan_label(chamber, plan) -> str:
    return f"Proposed {cm.encode_chamber_character(chamber)}{plan}"


def save_election_plots(chamber: str, root_directory: str, ensemble_directory: str, plots_directory: str,
                        current_plan: int, comparison_plans, plan_pnums, plan_legend_names, plan_colors):
    for election in get_elections():
        print(election)

        statistic_name = dt.build_election_filename_prefix(election)
        ensemble_matrix = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory, statistic_name)
        ensemble_matrix_transposed = ensemble_matrix.transpose()
        plan_vectors = cm.load_plan_vectors(chamber, root_directory, statistic_name,
                                            [current_plan] + comparison_plans)

        for comparison_plan in comparison_plans:
            print(comparison_plan)

            if True:
                current_plan_override = None if chamber == 'USCD' else current_plan

                with Timer(name='save_seats_voteshares_ensemble_comps_plot'):
                    save_seats_voteshares_ensemble_comps_plot(chamber, plots_directory, election, ensemble_matrix,
                                                              current_plan_override, comparison_plan, plan_vectors,
                                                              plan_pnums, plan_legend_names, plan_colors)
                with Timer(name='save_vote_vector_ensemble_plot'):
                    save_vote_vector_ensemble_plot(chamber, plots_directory, election, ensemble_matrix, comparison_plan,
                                                   plan_vectors)
                with Timer(name='save_seats_votes_ensemble_plot'):
                    save_seats_votes_ensemble_plot(chamber, plots_directory, election, ensemble_matrix, comparison_plan,
                                                   plan_vectors)
                with Timer(name='save_mean_median_partisan_bias_plot'):
                    save_mean_median_partisan_bias_plot(chamber, plots_directory, election, ensemble_matrix,
                                                        comparison_plan, plan_vectors)

            if True:
                comparison_label = build_plan_label(chamber, comparison_plan)
                with Timer(name='partisan_metrics_hist2D'):
                    figure = gw.partisan_metrics_hist2D(ensemble_matrix_transposed[-100000:, :],
                                                        plan_vectors[comparison_plan], comparison_label)
                    figure.savefig(f'{plots_directory}partisan-metrics-2D-{chamber}-{comparison_plan}-{election}.pdf')


def hist_ensemble_comps(chamber: str, ensemble_matrix_transposed, perc_thresh, title, hist_x_axis_label, fill_color,
                        do_small_histogram_pics=True, comp_plans=False, comp_plans_vv=[], comp_plans_names=[],
                        comp_plans_colors=[]):
    if do_small_histogram_pics:
        plt.rc('font', size=100)  # controls default text sizes
        plt.rc('axes', titlesize=100)  # fontsize of the axes title
        plt.rc('axes', labelsize=50)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)
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


def save_seats_voteshares_ensemble_comps_plot(chamber: str, output_directory: str, election: str, ensemble_matrix,
                                              current_plan, comparison_plan, plan_vectors, plan_pnums,
                                              plan_legend_names, plan_colors):
    plot_title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_election_pretty_name(election) + ')'

    plot = save_violin_comparison_plots(chamber, ensemble_matrix, current_plan, comparison_plan, plan_pnums,
                                        plan_legend_names, plan_colors, plan_vectors, plot_title,
                                        None, h_line_label="Needed to Win", y_axis_label="Democratic Vote Share")

    plot.savefig(
        f'{output_directory}seats-voteshares-ensemble-comps-{chamber}-{current_plan}-{comparison_plan}-{election}.pdf')

    plt.rcParams.update(plt.rcParamsDefault)


def save_violin_comparison_plots(chamber: str, ensemble_matrix, current_plan, comparison_plan, plan_pnums,
                                 plan_legend_names, plan_colors, plan_vectors, plot_title, fill_color, h_line_label,
                                 y_axis_label):
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
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
                                         comp_plans_names=plan_legend_names,
                                         comp_plans_colors=plan_colors, comp_plans_pnums=plan_pnums,
                                         fill_color=fill_color, h_line_label=h_line_label, y_axis_label=y_axis_label)


def save_vote_vector_ensemble_plot(chamber: str, output_directory: str, election: str, ensemble_matrix, plan,
                                   plan_vectors):
    plot_title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_election_pretty_name(
        election) + ')'
    plot = gw.vote_vector_ensemble(np.column_stack([plan_vectors[plan], ensemble_matrix]), plot_title, pc_thresh=.01,
                                   have_actual=True, comparison_label=build_plan_label(chamber, plan))
    plot.savefig(f'{output_directory}seats-voteshares-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_seats_votes_ensemble_plot(chamber: str, output_directory: str, election: str, ensemble_matrix, plan,
                                   plan_vectors):
    plot_title = get_chamber_pretty_name(chamber) + " Seats-Votes Curve (" + get_election_pretty_name(election) + ")"
    plot = gw.seats_votes_ensemble(np.column_stack([plan_vectors[plan], ensemble_matrix]), plot_title,
                                   have_actual=True)
    plot.savefig(f'{output_directory}seats-votes-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_mean_median_partisan_bias_plot(chamber: str, output_directory: str, election: str, ensemble_matrix, plan,
                                        plan_vectors):
    # OPTIONAL: pass in statewide=statewide_dem_share
    plot = gw.mean_median_partisan_bias(np.column_stack([plan_vectors[plan], ensemble_matrix]), have_actual=True,
                                        comparison_label=build_plan_label(chamber, plan))
    plot.savefig(f'{output_directory}mean-median-partisan-bias-ensemble-enacted-{chamber}-{plan}-{election}.pdf')


def save_racial_plots(chamber: str, root_directory: str, ensemble_directory: str, plots_directory: str, current_plan,
                      comparison_plans, plan_pnums, plan_legend_names, plan_colors):
    for racial_group in get_racial_groups():
        print(racial_group)

        # Load data. For each group of interest...
        racial_group_file_prefix = dt.transform_racial_group_file_prefix(racial_group)
        statistic_name = dt.build_race_filename_prefix(racial_group_file_prefix)
        ensemble_matrix = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory, statistic_name)
        ensemble_matrix_transposed = ensemble_matrix.transpose()
        plan_vectors = cm.load_plan_vectors(chamber, root_directory, statistic_name,
                                            [current_plan] + comparison_plans)

        for comparison_plan in comparison_plans:
            print(comparison_plan)
            current_plan_override = None if chamber == 'USCD' else current_plan

            if True:
                save_racial_ensemble_comps_plot(chamber, plots_directory, racial_group, ensemble_matrix,
                                                current_plan_override, comparison_plan, plan_vectors,
                                                plan_pnums, plan_legend_names, plan_colors)

            if True:
                save_racial_histograms(chamber, plots_directory, racial_group, ensemble_matrix_transposed,
                                       current_plan_override, comparison_plan, plan_vectors, plan_legend_names,
                                       plan_colors, True)


def save_racial_ensemble_comps_plot(chamber: str, output_directory: str, group: str, ensemble_matrix, current_plan,
                                    comparison_plan, plan_vectors, plan_pnums, plan_legend_names, plan_colors):
    plot_title = get_chamber_pretty_name(chamber) + ' District Results  (' + get_racial_group_pretty_name(group) + ')'
    group_fill_color = get_racial_group_fill_color(group)
    plot = save_violin_comparison_plots(chamber, ensemble_matrix, current_plan, comparison_plan, plan_pnums,
                                        plan_legend_names,
                                        plan_colors, plan_vectors, plot_title, fill_color=group_fill_color,
                                        h_line_label="Needed for Majority", y_axis_label="Voting Age Population")

    plot.savefig(
        f'{output_directory}violin-plot-{chamber}-{current_plan}-{comparison_plan}-{group}.pdf')
    plt.rcParams.update(plt.rcParamsDefault)


def save_racial_histograms(chamber: str, output_directory: str, group: str, ensemble_matrix_transposed: np.ndarray,
                           current_plan: int, comparison_plan: int, plan_vectors, plan_legend_names, plan_colors,
                           do_small_histogram_pics):
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
        # x is now perc_thresh
        hist_axis_label = f'Districts > {x * 100:.0f}%'
        plot = hist_ensemble_comps(chamber, ensemble_matrix_transposed, x, plot_title, hist_axis_label,
                                   group_fill_color,
                                   do_small_histogram_pics=do_small_histogram_pics,
                                   comp_plans=True,
                                   comp_plans_vv=comparison_plans,
                                   comp_plans_names=plan_legend_names,
                                   comp_plans_colors=plan_colors)
        # should have different titles if they are large
        size_string = 'small' if do_small_histogram_pics else 'large'
        plot.savefig(
            f'{output_directory}hist-{size_string}-{chamber}-{current_plan}-{comparison_plan}-{group}-{str(x)}.pdf')
        # Not needed?
        # plot.show()
    plt.rcParams.update(plt.rcParamsDefault)


def save_plots(chamber: str, root_directory: str, seed_description: str, ensemble_number, current_plan,
               comparison_plans):
    ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)
    ensemble_directory = cm.build_ensemble_directory(root_directory, ensemble_description)
    plots_directory = build_plots_directory(root_directory, ensemble_description)
    ut.ensure_directory_exists(plots_directory)

    # For which should we plot district #, if any?
    plan_pnums = [] if chamber == 'TXHD' else [False, True]
    # true_pnums = [i for i, x in enumerate(plan_pnums) if x]
    # print(true_pnums)

    # Now we give each plan a name
    plan_legend_names = ['Current', 'Proposed']
    # print(plan_legend_names)

    # Now we choose colors for each plan
    # Use RGB triplets OR strings
    plan_colors = ['green', 'red', 'darkviolet', 'turquoise', 'gold']
    # print(plan_colors[1])

    # elections
    if True:
        save_election_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan,
                            comparison_plans, plan_pnums, plan_legend_names, plan_colors)

    if True:
        save_racial_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan, comparison_plans,
                          plan_pnums, plan_legend_names, plan_colors)

    if True:
        save_election_racial_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan,
                                   comparison_plans)


def save_election_racial_plots(chamber, root_directory, ensemble_directory, plots_directory, current_plan,
                               comparison_plans):
    for election in get_elections():
        print(f"Election: {election}")
        statistic_name = dt.build_election_filename_prefix(election)
        ensemble_matrix_election = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory,
                                                                             statistic_name).transpose()
        plan_vectors_election = cm.load_plan_vectors(chamber, root_directory, statistic_name,
                                                     [current_plan] + comparison_plans)
        for racial_group in get_racial_groups():
            print(f"Racial Group: {racial_group}")
            racial_group_file_prefix = dt.transform_racial_group_file_prefix(racial_group)
            statistic_name = dt.build_race_filename_prefix(racial_group_file_prefix)
            ensemble_matrix_racial = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory,
                                                                               statistic_name).transpose()
            plan_vectors_racial = cm.load_plan_vectors(chamber, root_directory, statistic_name,
                                                       [current_plan] + comparison_plans)

            for comparison_plan in comparison_plans:
                print(f"Plan: {comparison_plan}")
                figure = gw.racial_vs_political_deviations(ensemble_matrix_election,
                                                           plan_vectors_election[comparison_plan],
                                                           ensemble_matrix_racial,
                                                           plan_vectors_racial[comparison_plan])
                figure.savefig(
                    f'{plots_directory}racial-political-deviations-{chamber}-{comparison_plan}-{election}-{racial_group}.pdf')


def build_plots_directory(directory: str, ensemble_description: str) -> str:
    return f'{directory}plots/plots_{ensemble_description}/'


def draw_rainbow_map(partition):
    plt.rcParams['figure.figsize'] = [20, 10]
    partition.plot(cmap="gist_rainbow")
    reset_rcParams()


def reset_rcParams():
    plt.rcParams.update(plt.rcParamsDefault)


def register_colormap():
    mpl.cm.unregister_cmap('CustomStacked')
    pastel1 = mpl.cm.get_cmap('Pastel1')
    set3 = mpl.cm.get_cmap('Set3')
    stacked = np.vstack([set3(np.linspace(0, 1, 128)), pastel1(np.linspace(0, 1, 128))])
    mpl.cm.register_cmap('CustomStacked', ListedColormap(stacked))


def build_points(graph: nx.Graph, assignment):
    node_geometries = [(assignment[geoid], node['geometry']) for (geoid, node) in graph.nodes.items()]

    district_geometry_components = defaultdict(list)
    for district, geometry in node_geometries:
        district_geometry_components[district].append(geometry)

    district_geometries = {}
    for district, geometry_components in district_geometry_components.items():
        district_geometries[district] = sh.ops.unary_union(geometry_components)

    points = {}
    for district, geometry in district_geometries.items():
        point_coordinates = geometry.representative_point().coords[:]
        points[district] = point_coordinates[0]

    return points


# noinspection PyCallingNonCallable
def save_plan_map(chamber: str, graph: Graph, plan: np.ndarray, output_path: str) -> None:
    assignment = cm.build_assignment(graph, plan)

    partition = GeographicPartition(graph, assignment=assignment, updaters=[])

    cut_edges = partition['cut_edges']
    cut_edges_districts = {(partition.assignment[x], partition.assignment[y]) for x, y in cut_edges}

    partition_graph = nx.Graph()
    vertices = {y for x in cut_edges_districts for y in x}

    partition_graph.add_nodes_from(vertices)
    partition_graph.add_edges_from(cut_edges_districts)

    max_degree = max([y for x, y in partition_graph.degree()])

    coloring = nx.equitable_color(partition_graph, max(12, max_degree + 1))
    coloring_assignment = {x: coloring[y] for x, y in zip(sorted(graph.nodes()), plan)}

    coloring_partition = GeographicPartition(graph, assignment=coloring_assignment, updaters=[])

    if chamber == 'TXHD':
        plt.rcParams['figure.figsize'] = [40, 40]
    else:
        plt.rcParams['figure.figsize'] = [20, 20]

    coloring_partition.plot(cmap='CustomStacked')

    points = build_points(graph, assignment)
    for district, point in points.items():
        plt.annotate(str(district), xy=point, horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path)

    reset_rcParams()


def save_ensemble_map(chamber: str, directory: str, graph: Graph, plans: np.ndarray, ensemble_description: str,
                      plan_relative_number: int, plan_absolute_number: int):
    plots_directory = build_plots_directory(directory, ensemble_description)
    ut.ensure_directory_exists(plots_directory)

    save_plan_map(chamber, graph, plans[plan_relative_number],
                  f'{plots_directory}{ensemble_description}_map_{plan_absolute_number}.png')


if __name__ == '__main__':
    def main():
        directory = 'C:/Users/rob/projects/election/rob/'

        chamber = 'TXHD'  # 'TXSN'  # 'USCD'  #
        seed_description, ensemble_number = cm.get_current_ensemble(chamber)

        if True:
            current_plan = 2100
            #comparison_plans = sorted(
            #    list(pp.get_valid_plans(chamber, pp.build_plans_directory(directory)) - {2100}), reverse=True)
            comparison_plans = [2315]

            save_plots(chamber, directory, seed_description, ensemble_number, current_plan, comparison_plans)

        if False:
            partition = load_partition(directory)
            draw_rainbow_map(partition)

        if False:
            register_colormap()

            settings = cm.build_proposed_plan_simulation_settings(chamber, 2176)
            geodata = si.load_geodataframe(directory, settings.redistricting_data_filename)
            graph = si.load_graph_with_geometry(directory, settings.networkX_graph_filename, geodata)

            seed_description, ensemble_number = cm.get_current_ensemble(chamber)
            ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)

            print(f"Creating maps for {ensemble_description}")
            plans = np.concatenate([cm.load_plans(directory, ensemble_description, x) for x in range(0, 1)])

            # plans = cm.determine_unique(plans)

            print(f"Number Unique Plans: {len(plans)}")
            for x in range(0, 10):
                plan_number = x * 100000
                print(plan_number)

                save_ensemble_map(chamber, directory, graph, plans, ensemble_description, plan_number, plan_number)
                # plt.show()


    main()


# TODO transpose load_ensemble_matrix_sorted_transposed
# TODO implement colormap from bokeh for use in pyplot
