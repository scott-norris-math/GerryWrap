from collections import defaultdict
import geopandas as gpd
from gerrychain import GeographicPartition, Graph
import maup as mp
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from shapely import wkt

import common as cm
import data_transform as dt
import GerryWrap as gw
import partitions_analysis as pa
import plan_statistics as ps
import plotting as pl
import proposed_plans as pp
import reporting
import simulation as si
import simulation_analysis as sa


def save_current_DCN_plan(directory: str) -> None:
    districts_gdf = load_DCN_districts_GeoDataFrame(directory)
    assignment_overrides = {'1130212003049': 3,
                            '1130181442017': 9}

    chamber = 'DCN'
    plan = 2100

    save_DCN_assignments(directory, districts_gdf, assignment_overrides,
                         f'{directory}plans_raw/PLAN{chamber}{plan}.csv', f'{directory}currentCityCouncils.pdf')


def load_DCN_districts_GeoDataFrame(directory: str) -> gpd.GeoDataFrame:
    dallas_council_geometry_path = f'{directory}Dallas/Dallas_2011CouncilDistricts_Adopted.csv'
    gdf = gpd.read_file(dallas_council_geometry_path)
    gdf['DIST_ID'] = gdf['DIST_ID'].astype(int)
    gdf.set_index('DIST_ID', inplace=True)
    gdf['geometry'] = gdf['the_geom'].apply(wkt.loads)
    return gdf


def save_DCN_assignments(directory: str, districts_gdf: gpd.GeoDataFrame, assignment_overrides: dict[str, int],
                         output_plan_path: str, output_plan_map_path: str) -> None:
    # replace with load plan with geodata?
    column = 'id-dallastx-dallastx_blocks-14-CouncilDistricts'
    plan = 77177  # has all 19039 census block geoids
    plan_path = f'{directory}plans/PlanDCN{plan}.csv'
    plan_df = pd.read_file(plan_path)
    plan_df['assignment'] = plan_df['assignment'].astype(int)
    plan_df['assignment'] = plan_df['assignment'] + 1
    plan_df[column] = plan_df[column].apply(lambda x: x[2:])
    plan_df.set_index(column, inplace=True)

    redistricting_data = sa.load_redistricting_data(directory)
    redistricting_data.set_index(['geoid'], inplace=True, drop=False)

    plan_df_with_geometry = redistricting_data.join(plan_df, how='inner')
    plan_gdf = gpd.GeoDataFrame(plan_df_with_geometry, geometry='polygon')
    plan_gdf['geometry'] = plan_gdf.geometry

    assignment_result = mp.assign(plan_gdf, districts_gdf)

    assignment = dict()
    missing_assignment_geoids = []
    for x, y in assignment_result.items():
        if math.isnan(y):
            if x in assignment_overrides:
                assignment[x] = assignment_overrides[x]
            else:
                missing_assignment_geoids.append(x)
        else:
            assignment[x] = int(y)

    if any(missing_assignment_geoids):
        raise RuntimeError("Unable to match all geoids")

    chamber = 'DCN'
    save_raw_plan(chamber, output_plan_path, assignment)

    plan_gdf['district'] = [assignment[x] for x in plan_gdf['geoid']]
    dual_graph = Graph.from_geodataframe(plan_gdf)
    geographic_partition = GeographicPartition(dual_graph, 'district')
    pl.plot_map(geographic_partition)
    plt.savefig(output_plan_map_path)


def save_raw_plan(chamber: str, output_path: str, assignment: dict[str, int]) -> None:
    if chamber != 'DCN':
        raise NotImplementedError("Saving raw plan is only implemented for DCN")

    columns = pp.build_plan_columns(chamber)
    geoids, districts = list(zip(*assignment.items()))
    current_plan = {columns[0]: geoids, columns[1]: districts}
    current_plan_df = pd.DataFrame.from_dict(current_plan)
    current_plan_df[columns[0]] = current_plan_df[columns[0]].apply(lambda x: '48' + x)
    current_plan_df[columns[1]] = current_plan_df[columns[1]].apply(lambda x: x - 1)
    current_plan_df.to_csv(output_path, index=False)


def save_current_proposed_plans_plot(chamber: str, directory: str, election: str) -> None:
    statistic_name = dt.build_election_filename_prefix(election, 'votes')

    current_plan = cm.determine_original_plan(chamber)
    comparison_plans = sorted(pp.get_valid_plans(chamber, pp.build_plans_directory(directory)) - {2100},
                              reverse=True)
    plans = ([] if current_plan is None else [current_plan]) + comparison_plans
    plan_vectors = cm.load_plan_vectors(chamber, directory, statistic_name, plans)

    mean_medians_ensemble, partisan_bias_ensemble = ps.load_ensemble_statistics(chamber, directory, statistic_name)
    chainlength = np.shape(mean_medians_ensemble)[0]
    districts = cm.get_number_districts(chamber)
    print(f"{chainlength} {districts}")
    hmid = 1 + (districts - 1) / 2.0

    plan_partisan_bias_groups = defaultdict(list)
    for plan_mean_median, plan_partisan_bias in zip(mean_medians_ensemble, partisan_bias_ensemble):
        plan_partisan_bias_groups[plan_partisan_bias].append(plan_mean_median / 100)

    ensemble_partisan_biases = sorted(plan_partisan_bias_groups)
    ensemble_partisan_bias_probs = defaultdict(float)
    for plan_partisan_bias in ensemble_partisan_biases:
        partisan_bias_group = plan_partisan_bias_groups[plan_partisan_bias]
        ensemble_partisan_bias_probs[plan_partisan_bias] = len(partisan_bias_group) / chainlength
        print(f'{plan_partisan_bias} {ensemble_partisan_bias_probs[plan_partisan_bias]} {np.median(partisan_bias_group)} {np.average(partisan_bias_group)}')

    plt.rcParams['figure.figsize'] = [15, 15]
    violin_values = [plan_partisan_bias_groups[x] for x in ensemble_partisan_biases]
    pc_thresh = .01

    y_last = defaultdict(float)
    plan_partisan_bias_groups = defaultdict(list)
    for x, y in plan_vectors.items():
        plan_mean_median, plan_partisan_bias = gw.calculate_mean_median_partisan_bias(districts, hmid, y)
        plan_partisan_bias_groups[plan_partisan_bias].append((x, plan_mean_median))

    for plan_partisan_bias in plan_partisan_bias_groups:
        plan_partisan_bias_groups[plan_partisan_bias] = list(sorted(plan_partisan_bias_groups[plan_partisan_bias], key=lambda x: x[1], reverse=True))

    plans_metadata = pp.load_plans_metadata(chamber, pp.build_plans_directory(directory))
    offset = -.005
    for plan_partisan_bias in sorted(plan_partisan_bias_groups):
        y_last[plan_partisan_bias] = 100
        for plan_number, plan_mean_median in plan_partisan_bias_groups[plan_partisan_bias]:
            plan_metadata = plans_metadata.loc[plan_number]
            plt.plot(plan_partisan_bias, plan_mean_median, 'rs')
            if plan_mean_median - y_last[plan_partisan_bias] > offset:
                y_last[plan_partisan_bias] = y_last[plan_partisan_bias] + offset
            else:
                y_last[plan_partisan_bias] = plan_mean_median
            x_coordinate = plan_partisan_bias + .15
            y_coordinate = y_last[plan_partisan_bias] - .001
            print(f"{plan_partisan_bias} {plan_mean_median} {plan_number} {x_coordinate} {y_coordinate}")
            plt.gca().annotate(f'{plan_number} {plan_metadata.description if plan_metadata.description is not np.nan else ""}',
                               xy=(plan_partisan_bias, plan_mean_median), xycoords='data',
                               xytext=(x_coordinate, y_coordinate), textcoords='data',
                               arrowprops=dict(arrowstyle='-', relpos=(0,0.5)))

    max_prob = max(ensemble_partisan_bias_probs.values())
    widths = [.8 * ensemble_partisan_bias_probs[x] / max_prob for x in ensemble_partisan_biases]
    plt.violinplot(violin_values, ensemble_partisan_biases, showextrema=False, widths=widths,
                   quantiles=[[pc_thresh, .5, 1 - pc_thresh] for _ in ensemble_partisan_biases])

    plt.xlabel("Partisan Bias")
    plt.ylabel("Mean Median")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, 0))

    plt.tight_layout()
    plt.savefig(f'{directory}dallasCityCouncilPlans.png')


if __name__ == '__main__':
    def main() -> None:
        directory = 'G:/rob/projects/election/rob/'
        chamber = 'DCN'
        election = 'SEN20'

        pl.register_colormap()

        if False:
            save_current_DCN_plan(directory)

        if True:
            save_current_proposed_plans_plot(chamber, directory, election)

        if False:
            plans_directory = pp.build_plans_directory(directory)
            plans_metadata = pp.load_plans_metadata(chamber, plans_directory)
            pp.save_current_merged_plans(chamber, directory, plans_metadata, force=False)

            pp.save_current_plan_vectors(chamber, directory)
            pp.save_plan_vectors_summary(chamber, directory)
            pp.save_plan_vectors_differences(chamber, directory, cm.determine_original_plan(chamber))

            min_plan = 101652

            sa.save_plan_seeds(chamber, directory, min_plan)

            pl.save_proposed_plan_diff_maps(chamber, directory, min_plan)

            pl.FAST_PLOTTING = True
            pl.save_plots(chamber, directory, min_plan)

            reporting.save_reports(chamber, directory, min_plan)

            save_current_proposed_plans_plot(chamber, directory, election)

            ps.save_plan_statistics([chamber], directory)

            # TODO: upload diff_map, report, plans plot


main()
