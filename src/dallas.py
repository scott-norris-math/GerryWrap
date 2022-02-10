import addict
from collections import defaultdict
from datetime import datetime
import geopandas as gpd
from gerrychain import GeographicPartition, Graph
import glob
from google.cloud import storage
from google import auth
import json
import maup as mp
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
import os
import pandas as pd
from scipy import stats
from shapely import wkt
import shutil
import sys
import time

import common as cm
import data_transform as dt
import GerryWrap as gw
import partitions_analysis as pa
import plan_statistics as ps
import plans_scraper as pls
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

    violin_values = [plan_partisan_bias_groups[x] for x in ensemble_partisan_biases]
    pc_thresh = .01

    y_last = defaultdict(float)
    plan_partisan_bias_groups = defaultdict(list)
    for x, y in plan_vectors.items():
        plan_mean_median, plan_partisan_bias = gw.calculate_mean_median_partisan_bias(districts, hmid, y)
        plan_partisan_bias_groups[plan_partisan_bias].append((x, plan_mean_median))

    for plan_partisan_bias in plan_partisan_bias_groups:
        plan_partisan_bias_groups[plan_partisan_bias] = list(sorted(plan_partisan_bias_groups[plan_partisan_bias], key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(15, 15))

    max_prob = max(ensemble_partisan_bias_probs.values())
    widths = [.8 * ensemble_partisan_bias_probs[x] / max_prob for x in ensemble_partisan_biases]
    plt.violinplot(violin_values, ensemble_partisan_biases, showextrema=False, widths=widths,
                   quantiles=[[pc_thresh, .5, 1 - pc_thresh] for _ in ensemble_partisan_biases])

    plans_metadata = pp.load_plans_metadata(chamber, pp.build_plans_directory(directory))
    offset = -.005
    for plan_partisan_bias in sorted(plan_partisan_bias_groups):
        y_last[plan_partisan_bias] = 100
        for plan_number, plan_mean_median in plan_partisan_bias_groups[plan_partisan_bias]:
            plan_metadata = plans_metadata.loc[plan_number]
            plt.plot(plan_partisan_bias, plan_mean_median, 'rs')

            if "betzen" in str(plan_metadata.description).lower() and plan_number > 103120:
                print(f"{plan_partisan_bias} {plan_mean_median} {plan_number}")
                continue

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

    plt.xlabel("Partisan Bias")
    plt.ylabel("Mean Median")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, 0))

    plt.tight_layout()
    plt.savefig(f'{directory}dallasCityCouncilPlans.png')


def backup_plans_metadata(chamber: str, plans_directory: str) -> None:
    source_path = pp.build_plans_metadata_path(chamber, plans_directory)
    target_path = pp.build_plans_metadata_path(chamber, f'{plans_directory}/backups/',
                                               f'_{datetime.now().strftime("%m_%d_%y_%H_%M_%S_%f")}')
    shutil.copyfile(source_path, target_path)


def retrieve_new_plans_metadata(known_plans: set[int]) -> list:
    plans_response = pls.retrieve_web_page(
        'https://districtr.org/.netlify/functions/eventRead?skip=0&limit=9&event=cityofdallas')
    plans = [x for x in addict.Dict(json.loads(plans_response)).plans if x.plan.problem.name == 'City Council']

    duplicate_plans = [x for x in plans if x['simple_id'] in known_plans]
    if not any(duplicate_plans):
        raise NotImplementedError("No already known plans")

    return [{
        'plan': x['simple_id'],
        'description': x['planName'].strip(),
        'submitter': '',
        'previous_plan': 0,
        'changed_rows': 0,
        'invalid': 0
    } for x in plans if not x['simple_id'] in known_plans]


def download_plan_raw(chamber: str, directory: str, plan: int) -> None:
    def parse_district(x) -> int:
        if type(x) is list:
            length = len(x)
            if length != 1:
                error_message = f'Error parsing district {x}'
                raise RuntimeError(error_message)
            return x[0]
        else:
            return x

    print(f"Downloading: {plan}")
    plan_response = pls.retrieve_web_page(f'https://districtr.org/.netlify/functions/planRead?id={plan}')
    plan_response_dictionary = addict.Dict(json.loads(plan_response))
    rows = ['id-dallastx-dallastx_blocks-14-CouncilDistricts,assignment'] + \
           [f'{x},{parse_district(y)}' for x, y in plan_response_dictionary.plan.assignment.items()]

    cm.save_all_text("\n".join(rows), pp.build_plan_raw_path(chamber, directory, plan))

    print("Download Sleep")
    time.sleep(30)


def upload_to_bucket(credentials_path: str, file_path: str, bucket_name: str, blob_path: str) -> None:
    storage_client = storage.Client.from_service_account_json(credentials_path)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    print(f"Uploading {file_path} to {bucket_name} {blob_path}")
    blob.upload_from_filename(file_path)


def process_new_plans(chamber: str, directory: str, election: str, min_plan: int):
    plans_directory = pp.build_plans_directory(directory)
    plans_metadata = pp.load_plans_metadata(chamber, plans_directory)
    pp.save_current_merged_plans(chamber, directory, plans_metadata, force=False)

    pp.save_current_plan_vectors(chamber, directory)
    pp.save_plan_vectors_summary(chamber, directory)
    pp.save_plan_vectors_differences(chamber, directory, cm.determine_original_plan(chamber))

    save_current_proposed_plans_plot(chamber, directory, election)

    ps.save_plan_statistics([chamber], directory)

    sa.save_plan_seeds(chamber, directory, min_plan)

    pl.save_proposed_plan_diff_maps(chamber, directory, min_plan)

    pl.FAST_PLOTTING = True
    pl.save_plots(chamber, directory, min_plan)

    reporting.save_reports(chamber, directory, min_plan)

    upload_to_google_storage(chamber, directory, min_plan)


def process_proposed_plans_pages(directory: str, election: str, known_plans: set[int]) -> set[int]:
    chamber = 'DCN'

    new_plans_metadata = retrieve_new_plans_metadata(known_plans)
    new_plans = list(sorted(x['plan'] for x in new_plans_metadata))

    if any(new_plans):
        print(f"New Plan Found: {new_plans}")

        plans_directory = pp.build_plans_directory(directory)
        plans_metadata_df = pls.update_plans_metadata(chamber, plans_directory, new_plans_metadata)
        known_plans = set(plans_metadata_df['plan'])

        for plan in new_plans:
            download_plan_raw(chamber, directory, plan)

        process_new_plans(chamber, directory, election, min(new_plans))

    return known_plans


def get_downloaded_plans(chamber: str, plans_raw_directory: str) -> set[int]:
    plans = set()
    for path in glob.glob(f'{plans_raw_directory}PLAN{cm.encode_chamber(chamber)}*.csv'):
        path = os.path.normpath(path)
        plan_string = str(path.replace(os.path.dirname(path), '').removesuffix('.csv')[1:]). \
            removeprefix(f'PLAN{cm.encode_chamber(chamber)}')
        plans.add(int(plan_string))
    return plans


def upload_to_google_storage(chamber: str, directory: str, min_plan: int) -> None:
    credentials_path = f'{directory}config/google_cloud_credentials.json'
    bucket_name = 'mum_project'

    if chamber == 'DCN':
        upload_to_bucket(credentials_path, f'{directory}dallasCityCouncilPlans.png', bucket_name,
                         'dallasCityCouncilPlans.png')

    plans_directory = pp.build_plans_directory(directory)
    plans_metadata = pp.load_plans_metadata(chamber, plans_directory)
    plans = [x for x in plans_metadata['plan'] if x >= min_plan]
    for plan in plans:
        diff_map_path = f'{pl.build_diff_map_path_prefix(chamber, directory, plan)}.pdf'
        upload_to_bucket(credentials_path, diff_map_path, bucket_name, diff_map_path.removeprefix(directory))

        report_directory, report_filename_prefix = cm.build_reports_directory_and_filename(chamber, directory,
                                                                                           plan)
        report_path = f'{report_directory}{report_filename_prefix}.pdf'
        upload_to_bucket(credentials_path, report_path, bucket_name, report_path.removeprefix(directory))


def run(directory: str, election: str) -> None:
    sys.excepthook = pls.handle_exception

    known_plans = get_downloaded_plans('DCN', pp.build_plans_raw_directory(directory))
    print(f"Initial Plans: {pls.format_plans(known_plans)}")

    while True:
        print(datetime.now().strftime("%H:%M:%S"))

        known_plans = process_proposed_plans_pages(directory, election, known_plans)

        print("Sleeping")
        time.sleep(10 * 60)


if __name__ == '__main__':
    def main() -> None:
        # Have to set the key log file to nothing or else we get a permission error
        pls.disable_ssl_logging()

        chamber = 'DCN'
        directory = 'G:/rob/projects/election/rob/'
        election = 'SEN20'

        pl.register_colormap()

        if False:
            save_current_DCN_plan(directory)

        if False:
            save_current_proposed_plans_plot(chamber, directory, election)

        if True:
            run(directory, election)

        if False:
            process_new_plans(chamber, directory, election, 103506)

        if False:
            upload_to_google_storage(chamber, directory, 203120)


    main()
