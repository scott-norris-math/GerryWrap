from addict import Dict
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
from typing import Callable, Iterable, Optional

import common as cm
import data_transform as dt
import plotting as pl
import proposed_plans as pp


def calculate_ensemble_matrix_statistics(ensemble_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, number_ensemble_plans = np.shape(ensemble_matrix)
    mean_median = np.zeros(number_ensemble_plans)
    partisan_bias = np.zeros(number_ensemble_plans)

    for i in np.arange(number_ensemble_plans):
        if i % 100000 == 0:
            print(i)
        mean_median[i] = dt.calculate_mean_median(ensemble_matrix[:, i])
        partisan_bias[i] = dt.calculate_partisan_bias(ensemble_matrix[:, i])

    return mean_median, partisan_bias


def load_ensemble_statistics(chamber: str, root_directory: str, input_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    path = f'{root_directory}ensemble_{chamber}_{input_prefix}_statistics.npz'
    if os.path.exists(path):
        archive = np.load(path)
        return archive['mean_median'], archive['partisan_bias']

    seed_description, ensemble_number = cm.get_current_ensemble(chamber)
    ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)
    ensemble_directory = cm.build_ensemble_directory(root_directory, ensemble_description)

    ensemble = cm.load_ensemble_matrix_sorted(ensemble_directory, input_prefix).transpose()

    print("Calculating Ensemble Matrix Statistics")
    mean_median, partisan_bias = calculate_ensemble_matrix_statistics(ensemble)
    np.savez(path, mean_median=mean_median, partisan_bias=partisan_bias)
    return mean_median, partisan_bias


def determine_plans(chamber: str, directory: str) -> list[int]:
    if chamber == 'USCD':
        min_plan = -1
    elif chamber == 'TXSN':
        min_plan = -1
    elif chamber == 'TXHD':
        min_plan = -1
    else:
        raise RuntimeError("Unknown chamber")

    plans_metadata = pp.load_plans_metadata(chamber, pp.build_plans_directory(directory))
    plans = [x.plan for x in plans_metadata.itertuples() if x.plan > min_plan and not x.invalid]
    return plans


def save_statistics_statements(chamber: str, directory: str,
                               ensemble_statistics: dict[str, tuple[np.ndarray, np.ndarray]],
                               election: str, plans: Iterable[int]) -> None:
    plans = sorted(list(plans))
    ensemble_mean_median, ensemble_partisan_bias = ensemble_statistics[chamber]
    file_prefix = dt.build_election_filename_prefix(election, 'votes')
    plan_vectors = cm.load_plan_vectors(chamber, directory, file_prefix, plans)

    ensemble_statistics, plan_statistics_list = calculate_statistics(chamber, ensemble_mean_median,
                                                                     ensemble_partisan_bias, plans, plan_vectors)
    statements = determine_statements(ensemble_statistics, plan_statistics_list)
    cm.save_all_text("\n".join(statements), f'{directory}statistics_{chamber}_{election}.txt')


def calculate_statistics(chamber: str, mm_ensemble: np.ndarray, pb_ensemble: np.ndarray, plans: Iterable[int],
                         plan_vectors: dict[int, np.ndarray]) -> tuple[Dict, list[Dict]]:
    mm_ensemble_median = np.median(mm_ensemble)
    pb_ensemble_median = np.median(pb_ensemble)

    ensemble_statistics = Dict()
    ensemble_statistics.mm_ensemble_min = min(mm_ensemble)
    ensemble_statistics.mm_ensemble_median = mm_ensemble_median
    ensemble_statistics.mm_ensemble_max = max(mm_ensemble)
    ensemble_statistics.pb_ensemble_min = min(pb_ensemble)
    ensemble_statistics.pb_ensemble_median = pb_ensemble_median
    ensemble_statistics.pb_ensemble_max = max(pb_ensemble)

    ensemble_statistics.mm_ensemble_median_percentile = stats.percentileofscore(mm_ensemble, mm_ensemble_median,
                                                                                kind='mean')
    ensemble_statistics.pb_ensemble_median_percentile = stats.percentileofscore(pb_ensemble, pb_ensemble_median,
                                                                                kind='mean')

    number_ensemble_plans = len(mm_ensemble)
    ensemble_statistics.number_ensemble_plans = number_ensemble_plans
    plan_statistics_list = []
    for plan in plans:
        plan_statistics = Dict()
        plan_statistics.plan = plan
        plan_statistics.plan_name = f'{cm.encode_chamber(chamber)}{plan}'

        plan_vector = plan_vectors[plan]

        mm_plan = dt.calculate_mean_median(plan_vector)
        mm_portion = np.count_nonzero(mm_plan <= mm_ensemble) / number_ensemble_plans
        plan_statistics.mm_plan = mm_plan
        plan_statistics.mm_portion = mm_portion
        plan_statistics.mm_percentile = stats.percentileofscore(mm_ensemble, mm_plan, kind='mean')

        pb_plan = dt.calculate_partisan_bias(plan_vector)
        pb_less_than_portion = np.count_nonzero(pb_plan < pb_ensemble) / number_ensemble_plans
        pb_equals_portion = np.count_nonzero(pb_plan == pb_ensemble) / number_ensemble_plans
        plan_statistics.pb_plan = pb_plan
        plan_statistics.pb_less_than_portion = pb_less_than_portion
        plan_statistics.pb_equals_portion = pb_equals_portion
        plan_statistics.pb_percentile = stats.percentileofscore(pb_ensemble, pb_plan, kind='mean')

        if mm_plan < mm_ensemble_median:
            if pb_plan < pb_ensemble_median:
                plan_statistics.bias = 'R'
                ckPlg = 1
            else:
                plan_statistics.bias = 'N'
                ckPlg = 0
        else:
            if pb_plan > pb_ensemble_median:
                plan_statistics.bias = 'D'
                ckPlg = -1
            else:
                plan_statistics.bias = 'N'
                ckPlg = 0
        plan_statistics.ckPlg = ckPlg

        # Here, how we quantify how gerrymandered it is should depend on where it
        #   lives WRT two metrics
        # If overall plan favors R, we check in that direction
        # If overall plan favors D, we check in that direction
        if ckPlg == 1:
            plan_statistics.less_gerrymandered_than = np.count_nonzero(
                np.logical_and(mm_ensemble <= mm_plan, pb_ensemble <= pb_plan))
        elif ckPlg == -1:
            plan_statistics.less_gerrymandered_than = np.count_nonzero(
                np.logical_and(mm_ensemble >= mm_plan, pb_ensemble >= pb_plan))

        plan_statistics_list.append(plan_statistics)

    return ensemble_statistics, plan_statistics_list


def determine_statements(ensemble_statistics: Dict, plan_statistics_list: list[Dict]) -> list[str]:
    statements = build_ensemble_statistics_statements(ensemble_statistics)

    for plan_statistics in plan_statistics_list:
        statements.append(plan_statistics.plan_name)

        statements.append(plan_statistics.plan_name + ": MM = " + str(plan_statistics.mm_plan)
                          + " is <= %6.6f" % plan_statistics.mm_portion
                          + " and is > %6.6f" % (1 - plan_statistics.mm_portion)
                          + " Percentile: %6.6f" % plan_statistics.mm_percentile)

        statements.append(plan_statistics.plan_name + ": PB = " + str(plan_statistics.pb_plan)
                          + " is < %6.6f" % plan_statistics.pb_less_than_portion
                          + ", is == %6.6f" % plan_statistics.pb_equals_portion
                          + ", and is > %6.6f" % (
                                  1 - plan_statistics.pb_less_than_portion - plan_statistics.pb_equals_portion)
                          + " Percentile: %6.6f" % plan_statistics.pb_percentile)

        if plan_statistics.ckPlg == 1:
            statements.append(plan_statistics.plan_name + " favors Republicans")
        elif plan_statistics.ckPlg == 0:
            statements.append(plan_statistics.plan_name + " is ambiguous")
        elif plan_statistics.ckPlg == -1:
            statements.append(plan_statistics.plan_name + " favors Democrats")
        else:
            raise RuntimeError("Unhandled ckPlg")

        # Here, how we quantify how gerrymandered it is should depend on where it
        #   lives WRT two metrics
        # If overall plan favors R, we check in that direction
        # If overall plan favors D, we check in that direction
        if plan_statistics.ckPlg != 0:
            statements.append(plan_statistics.plan_name + " is LESS gerrymandered than "
                              + str(plan_statistics.less_gerrymandered_than) + " out of "
                              + str(ensemble_statistics.number_ensemble_plans) + " plans")

        statements.append("\n")

    return statements


def build_ensemble_statistics_statements(ensemble_statistics: Dict) -> list[str]:
    return [
        f"Ensemble Mean-Median - Min: {ensemble_statistics.mm_ensemble_min} "
        f"Median: {ensemble_statistics.mm_ensemble_median} "
        f"Percentile: {ensemble_statistics.mm_ensemble_median_percentile} Max: {ensemble_statistics.mm_ensemble_max}",
        f"Ensemble Partisan Bias - Min: {ensemble_statistics.pb_ensemble_min} "
        f"Median: {ensemble_statistics.pb_ensemble_median} "
        f"Percentile: {ensemble_statistics.pb_ensemble_median_percentile} Max: {ensemble_statistics.pb_ensemble_max}"]


def build_plan_name_cell_html(chamber: str, plan: int) -> str:
    media_anchor_map = {
        ('TXSN', 2101): '31A47221BAD242F283515D8F9A5581AD',
        ('TXSN', 2129): '3D5BF9F9329A45339B662200A7C44B9F',
        ('USCD', 2101): '03ACC422B3B4483BB99AF151159D72BC',
        ('USCD', 2102): '03ACC422B3B4483BB99AF151159D72BC',
        ('USCD', 2103): '03ACC422B3B4483BB99AF151159D72BC',
        ('USCD', 2104): '03ACC422B3B4483BB99AF151159D72BC',
        ('USCD', 2105): '03ACC422B3B4483BB99AF151159D72BC',
        ('USCD', 2135): 'B734C5E60A8C4202B83B007B15325E23'
    }

    report_url_prefix = 'https://storage.googleapis.com/mum_project/reports/report_'  # S2125.pdf'

    plan_name = cm.build_plan_name(chamber, plan)
    media_anchor_id = media_anchor_map.get((chamber, plan))
    return plan_name if plan == 2100 else f'<a href="{report_url_prefix}{plan_name}.pdf">{plan_name}</a>' \
        if media_anchor_id is None else f'<a href="-/media/{media_anchor_id}.ashx">{plan_name}</a>'


def build_submitter_string(parsed_submitter: tuple[Optional[str], str], submitter_party: Optional[str]):
    if not any(parsed_submitter):
        return ""

    s = f"{parsed_submitter[0]} " if parsed_submitter[0] is not None else ""
    s = f"{s}{parsed_submitter[1]}"

    if submitter_party is not None:
        s = f"{s} ({submitter_party})"

    return s


def build_row(chamber: str, number_ensemble_plans: int, plan_statistics: Dict,
              parsed_submitter: tuple[Optional[str], str], submitter_party: Optional[str],
              description: Optional[str]) -> str:
    def format_float(x: float) -> str:
        return format(x, '.2f').rstrip('0').rstrip('.')

    def build_number_plans_str(bias: str, less_gerrymandered_than: int, number_ensemble_plans: int) -> str:
        if bias == 'N':
            return "N/A"
        else:
            return f"{less_gerrymandered_than:,d} out of {number_ensemble_plans:,d}"

    row_text = """
<tr class="telerik-reTableFooterRow-2">
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{0}<br></td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{1}</td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{2}<br></td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{3}%<br></td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{4}<br></td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{5}%<br></td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{6}</td>
    <td class="telerik-reTableHeaderFirstCol-2" style="border:1px solid #4f81bd;   font-family: arial; font-size: 14pt;">{7}</td>
</tr>
"""

    number_plans_str = build_number_plans_str(plan_statistics.bias, plan_statistics.less_gerrymandered_than,
                                              number_ensemble_plans)

    return row_text.format(build_plan_name_cell_html(chamber, plan_statistics.plan),
                           description if description is not None else
                           build_submitter_string(parsed_submitter, submitter_party),
                           format_float(plan_statistics.mm_plan), format_float(plan_statistics.mm_percentile),
                           plan_statistics.pb_plan, format_float(plan_statistics.pb_percentile), plan_statistics.bias,
                           number_plans_str)


def save_statistics_rows(chamber: str, directory: str, ensemble_statistics: dict[str, tuple[np.ndarray, np.ndarray]],
                         election: str, plans_metadata: pd.DataFrame, use_description: bool) -> None:
    def build_description(plan_metadata) -> str:
        return "" if plan_metadata.description is np.nan else plan_metadata.description

    ensemble_mean_median, ensemble_partisan_bias = ensemble_statistics[chamber]
    plans = [x.plan for x in plans_metadata.itertuples()]
    file_prefix = dt.build_election_filename_prefix(election, 'votes')
    plan_vectors = cm.load_plan_vectors(chamber, directory, file_prefix, plans)

    ensemble_calculation, plan_statistics_list = calculate_statistics(chamber, ensemble_mean_median,
                                                                      ensemble_partisan_bias, plans, plan_vectors)
    ensemble_statements = build_ensemble_statistics_statements(ensemble_calculation)
    print("\n".join(ensemble_statements))

    party_lookup = pp.build_party_lookup(directory)
    rows = []
    for plan_statistics in plan_statistics_list:
        plan = plan_statistics.plan
        if plan == 2100:
            continue

        plan_metadata = plans_metadata.loc[plan]
        parsed_submitter = pp.parse_submitter(plan_metadata.submitter) if chamber in cm.CHAMBERS else (None, "")
        submitter_party = pp.determine_party(party_lookup, parsed_submitter) if chamber in cm.CHAMBERS else ""
        rows.append(build_row(chamber, ensemble_calculation.number_ensemble_plans, plan_statistics, parsed_submitter,
                              submitter_party, build_description(plan_metadata) if use_description else None))

    cm.save_all_text("\n".join(rows[0:6]), f'{directory}statistics_rows_{chamber}_{election}_short.txt')
    cm.save_all_text("\n".join(rows), f'{directory}statistics_rows_{chamber}_{election}.txt')


def save_plan_statistics(chambers: Iterable[str], directory: str) -> None:
    for election in pl.build_elections():
        file_prefix = dt.build_election_filename_prefix(election, 'votes')
        ensemble_statistics = {chamber: load_ensemble_statistics(chamber, directory, file_prefix) for
                               chamber in chambers}

        for chamber in chambers:
            plans_metadata = pp.load_plans_metadata(chamber, pp.build_plans_directory(directory))
            valid_plans_metadata = plans_metadata[plans_metadata['invalid'] == False].copy()
            valid_plans_metadata.set_index('plan', drop=False, inplace=True)

            print(f"Chamber: {chamber} {len([x for x in valid_plans_metadata.itertuples() if x.plan >= 2101])}")

            valid_plans_metadata.sort_index(ascending=False, inplace=True)

            valid_plans = [x.plan for x in valid_plans_metadata.itertuples() if not x.invalid]
            save_statistics_statements(chamber, directory, ensemble_statistics, election, valid_plans)
            save_statistics_rows(chamber, directory, ensemble_statistics, election, valid_plans_metadata,
                                 chamber == 'DCN')


if __name__ == '__main__':
    def main() -> None:
        directory = 'G:/rob/projects/election/rob/'

        if True:
            chambers = cm.CHAMBERS + ['DCN']
            save_plan_statistics(chambers, directory)


    main()
