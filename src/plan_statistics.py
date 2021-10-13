import os
import numpy as np

import common as cm
import data_transform as dt
import proposed_plans as pp


def calculate_mean_median(plan_vector: np.ndarray) -> float:
    # mean-median score, denominated in terms of "percentage needed for
    #  majority" (D - R). Effectively this is: x -1 (to get D-R),
    #   x 2 (because MM measures diff. from midpoint), and
    #   x 100 (to make it a %)
    return -200 * (np.median(plan_vector) - np.mean(plan_vector))


def calculate_partisan_bias(plan_vector: np.ndarray) -> float:
    mean_voteshare = np.mean(plan_vector)
    # DOES NOT matter if sorted, because we will sum up how many above 50%
    seats_votes1 = mean_voteshare - np.array(plan_vector) + 0.5
    seats_votes2 = np.flip(1 - seats_votes1)

    number_seats1 = np.count_nonzero(seats_votes1 <= 0.5)
    number_seats2 = np.count_nonzero(seats_votes2 <= 0.5)

    return number_seats1 - number_seats2


def determine_statements(chamber: str, mm_ensemble: np.ndarray, pb_ensemble: np.ndarray, plans: list[int],
                         plan_vectors: dict[int: np.ndarray]) -> list[str]:
    mm_ensemble_median = np.median(mm_ensemble)
    pb_ensemble_median = np.median(pb_ensemble)

    statements = [
        f"Ensemble Mean-Median - Min: {min(mm_ensemble)} Median: {mm_ensemble_median} Max: {max(mm_ensemble)}",
        f"Ensemble Partisan Bias - Min: {min(pb_ensemble)} Median: {pb_ensemble_median} Max: {max(pb_ensemble)}", "\n"]

    number_ensemble_plans = len(mm_ensemble)
    for plan in plans:
        plan_name = f'{cm.encode_chamber_character(chamber)}{plan}'
        statements.append(plan_name)

        plan_vector = plan_vectors[plan]
        mm_plan = calculate_mean_median(plan_vector)
        pb_plan = calculate_partisan_bias(plan_vector)

        mm_portion = np.count_nonzero(mm_plan <= mm_ensemble) / number_ensemble_plans
        statements.append(plan_name + ": MM = " + str(mm_plan)
                          + " is <= %6.6f" % mm_portion
                          + " and is > %6.6f" % (1 - mm_portion))

        pb_less_than_portion = np.count_nonzero(pb_plan < pb_ensemble) / number_ensemble_plans
        pb_equals_portion = np.count_nonzero(pb_plan == pb_ensemble) / number_ensemble_plans
        statements.append(plan_name + ": PB = " + str(pb_plan)
                          + " is < %6.6f" % pb_less_than_portion
                          + ", is == %6.6f" % pb_equals_portion
                          + ", and is > %6.6f" % (1 - pb_less_than_portion - pb_equals_portion))

        if mm_plan > mm_ensemble_median:
            if pb_plan < pb_ensemble_median:
                statements.append(plan_name + " favors Republicans")
                ckPlg = 1
            else:
                statements.append(plan_name + " is ambiguous")
                ckPlg = 0
        else:
            if pb_plan > pb_ensemble_median:
                statements.append(plan_name + " favors Democrats")
                ckPlg = -1
            else:
                statements.append(plan_name + " is ambiguous")
                ckPlg = 0

        # Here, how we quantify how gerrymandered it is should depend on where it
        #   lives WRT two metrics
        # If overall plan favors R, we check in that direction
        # If overall plan favors D, we check in that direction
        if ckPlg == 1:
            less_gerrymandered_than = np.count_nonzero(np.logical_and(mm_ensemble >= mm_plan, pb_ensemble <= pb_plan))
        elif ckPlg == -1:
            less_gerrymandered_than = np.count_nonzero(np.logical_and(mm_ensemble <= mm_plan, pb_ensemble >= pb_plan))

        if ckPlg != 0:
            statements.append(plan_name + " is LESS gerrymandered than "
                              + str(less_gerrymandered_than) + " out of "
                              + str(number_ensemble_plans) + " plans")

        statements.append("\n")

    return statements


def calculate_ensemble_matrix_statistics(ensemble_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    _, number_ensemble_plans = np.shape(ensemble_matrix)
    mean_median = np.zeros(number_ensemble_plans)
    partisan_bias = np.zeros(number_ensemble_plans)

    for i in np.arange(number_ensemble_plans):
        mean_median[i] = calculate_mean_median(ensemble_matrix[:, i])
        partisan_bias[i] = calculate_partisan_bias(ensemble_matrix[:, i])

    return mean_median, partisan_bias


def load_ensemble_statistics(chamber: str, root_directory: str, input_prefix: str) -> (np.ndarray, np.ndarray):
    path = f'{root_directory}ensemble_{chamber}_{input_prefix}_statistics.npz'
    if os.path.exists(path):
        archive = np.load(path)
        return archive['mean_median'], archive['partisan_bias']

    seed_description, ensemble_number = cm.get_current_ensemble(chamber)
    ensemble_description = cm.build_ensemble_description(chamber, seed_description, ensemble_number)
    ensemble_directory = cm.build_ensemble_directory(root_directory, ensemble_description)

    ensemble = cm.load_ensemble_matrix_sorted_transposed(ensemble_directory, input_prefix)

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


def save_statistics(chamber: str, directory: str, ensemble_statistics: (np.ndarray, np.ndarray),
                    file_prefix: str, plans):
    plans = sorted(list(plans))
    ensemble_mean_median, ensemble_partisan_bias = ensemble_statistics[chamber]
    plan_vectors = cm.load_plan_vectors(chamber, directory, file_prefix, plans)
    statements = determine_statements(chamber, ensemble_mean_median, ensemble_partisan_bias, plans, plan_vectors)
    cm.save_all_text("\n".join(statements), f'{directory}statistics_{chamber}.txt')


if __name__ == '__main__':
    def main():
        chamber = "TXSN" # "USCD"  #

        root_directory = 'C:/Users/rob/projects/election/rob/'

        if False:
            election = "PRES20"  # "SEN20" #
            file_prefix = dt.build_election_filename_prefix(election)

            print("Loading ensemble statistics")
            mean_median, partisan_bias = load_ensemble_statistics(chamber, root_directory, file_prefix)

            plans = determine_plans(chamber, root_directory)

            plan_vectors = cm.load_plan_vectors(chamber, root_directory, file_prefix, plans)

            statements = determine_statements(chamber, mean_median, partisan_bias, plans, plan_vectors)
            for x in statements:
                print(x)

        if True:
            file_prefix = dt.build_election_filename_prefix('PRES20')
            admissible_chambers = cm.CHAMBERS  # ['TXHD']
            ensemble_statistics = {chamber: load_ensemble_statistics(chamber, root_directory, file_prefix) for
                                   chamber in admissible_chambers}

            for chamber in admissible_chambers:
                plans_metadata_df = pp.load_plans_metadata(chamber, pp.build_plans_directory(root_directory))
                valid_proposed_plans = pp.determine_valid_plans(plans_metadata_df)
                save_statistics(chamber, root_directory, ensemble_statistics, file_prefix, valid_proposed_plans)


    main()
