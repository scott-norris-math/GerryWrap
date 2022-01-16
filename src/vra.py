import geopandas as gpd
from gerrychain import Election, updaters, Partition
import numpy as np
import operator
import os
import pandas as pd
from typing import Any

from run_functions import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
    cand_pref_all_draws_outcomes, precompute_state_weights, compute_district_weights


# modification of https://github.com/mggg/VRA_ensembles/TX/TX_elections_model.py

##################################################################
# key column names from Texas VTD shapefile
tot_pop = 'TOTPOP_x'
# white_pop = 'NH_WHITE'
CVAP = "1_2018"
WCVAP = "7_2018"
HCVAP = "13_2018"
BCVAP = "5_2018"  # with new CVAP codes!
geo_id = 'CNTYVTD'
# county_split_id = "CNTY_x"
# C_X = "C_X"
# C_Y = "C_Y"


def load_elec_model(num_districts: int, state_gdf: gpd.GeoDataFrame) -> tuple[Any, Any, Any]:
    print("Begin load_elect_model")

    input_directory = './VRA_ensembles/TX/'

    # user input parameters######################################
    record_statewide_modes = True
    record_district_mode = False

    # fixed parameters#################################################
    # num_districts = 36  # Number Congressional districts
    plot_path = f'{input_directory}TX_VTDs/TX_VTDs.shp'  # for shapefile

    # read files###################################################################
    elec_data = pd.read_csv(f"{input_directory}TX_elections.csv")
    TX_columns = list(pd.read_csv(f"{input_directory}TX_columns.csv")["Columns"])
    dropped_elecs = pd.read_csv(f"{input_directory}dropped_elecs.csv")["Dropped Elections"]
    recency_weights = pd.read_csv(f"{input_directory}recency_weights.csv")
    min_cand_weights = pd.read_csv(f"{input_directory}ingroup_weight.csv")
    cand_race_table = pd.read_csv(f"{input_directory}Candidate_Race_Party.csv")
    EI_statewide = pd.read_csv(f"{input_directory}statewide_rxc_EI_preferences.csv")
    prec_ei_df = pd.read_csv(f"{input_directory}prec_count_quants.csv", dtype={'CNTYVTD': 'str'})
    logit_params = pd.read_csv(f'{input_directory}TX_logit_params.csv')

    # initialize state_gdf########################################################
    # reformat/re-index enacted map plans
    if state_gdf is None:
        state_gdf = gpd.read_file(plot_path)
        state_gdf["CD"] = state_gdf["CD"].astype('int')
        state_gdf["Seed_Demo"] = state_gdf["Seed_Demo"].astype('int')
    state_gdf.columns = state_gdf.columns.str.replace("-", "_")

    # replace cut-off candidate names from shapefile with full names
    state_gdf_cols = list(state_gdf.columns)
    cand1_index = state_gdf_cols.index('RomneyR_12')
    cand2_index = state_gdf_cols.index('ObamaD_12P')
    state_gdf_cols[cand1_index:cand2_index + 1] = TX_columns
    state_gdf.columns = state_gdf_cols
    #state_df = pd.DataFrame(state_gdf)
    #state_df = state_df.drop(['geometry'], axis=1)

    # set up elections data structures ################################################
    elections = list(elec_data["Election"])
    elec_cand_list = TX_columns

    elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
    elec_data_trunc = elec_data[elecs_bool].reset_index(drop=True)
    elec_sets = list(set(elec_data_trunc["Election Set"]))
    elections = list(elec_data_trunc["Election"])
    general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
    primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
    runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)

    # this dictionary matches a specific election with the election set it belongs to
    elec_set_dict = {}
    for elec_set in elec_sets:
        elec_set_df = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
        elec_set_dict[elec_set] = dict(zip(elec_set_df.Type, elec_set_df.Election))

    elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

    # dictionary that maps an election to its candidates
    # only include 2 major party candidates in generals (assumes here major party candidates are first in candidate list)
    candidates = {}
    for elec in elections:
        # get rid of republican candidates in primaries or runoffs (primary runoffs)
        cands = [y for y in elec_cand_list if elec in y and "R_" not in y.split('1')[0]] if \
            "R_" in elec[:4] or "P_" in elec[:4] else [y for y in elec_cand_list if elec in y]

        elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]
        if elec in general_elecs:
            # assumes D and R are always first two candidates
            cands = cands[:2]
        candidates[elec] = dict(zip(list(range(len(cands))), cands))

    cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
    min_cand_weights_dict = {key: min_cand_weights.to_dict()[key][0] for key in min_cand_weights.to_dict().keys()}

    # ####################### pre-compute as much as possible for elections model ##############
    # pre-compute election recency weights "W1" df for all model modes
    elec_years = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str)
                  for elec_set in elec_sets]
    recency_scores = [recency_weights[elec_year][0] for elec_year in elec_years]
    recency_W1 = np.tile(recency_scores, (num_districts, 1)).transpose()

    # precompute statewide EI and recency (W1), in-group candidate(W2),
    # and candidate confidence (W3) for statewide/equal modes
    if record_statewide_modes:
        black_weight_state, hisp_weight_state, neither_weight_state, black_weight_equal, \
        hisp_weight_equal, neither_weight_equal, black_pref_cands_prim_state, hisp_pref_cands_prim_state, \
        black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state \
            = precompute_state_weights(num_districts, elec_sets, elec_set_dict, recency_W1, EI_statewide, primary_elecs,
                                       runoff_elecs, elec_match_dict, min_cand_weights_dict, cand_race_dict)

    # precompute set-up for district mode (need precinct-level EI data)
    if record_district_mode:
        print("Record District Mode")

        demogs = ['BCVAP', 'HCVAP']
        bases = {col.split('.')[0] + '.' + col.split('.')[1] for col in prec_ei_df.columns if
                 col[:5] in demogs and 'abstain' not in col and \
                 not any(x in col for x in general_elecs)}
        base_dict = {b: (b.split('.')[0].split('_')[0], '_'.join(b.split('.')[1].split('_')[1:-1])) for b in bases}
        outcomes = {val: [] for val in base_dict.values()}
        for b in bases:
            outcomes[base_dict[b]].append(b)

        prec_draws_outcomes_path = f'{input_directory}prec_draws_outcomes.npz'
        if os.path.exists(prec_draws_outcomes_path):
            prec_draws_outcomes_dict = np.load(prec_draws_outcomes_path)
            prec_draws_outcomes = {(x.split("-")[0], x.split("-")[1]): y for x, y in prec_draws_outcomes_dict.items()}
        else:
            precs = list(state_gdf[geo_id])
            prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

            prec_draws_outcomes_dict = {f'{x}-{y}': z for (x, y), z in prec_draws_outcomes.items()}
            np.savez_compressed(prec_draws_outcomes_path, **prec_draws_outcomes_dict)

    def final_elec_model(partition: Partition) -> tuple[dict, dict, dict]:
        """
        The output of the elections model is a probability distribution for each district:
        (Latino, Black, Neither or Overlap)-effective
        To compute these, each election set is first weighted (different for Black and Latino)
        according to three factors:
        a recency weight (W1), "in-group"-minority-preference weight (W2) and
        a preferred-candidate-confidence weight (W3).
        If the Black (Latino) preferred candidate wins the election (set) a number of points equal to
        the set's weight is accrued. The ratio of the accrued points to the total possible points
        is the raw Black (Latino)-effectiviness score for the district.

        Raw scores are adjusted by multiplying them by a "Group Control" factor,
        which measures the share of votes cast
        for a minority-preferred candidate by the minority group itself.

        Finally, the Black, Latino, Overlap, and Neither distribution (the values sum to 1)
        is computed, by feeding the adjusted effectiveness scores through a fitted logit function,
        and interpolating for the final four values. The output scores can be interpreted as the
        probability a district is effective for each group.

        We need to track several entities in the model, which will be dataframes or arrays,
        whose columns are districts and rows are election sets (or sometimes individual elections).
        These dataframes each store one of the following: Black (Latino) preferred candidates (in the
        election set's primary), Black (Latino) preferred candidates in runoffs, winners of primary,
        runoff and general elections, weights W1, W2 and W3
        and final election set weights for Black and Latino voters.
        """
        ###########################################################
        # We only need to run model on two ReCom districts that have changed in each step
        if partition.parent is not None:
            dict1 = dict(partition.parent.assignment)
            dict2 = dict(partition.assignment)
            differences = set([dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]]).union(
                set([dict2[k] for k in dict2.keys() if dict1[k] != dict2[k]]))

        dist_changes = range(num_districts) if partition.parent is None else sorted(differences)

        # dictionary to store district-level candidate vote shares
        dist_elec_results = {}
        order = [x for x in partition.parts]
        for elec in elections:
            cands = candidates[elec]
            dist_elec_results[elec] = {}
            outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]
            dist_elec_results[elec] = {d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in
                                       range(num_districts)}
        ##########################################################################################
        # compute winners of each election in each district and store:
        map_winners = pd.DataFrame(columns=dist_changes)
        map_winners["Election"] = elections
        map_winners["Election Set"] = elec_data_trunc["Election Set"]
        map_winners["Election Type"] = elec_data_trunc["Type"]
        for i in dist_changes:
            map_winners[i] = [max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0] for elec in elections]

        ######################################################################################
        # If we compute statewide scores: compute district effectiveness probabilities  #################
        if record_statewide_modes:
            # district probability distribution: statewide
            final_state_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state,
                                                       black_pref_cands_runoffs_state,
                                                       hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
                                                       neither_weight_state,
                                                       black_weight_state, hisp_weight_state, dist_elec_results,
                                                       dist_changes,
                                                       cand_race_table, num_districts, candidates, elec_sets, elec_set_dict,
                                                       "statewide", partition, logit_params, logit=True)

            # district probability distribution: equal
            final_equal_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state,
                                                       black_pref_cands_runoffs_state,
                                                       hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state,
                                                       neither_weight_equal,
                                                       black_weight_equal, hisp_weight_equal, dist_elec_results,
                                                       dist_changes,
                                                       cand_race_table, num_districts, candidates, elec_sets, elec_set_dict,
                                                       "equal", partition, logit_params, logit=True)

        # If we are computing district score: ######################################################
        # compute district weights, preferred candidates and district probability distribution: district
        if record_district_mode:
            black_weight_dist, hisp_weight_dist, neither_weight_dist, black_pref_cands_prim_dist, \
            black_pref_cands_runoffs_dist, hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist \
                = compute_district_weights(dist_changes, elec_sets, elec_set_dict, state_gdf, partition,
                                           prec_draws_outcomes,
                                           geo_id, primary_elecs, runoff_elecs, elec_match_dict, bases, outcomes,
                                           recency_W1, cand_race_dict, min_cand_weights_dict)

            final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist,
                                                      black_pref_cands_runoffs_dist,
                                                      hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist,
                                                      neither_weight_dist,
                                                      black_weight_dist, hisp_weight_dist, dist_elec_results, dist_changes,
                                                      cand_race_table, num_districts, candidates, elec_sets, elec_set_dict,
                                                      'district', partition, logit_params, logit=True)

        # New vector of probability distributions-by-district is the same as last ReCom step,
        # except in 2 changed districts
        if partition.parent is None:
            final_state_prob = {key: final_state_prob_dict[key] for key in sorted(final_state_prob_dict)} \
                if record_statewide_modes else {key: "N/A" for key in sorted(dist_changes)}

            final_equal_prob = {key: final_equal_prob_dict[key] for key in sorted(final_equal_prob_dict)} \
                if record_statewide_modes else {key: "N/A" for key in sorted(dist_changes)}

            final_dist_prob = {key: final_dist_prob_dict[key] for key in sorted(final_dist_prob_dict)} \
                if record_district_mode else {key: "N/A" for key in sorted(dist_changes)}

        else:
            final_state_prob = partition.parent["final_elec_model"][0].copy()
            final_equal_prob = partition.parent["final_elec_model"][1].copy()
            final_dist_prob = partition.parent["final_elec_model"][2].copy()

            for i in dist_changes:
                if record_statewide_modes:
                    final_state_prob[i] = final_state_prob_dict[i]
                    final_equal_prob[i] = final_equal_prob_dict[i]

                if record_district_mode:
                    final_dist_prob[i] = final_dist_prob_dict[i]

        return final_state_prob, final_equal_prob, final_dist_prob

    print("End load_elect_model")
    return candidates, elections, final_elec_model


def build_updaters(candidates, elections, final_elec_model):
    # ####construct updaters for Chain###############################################
    my_updaters = {
        "population": updaters.Tally(tot_pop, alias="population"),
        "CVAP": updaters.Tally(CVAP, alias="CVAP"),
        "WCVAP": updaters.Tally(WCVAP, alias="WCVAP"),
        "HCVAP": updaters.Tally(HCVAP, alias="HCVAP"),
        "BCVAP": updaters.Tally(BCVAP, alias="BCVAP"),
        # "Sum_CX": updaters.Tally(C_X, alias="Sum_CX"),
        # "Sum_CY": updaters.Tally(C_Y, alias="Sum_CY"),
        # "cut_edges": cut_edges,
        # "num_cut_edges": num_cut_edges,
        # "num_county_splits": num_county_splits,
        # "demo_percents": demo_percents,
        "final_elec_model": final_elec_model,
        # "centroids": centroids
    }

    # add elections updaters
    elections_track = [
        Election("PRES16", {"Democratic": 'ClintonD_16G_President', "Republican": 'TrumpR_16G_President'},
                 alias="PRES16"),
        Election("PRES12", {"Democratic": 'ObamaD_12G_President', "Republican": 'RomneyR_12G_President'},
                 alias="PRES12"),
        Election("SEN18", {"Democratic": "ORourkeD_18G_US_Sen", "Republican": 'CruzR_18G_US_Sen'}, alias="SEN18"),
        Election("GOV18", {"Democratic": "ValdezD_18G_Governor", "Republican": 'AbbottR_18G_Governor'}, alias="GOV18")
    ]

    election_updaters = {election.name: election for election in elections_track}
    my_updaters.update(election_updaters)

    election_functions = [Election(j, candidates[j]) for j in elections]
    election_updaters = {election.name: election for election in election_functions}
    my_updaters.update(election_updaters)

    return my_updaters
