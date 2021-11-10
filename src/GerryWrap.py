import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from typing import Any


USED_STUFFED_IN_PLOTS = False


def get_comparison_market_size(districts: int) -> int:
    if districts <= 40:
        msize = 4
    elif districts <= 80:
        msize = 2
    else:
        msize = 1
    return msize


def vote_vector_ensemble_comps(ensemble: np.ndarray, title: str, pc_thresh: float = 0.01, have_actual: bool = True,
                               comp_plans=False, comp_plans_vv: list[np.ndarray] = [], comp_plans_names: list[str] = [],
                               comp_plans_colors: list[str] = [], comp_plans_pnums: list[bool] = [], fill_color=None,
                               h_line_label: str = '', y_axis_label: str = '') -> Any:
    # PURPOSE: allow us to include multiple plans for comparison
    #
    # INPUTS
    #     ensemble:  (nDistrict x nPlan) array of values. MUST BE SORTED WITHIN EACH COLUMN
    #     title: string for title
    #
    # OPTIONAL INPUTS [def/type/[default value] ]
    #     pc_thresh:  quantile markers for violin plots (e.g.: 0.01 means 1% and 99%)
    #     have_actual:   Does ensemble include data assoc. with an exacted plan? 
    #             If so, assume FIRST COLUMN is enacted/[True]
    #     comp_plans:  Do we include a list of plans for comparison?/Bool/[False]
    #     comp_plans_vv: vote vectors for plans to compare/
    #             list of K numpy arrays, where "K"=# of plans provided/[]
    #     comp_plans_names: names to use for legend/list of K strings/[]
    #     comp_plans_colors: colors for dots/list of K strings/[]
    #     comp_plans_pnums: plot district numbers?/ list of K Bool/[]
    #     

    # for making things look nice
    vbuffer = .04

    # get shape of data
    districts, chainlength = np.shape(ensemble)

    # list of integers (for plotting)
    seats = np.arange(districts) + 1

    # create Seats/votes curve for enacted plan
    if have_actual:
        vs_actual = np.array(sorted(ensemble[:, 0]))

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5 * chainlength), chainlength, 10)
    vs_ensemble = [ensemble[ii, samples_to_plot] for ii in range(districts)]
    vs_lower = [np.percentile(vse, 100 * pc_thresh) for vse in vs_ensemble]
    vs_median = [np.percentile(vse, 50) for vse in vs_ensemble]
    vs_upper = [np.percentile(vse, 100 * (1 - pc_thresh)) for vse in vs_ensemble]

    # if have_actual:
    #    print(np.sum([a - b for a, b in zip(vs_actual, vs_median)]))

    # identify stuffed, packed, and cracked districts
    # ONLY makes sense if actual plan present
    if have_actual:
        dmin = 0
        dmax = districts
        stuffed = [ii for ii in range(dmin, dmax) if (vs_upper[ii] < vs_actual[ii] < .5)]
        # if len(stuffed) > 0:  dmin = stuffed[-1]
        cracked = [ii for ii in range(dmin, dmax) if
                   (vs_lower[ii] > vs_actual[ii] > .3 and vs_actual[ii] < .5)]
        # if len(cracked) > 0:  dmin = cracked[-1]
        packed = [ii for ii in range(dmin, dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5)]
    else:
        stuffed = []
        cracked = []
        packed = []

    # -----------------------------------------------

    # Create the plot
    myplot = plt.figure(figsize=(6.5, 3.5))

    violin_parts = plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6,
                                  quantiles=[[pc_thresh, 1 - pc_thresh] for _ in seats])

    if fill_color is not None:
        for vp in violin_parts['bodies']:
            vp.set_facecolor(fill_color)
            vp.set_alpha(0.5)

    plt.plot(seats, vs_median, 'bo', markersize=2, label="Median of Ensemble")

    marker_size = get_comparison_market_size(districts)
    if have_actual:
        plt.plot(seats, vs_actual, 'ro', markersize=marker_size, label="Actual Vote Shares")
    plt.plot(seats, 0 * seats + 0.5, 'k--', lw=0.75, label=h_line_label)

    # ---------------------------------------------

    # labelling stuffed districts
    if len(stuffed) > 2:
        smin_loc = 1 + stuffed[0] - 0.5
        smax_loc = 1 + stuffed[-1] + 0.5
        smin_val = vs_actual[stuffed[0]] - vbuffer
        smax_val = vs_actual[stuffed[-1]] + vbuffer
        if USED_STUFFED_IN_PLOTS:
            plt.fill([smin_loc, smax_loc, smax_loc, smin_loc], [smin_val, smin_val, smax_val, smax_val], 'y', alpha=0.3)
            plt.text(0.5 * (smin_loc + smax_loc), smax_val + vbuffer, '"Stuffing"',
                     **{'ha': 'center', 'weight': 'bold'})

    # labelling cracked districts
    if len(cracked) > 2:
        cmin_loc = 1 + cracked[0] - 0.5
        cmax_loc = 1 + cracked[-1] + 0.5
        cmin_val = vs_actual[cracked[0]] - vbuffer
        cmax_val = vs_actual[cracked[-1]] + vbuffer
        plt.fill([cmin_loc, cmax_loc, cmax_loc, cmin_loc], [cmin_val, cmin_val, cmax_val, cmax_val], 'y', alpha=0.3)
        plt.text(0.5 * (cmin_loc + cmax_loc), cmin_val - 2 * vbuffer, '"Cracking"',
                 **{'ha': 'center', 'weight': 'bold'})

    # labelling packed districts
    if len(packed) > 2:
        pmin_loc = 1 + packed[0] - 0.5
        pmax_loc = 1 + packed[-1] + 0.5
        pmin_val = vs_actual[packed[0]] - vbuffer
        pmax_val = vs_actual[packed[-1]] + vbuffer
        plt.fill([pmin_loc, pmax_loc, pmax_loc, pmin_loc], [pmin_val, pmin_val, pmax_val, pmax_val], 'y', alpha=0.3)
        plt.text(0.5 * (pmin_loc + pmax_loc), pmax_val - 2 * vbuffer, '"Packing"', **{'ha': 'center', 'weight': 'bold'})

    # ---------------------------------------------------------    

    # "Comparison Plans", if applicable
    if comp_plans:
        for i in np.arange(len(comp_plans_vv)):
            # print(comp_plans_vv[i])
            # print(comp_plans_colors[i])
            # print(comp_plans_names[i])
            y1 = np.array(sorted(comp_plans_vv[i]))
            # print(y1)
            # plt.plot(seats, y1, 'ro', markersize=4, label=comp_plans_names[i])
            plt.plot(seats, y1, color=comp_plans_colors[i], marker='o', markersize=marker_size, ls='',
                     label=comp_plans_names[i])

    # Replace 1-n with actual district numbers
    if comp_plans_pnums:
        # Determine if district number should be plotted
        true_pnums = [i for i, x in enumerate(comp_plans_pnums) if x]
        whpnum = true_pnums[0]

        # Order of districts for x-axis
        dnumvec = np.argsort(comp_plans_vv[whpnum])

        # Use dnumvec computed above
        dnumstr = [str(x + 1) for x in dnumvec]

        # print(dnumstr)

        # plt.rc('xtick', labelsize=6)   # Fontsize of tick labels
        plt.xticks(np.arange(len(dnumvec)) + 1, dnumstr)
    else:
        plt.xlim(0, districts + 1)

    # Set y axis limits based on data; in particular actual/compaison plans
    if have_actual:
        plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)
    elif comp_plans:
        # Get min and max of comparison plans 
        ymin = 0.5
        ymax = 0.5
        for i in np.arange(len(comp_plans_vv)):
            y1 = np.array(sorted(comp_plans_vv[i]))
            ymin = np.min([y1[0], ymin])
            ymax = np.max([y1[-1], ymax])

        plt.ylim(ymin - vbuffer, ymax + vbuffer)
        # print([ymin, ymax])
    else:
        plt.ylim(vs_lower[0] - vbuffer, vs_upper[-1] + vbuffer)

    plt.xlabel('District Number')
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot


def vote_vector_ensemble(ensemble: np.ndarray, title: str, pc_thresh: float = 0.01, have_actual: bool = True,
                         comparison_label: str = "Actual Vote Shares") -> Any:
    # INPUTS
    #     ensemble:  (nDistrict x nPlan) array of values. MUST BE SORTED WITHIN EACH COLUMN
    #     title: string for title
    #
    # OPTIONAL INPUTS [name/def/[default value]
    #     pc_thresh:  quantile markers for violin plots (e.g.: 0.01 means 1% and 99%)
    #     have_actual:   Does ensemble include data assoc. with an exacted plan? 
    #             If so, assume FIRST COLUMN is enacted/[True]  

    # for making things look nice
    vbuffer = .02

    # get shape of data
    districts, chainlength = np.shape(ensemble)

    # list of integers (for plotting)
    seats = np.arange(districts) + 1

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5 * chainlength), chainlength, 10)
    vs_ensemble = [ensemble[ii, samples_to_plot] for ii in range(districts)]
    vs_lower = [np.percentile(vse, 100 * pc_thresh) for vse in vs_ensemble]
    vs_median = [np.percentile(vse, 50) for vse in vs_ensemble]
    vs_upper = [np.percentile(vse, 100 * (1 - pc_thresh)) for vse in vs_ensemble]

    # create Seats/votes curve for enacted plan
    # identify stuffed, packed, and cracked districts
    # ONLY makes sense if actual plan present
    if have_actual:
        vs_actual = np.array(sorted(ensemble[:, 0]))
        # print(np.sum([a - b for a, b in zip(vs_actual, vs_median)]))

        dmin = 0
        dmax = districts
        stuffed = [ii for ii in range(dmin, dmax) if (vs_upper[ii] < vs_actual[ii] < .5)]
        # if len(stuffed) > 0:  dmin = stuffed[-1]
        cracked = [ii for ii in range(dmin, dmax) if
                   (vs_lower[ii] > vs_actual[ii] > .3 and vs_actual[ii] < .5)]
        # if len(cracked) > 0:  dmin = cracked[-1]
        packed = [ii for ii in range(dmin, dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5)]
    else:
        vs_actual = None
        stuffed = []
        cracked = []
        packed = []

    # -----------------------------------------------

    # Create the plot
    myplot = plt.figure(figsize=(6.5, 3.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6,
                   quantiles=[[pc_thresh, 1 - pc_thresh] for _ in seats])
    plt.plot(seats, vs_median, 'bo', markersize=2, label="Median of Ensemble")
    if have_actual:
        marker_size = get_comparison_market_size(districts)
        plt.plot(seats, vs_actual, 'ro', markersize=marker_size, label=comparison_label)
    plt.plot(seats, 0 * seats + 0.5, 'k--', lw=0.75, label="Needed to Win")

    # labelling stuffed districts
    if len(stuffed) > 2:
        smin_loc = 1 + stuffed[0] - 0.5
        smax_loc = 1 + stuffed[-1] + 0.5
        smin_val = vs_actual[stuffed[0]] - vbuffer
        smax_val = vs_actual[stuffed[-1]] + vbuffer

        if USED_STUFFED_IN_PLOTS:
            plt.fill([smin_loc, smax_loc, smax_loc, smin_loc], [smin_val, smin_val, smax_val, smax_val], 'y', alpha=0.3)
            plt.text(0.5 * (smin_loc + smax_loc), smax_val + vbuffer, '"Stuffing"',
                     **{'ha': 'center', 'weight': 'bold'})

    # labelling cracked districts
    if len(cracked) > 2:
        cmin_loc = 1 + cracked[0] - 0.5
        cmax_loc = 1 + cracked[-1] + 0.5
        cmin_val = vs_actual[cracked[0]] - vbuffer
        cmax_val = vs_actual[cracked[-1]] + vbuffer
        plt.fill([cmin_loc, cmax_loc, cmax_loc, cmin_loc], [cmin_val, cmin_val, cmax_val, cmax_val], 'y', alpha=0.3)
        plt.text(0.5 * (cmin_loc + cmax_loc), cmin_val - 2 * vbuffer, '"Cracking"',
                 **{'ha': 'center', 'weight': 'bold'})

    # labelling packed districts
    if len(packed) > 2:
        pmin_loc = 1 + packed[0] - 0.5
        pmax_loc = 1 + packed[-1] + 0.5
        pmin_val = vs_actual[packed[0]] - vbuffer
        pmax_val = vs_actual[packed[-1]] + vbuffer
        plt.fill([pmin_loc, pmax_loc, pmax_loc, pmin_loc], [pmin_val, pmin_val, pmax_val, pmax_val], 'y', alpha=0.3)
        plt.text(0.5 * (pmin_loc + pmax_loc), pmax_val - 2 * vbuffer, '"Packing"', **{'ha': 'center', 'weight': 'bold'})

    plt.xlim(0, districts + 1)
    if have_actual:
        plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)
    else:
        plt.ylim(vs_lower[0] - vbuffer, vs_upper[-1] + vbuffer)

    plt.xlabel('District Number')
    plt.ylabel('Democratic vote share')
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot


def seats_votes_varying_maps_2(ensemble: np.ndarray, title: str, pc_thresh: float = 0.05) -> Any:
    # for making things look nice
    vbuffer = .02

    # get shape of data
    districts, chainlength = np.shape(ensemble)

    # list of integers (for plotting)
    seats = np.arange(districts) + 1

    # create Seats/votes curve for enacted plan
    vs_actual = np.array(sorted(ensemble[:, 0]))

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5 * chainlength), chainlength, 10)
    vs_ensemble = [ensemble[ii, samples_to_plot] for ii in range(districts)]
    vs_median = np.array([np.percentile(vse, 50) for vse in vs_ensemble])
    vs_lower = np.array([np.percentile(vse, 100 * pc_thresh) for vse in vs_ensemble])
    vs_upper = np.array([np.percentile(vse, 100 * (1 - pc_thresh)) for vse in vs_ensemble])

    # identify stuffed, packed, and cracked districts
    inrange = np.array([ii for ii in range(districts) if vs_upper[ii] > vs_actual[ii] > vs_lower[ii]],
                       dtype=np.int32)
    packed = np.array([ii for ii in range(districts) if vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5],
                      dtype=np.int32)
    stuffed = np.array([ii for ii in range(districts) if vs_upper[ii] < vs_actual[ii] < .5],
                       dtype=np.int32)
    cracked = np.array([ii for ii in range(districts) if vs_actual[ii] < vs_lower[ii] and vs_actual[ii] < .5],
                       dtype=np.int32)

    # -----------------------------------------------

    # Create the plot
    myplot = plt.figure(figsize=(6.5, 4.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6,
                   quantiles=[[pc_thresh, 1 - pc_thresh] for _ in seats])
    plt.plot(seats, vs_median, 'ko', markersize=2, label="Median of Ensemble")
    plt.scatter(1 + inrange, vs_actual[inrange], s=5, color="black", label='Enacted: in Range')
    plt.scatter(1 + packed, vs_actual[packed], s=8, color="red", label='Enacted: "Packed"')
    plt.scatter(1 + cracked, vs_actual[cracked], s=8, color="blue", label='Enacted: "Cracked"')
    if USED_STUFFED_IN_PLOTS:
        plt.scatter(1 + stuffed, vs_actual[stuffed], s=8, color="orange", label='Enacted: "Stuffed"')
    plt.plot(seats, 0 * seats + 0.5, 'k--', lw=0.75, label="Needed to Win")

    plt.xlim(0, districts + 1)
    plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)

    plt.xlabel('District Number')
    plt.ylabel('Democratic vote share')
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot


def seats_votes_ensemble(ensemble: np.ndarray, title: str, statewide: Any = None, have_actual: bool = True) -> Any:
    # INPUTS
    #     ensemble:  (nDistrict x nPlan) array of values. MUST BE SORTED WITHIN EACH COLUMN
    #     title: string for title
    # OPTIONAL INPUTS [name/def/[default value]
    #    statewide/ Statewide Democratic vote %. If None, use mean of enacted vote vector/[None]
    #    have_actual/Does ensemble include data assoc. with an exacted plan? 
    #             If so, assume FIRST COLUMN is enacted/[True]

    # get shape of data
    districts, chainlength = np.shape(ensemble)

    if have_actual:
        # get the actual outcomes
        actuals = np.array(sorted(ensemble[:, 0], reverse=True))
        actual_seats = np.sum(actuals > .5)
        actual_votes = np.mean(actuals) if statewide is None else statewide
    else:
        # If None, we just don't plot
        actual_votes = statewide

    if have_actual:
        # seats / votes curve, assuming uniform swing
        seatsvotes1 = actual_votes - actuals + 0.5

        # apply reflection about (.5, .5)
        seatsvotes2 = np.flip(1 - seatsvotes1)

    # now compute an average seats/votes over many samples
    # ------------------------------------------------
    avg_range = range(int(chainlength / 2), chainlength)
    avg_seatsvotes = 0 * np.array(sorted(ensemble[:, 0], reverse=True))
    avg_seats = 0
    avg_votes = 0
    for step in avg_range:
        tmp_race_results = np.array(sorted(ensemble[:, step], reverse=True))
        tmp_seats = np.sum(tmp_race_results > .5)
        tmp_votes = np.mean(tmp_race_results) if statewide is None else statewide
        tmp_seatsvotes = [tmp_votes - r + 0.5 for r in tmp_race_results]

        avg_seatsvotes += tmp_seatsvotes
        avg_seats += tmp_seats
        avg_votes += tmp_votes

    avg_seatsvotes /= len(avg_range)
    avg_votes /= len(avg_range)
    avg_seats /= len(avg_range)

    # Convert to arrays, reflect seats-votes curve about (.5, .5)
    avg_seatsvotes1 = np.array(avg_seatsvotes)
    avg_seatsvotes2 = np.flip(1 - avg_seatsvotes1)

    #     plot them together in the same figure
    # -------------------------------------------------

    seats = range(1, districts + 1)

    myfigure = plt.figure(figsize=(6.5, 5))

    plt.subplot(211)
    plt.title(title)

    # panel 1: the ensemble
    # -------------------
    plt.plot(seats, avg_seatsvotes1, 'b', lw=2, label="Democrats")
    plt.plot(seats, avg_seatsvotes2, 'r', lw=2, label="Republicans")
    plt.fill_between(seats, avg_seatsvotes1, avg_seatsvotes2, where=(avg_seatsvotes1 > avg_seatsvotes2),
                     interpolate=True, **{'alpha': .2})

    # labeling
    hbuff = 0.25
    vbuff = 0.01
    dpos = [ii for ii in range(districts) if avg_seatsvotes1[ii] > 0.5]
    first = dpos[0]
    start = avg_seatsvotes1[first - 1]
    final = avg_seatsvotes1[first]
    slope = final - start
    hmin = 1 + (first - 1) + (0.5 - start) / slope
    hmid = 1 + (districts - 1) / 2.0
    hmax = hmin + 2 * (hmid - hmin)
    vmax = avg_seatsvotes1[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
            avg_seatsvotes1[round(hmid - 0.5) - 1] + avg_seatsvotes1[round(hmid + 0.5) - 1])
    vmid = 0.5
    vmin = vmax - 2 * (vmax - vmid)
    plt.plot([hmin - hbuff, hmax + hbuff], [vmid, vmid], '--', lw=2, color="green", label="Equal Voteshares")
    plt.plot([hmid, hmid], [vmin - vbuff, vmax + vbuff], '--', lw=2, color="gray", label="Majority Seat")

    # Should "actual" result be the location on the curve at the actual statewide percentage?
    # vs. the 
    plt.plot([avg_seats + 0.5], [avg_votes], 'bs', lw=2, label="Actual Dem. Result")
    plt.plot([districts - avg_seats + 0.5], [1 - avg_votes], 'rs', label="Actual Rep. Result")

    print("Vote Needed for Majority (D-R) -- Ensemble:   %4.4f" % (vmax - vmin))
    print("Seats at 50%% Voteshare   (D-R) -- Ensemble:     %4d" % (hmax - hmin))

    # cleaning up
    plt.xlim(0, districts + 1)
    plt.ylim(0.25, 0.75)
    plt.text(hmid, .7, "Average of \n Sampled Plans", **{'va': 'top', 'ha': 'center', 'size': 10, 'weight': 'bold'})
    plt.ylabel("Statewide Vote Share")
    # plt.xlabel("Number of Seats")
    plt.legend(**{'fontsize': 8})

    if have_actual:
        # panel 2
        # -------------------
        plt.subplot(212)

        plt.title(title)

        plt.plot(seats, seatsvotes1, 'b', lw=2, label="Democrats")
        plt.plot(seats, seatsvotes2, 'r', lw=2, label="Republicans")
        plt.fill_between(seats, seatsvotes1, seatsvotes2, where=(seatsvotes1 > seatsvotes2), interpolate=True,
                         **{'alpha': .2})

        # labeling
        dpos = [ii for ii in range(districts) if seatsvotes1[ii] > 0.5]
        first = dpos[0]
        start = seatsvotes1[first - 1]
        final = seatsvotes1[first]
        slope = final - start

        hmin = 1 + (first - 1) + (0.5 - start) / slope
        hmid = 1 + (districts - 1) / 2.0
        hmax = hmin + 2 * (hmid - hmin)

        vmax = seatsvotes1[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
                seatsvotes1[round(hmid - 0.5) - 1] + seatsvotes1[round(hmid + 0.5) - 1])
        vmid = 0.5
        vmin = vmax - 2 * (vmax - vmid)

        plt.plot([hmin - hbuff, hmax + hbuff], [vmid, vmid], '--', lw=2, color="green", label="Equal Voteshares")
        plt.plot([hmid, hmid], [vmin - vbuff, vmax + vbuff], '--', lw=2, color="gray", label="Majority Seat")
        plt.plot([actual_seats + 0.5], [actual_votes], 'bs', lw=2, label="Actual Dem. Result")
        plt.plot([districts - actual_seats + 0.5], [1 - actual_votes], 'rs', label="Actual Rep. Result")

        print("Vote Needed for Majority (D-R) -- Actual:     %4.4f" % (vmax - vmin))
        print("Seats at 50%% Voteshare   (D-R) -- Actual:       %4d" % (hmax - hmin))

        # cleaning up
        plt.xlim(0, districts + 1)
        plt.ylim(0.25, 0.75)
        plt.ylabel("Percent Vote Share")
        plt.xlabel("Number of Seats")
        plt.text(hmid, .7, "Enacted Plan", **{'va': 'top', 'ha': 'center', 'size': 10, 'weight': 'bold'})
        plt.legend(**{'fontsize': 8})
        plt.tight_layout()

    return myfigure


# statewide=None,
def mean_median_partisan_bias(ensemble: np.ndarray, have_actual: bool = True,
                              comparison_label: str = "Enacted Districts"):
    # get overall voteshare for party A
    # overall_result = 0
    # if statewide != None:
    #    overall_result = statewide
    # else:
    #     overall_result = np.mean(np.mean(ensemble))
    # Do we actually use overall_result? 
    # Below we appear to use the mean of each vote-share vector

    # get shape of data
    districts, chainlength = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0

    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats = np.zeros((2, chainlength))
    for j in range(0, chainlength):
        race_results = sorted(ensemble[:, j], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1 - seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0, j] = seatsvotes1[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
                seatsvotes1[round(hmid - 0.5) - 1] + seatsvotes1[round(hmid - 0.5) - 1])
        majority_vs[1, j] = seatsvotes2[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
                seatsvotes2[round(hmid - 0.5) - 1] + seatsvotes2[round(hmid - 0.5) - 1])

        # What is the number of seats you get at 50%?
        number_seats[0, j] = np.sum(seatsvotes1 <= 0.5)
        number_seats[1, j] = np.sum(seatsvotes2 <= 0.5)

    # things we need to plot
    vdiffs = majority_vs[0, :] - majority_vs[1, :]
    mvdiff = np.median(vdiffs)
    cvdiff = vdiffs[0]

    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max + .01, .01)
    vdbuff = (vdmax - vdmin) * .025

    vhist, vedges = np.histogram(vdiffs, bins=vbinbounds)
    if have_actual:
        print("MM Enacted Plan Percentile = ", stats.percentileofscore(vdiffs, cvdiff))
        print(majority_vs[:, 0])
        print(mvdiff)

    # things we need to plot
    ndiffs = number_seats[0, :] - number_seats[1, :]
    mndiff = np.median(ndiffs)
    cndiff = ndiffs[0]

    # for making it look nice
    ndmin = min(ndiffs)
    ndmax = max(ndiffs)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min - 0.5, nbinedge_max + 0.5)
    ndbuff = (ndmax - ndmin) * .01

    nhist, nedges = np.histogram(ndiffs, bins=nbinbounds)
    if have_actual:
        print("PB Enacted Plan Percentile = ", stats.percentileofscore(ndiffs, cndiff))
        print(number_seats[:, 0])
        print(mndiff)
        print(np.count_nonzero(ndiffs == cndiff))
        print(np.count_nonzero(ndiffs < cndiff))
        print(min(ndiffs))

    myfigure = plt.figure(figsize=(6.5, 3.5))
    axes = plt.subplot(121)

    plt.hist(vdiffs, bins=vbinbounds, color="xkcd:dark blue", **{'alpha': 0.3})
    # sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mvdiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    if have_actual:
        plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label=comparison_label)
    plt.xlim(vdmin - vdbuff, vdmax + vdbuff)
    axes.xaxis.set_major_formatter(lambda x, pos: format(100 * x, ".0f"))

    plt.ylim(0, np.max(vhist) * 1.4)
    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.xlabel("Voteshare Difference for Majority % (D-R)")
    plt.ylabel("Relative Frequency")
    plt.title('"Mean-Median" Score')
    plt.legend(loc="upper center")

    plt.subplot(122)
    plt.hist(ndiffs, bins=nbinbounds, color="xkcd:dark blue", **{'alpha': 0.3})
    # sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mndiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    if have_actual:
        plt.axvline(x=cndiff, color="purple", ls='--', lw=2.5, ymax=0.75, label=comparison_label)
    plt.xlim(ndmin - ndbuff, ndmax + ndbuff)
    plt.ylim(0, np.max(nhist) * 1.4)

    if (nbinedge_max - nbinedge_min) <= 12:
        plt.xticks(np.arange(nbinedge_min, nbinedge_max, 2))
    else:
        plt.xticks(np.arange(nbinedge_min, nbinedge_max, 4))

    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.title('"Partisan Bias" Score')
    plt.xlabel("Seat Difference at Equal Votes (D-R)")
    # plt.ylabel("Frequency")
    plt.legend(loc="upper center")

    plt.tight_layout()
    return myfigure


def partisan_metrics_histpair(ensemble: np.ndarray, instance: np.ndarray) -> Any:  # , statewide=None):
    # get overall voteshare for party A
    # overall_result = 0
    # if statewide != None:
    #     overall_result = statewide
    # else:
    #     overall_result = np.mean(np.mean(ensemble))

    # get shape of data
    chainlength, districts = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0

    # logic for odd or even numbers of districts
    # odd_number_of_districts = (districts % 2 != 0)
    odd_midindex = round(hmid) - 1
    evl_midindex = round(hmid - 0.5) - 1
    evr_midindex = round(hmid + 0.5) - 1

    # information for the instance
    actual_race_results = sorted(instance, reverse=True)
    actual_mean_voteshare = np.mean(actual_race_results)
    actual_seatsvotes1 = actual_mean_voteshare - np.array(actual_race_results) + 0.5
    actual_seatsvotes2 = np.flip(1 - actual_seatsvotes1)

    # What is the percentage nec. for a majority?
    actual_majority_vs1 = actual_seatsvotes1[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
            actual_seatsvotes1[round(hmid - 0.5) - 1] + actual_seatsvotes1[round(hmid - 0.5) - 1])
    actual_majority_vs2 = actual_seatsvotes2[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
            actual_seatsvotes2[round(hmid - 0.5) - 1] + actual_seatsvotes2[round(hmid - 0.5) - 1])
    cvdiff = actual_majority_vs1 - actual_majority_vs2

    # What is the number of seats arising from a split vote?
    # actual_number_seats1 = np.sum(actual_seatsvotes1 <= 0.5)
    # actual_number_seats2 = np.sum(actual_seatsvotes2 <= 0.5)

    # What is the number of seats arising from a split vote?
    actual_number_seats_int1 = np.sum(actual_seatsvotes1 <= 0.5)
    seatindex1 = actual_number_seats_int1 - 1
    v0 = actual_seatsvotes1[seatindex1]
    v1 = actual_seatsvotes1[seatindex1 + 1]
    actual_number_seats_float1 = actual_number_seats_int1 + (0.5 - v0) / (v1 - v0)

    actual_number_seats_int2 = np.sum(actual_seatsvotes2 <= 0.5)
    seatindex2 = actual_number_seats_int2 - 1
    v0 = actual_seatsvotes2[seatindex2]
    v1 = actual_seatsvotes2[seatindex2 + 1]
    actual_number_seats_float2 = actual_number_seats_int2 + (0.5 - v0) / (v1 - v0)

    cndiff_int = actual_number_seats_int1 - actual_number_seats_int2
    cndiff_float = actual_number_seats_float1 - actual_number_seats_float2

    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats_float = np.zeros((2, chainlength))
    number_seats_int = np.zeros((2, chainlength))

    # seats votes curve for the ensemble
    for j in range(0, chainlength):
        race_results = sorted(ensemble[j, :], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1 - seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0, j] = seatsvotes1[odd_midindex] if districts % 2 != 0 else 0.5 * (
                seatsvotes1[evl_midindex] + seatsvotes1[evr_midindex])
        majority_vs[1, j] = seatsvotes2[odd_midindex] if districts % 2 != 0 else 0.5 * (
                seatsvotes2[evl_midindex] + seatsvotes2[evr_midindex])

        # What is the number of seats arising from a split vote?
        number_seats_int[0, j] = np.sum(seatsvotes1 <= 0.5)
        lastseat1 = round(number_seats_int[0, j] - 1)
        v0 = seatsvotes1[lastseat1]
        v1 = seatsvotes1[lastseat1 + 1]
        number_seats_float[0, j] = number_seats_int[0, j] + (0.5 - v0) / (v1 - v0)

        number_seats_int[1, j] = np.sum(seatsvotes2 <= 0.5)
        lastseat2 = round(number_seats_int[1, j] - 1)
        v0 = seatsvotes2[lastseat2]
        v1 = seatsvotes2[lastseat2 + 1]
        number_seats_float[1, j] = number_seats_int[1, j] + (0.5 - v0) / (v1 - v0)

    #
    #   Voteshares
    #
    # things we need to plot
    vdiffs = majority_vs[0, :] - majority_vs[1, :]
    mvdiff = np.median(vdiffs)

    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max + .01, .01)

    vdbuff = (vdmax - vdmin) * .2

    vhist, vedges = np.histogram(vdiffs, bins=vbinbounds)
    print("Mean-Median:            %4.2f  (%8.6f percentile)." % (cvdiff, stats.percentileofscore(vdiffs, cvdiff)))

    #
    #   Number of Seats
    #
    # things we need to plot
    ndiffs_int = number_seats_int[0, :] - number_seats_int[1, :]
    # mndiff_int = np.median(ndiffs_int)

    ndiffs_float = number_seats_float[0, :] - number_seats_float[1, :]
    mndiff_float = np.median(ndiffs_float)

    # for making it look nice
    ndmin = min(ndiffs_float)
    ndmax = max(ndiffs_float)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min - 0.5, nbinedge_max + 0.5)
    ndbuff = (ndmax - ndmin) * .2

    nhist, nedges = np.histogram(ndiffs_float, bins=nbinbounds)
    print("Partisan Bias (int):    %4.2f  (%8.6f percentile)." % (
        cndiff_int, stats.percentileofscore(ndiffs_int, cndiff_int)))
    print("Partisan Bias (float):  %4.2f  (%8.6f percentile)." % (
        cndiff_float, stats.percentileofscore(ndiffs_float, cndiff_float)))

    # ----------------------------------------
    #   Show the 1D distributions separately
    # ----------------------------------------
    myfigure = plt.figure(figsize=(6.5, 3.5))
    axes = plt.subplot(121)

    plt.hist(vdiffs, bins=vbinbounds, color="xkcd:dark blue", **{'alpha': 0.3})
    # sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mvdiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    # plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Current Districts")
    plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Proposed Districts")
    plt.xlim(vdmin - vdbuff, vdmax + vdbuff)
    axes.xaxis.set_major_formatter(lambda x, pos: format(100 * x, ".0f"))

    plt.ylim(0, np.max(vhist) * 1.4)
    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.xlabel("Voteshare Difference for Majority (D-R)")
    plt.ylabel("Relative Frequency")
    plt.title('"Mean-Median" Score')
    plt.legend(loc="upper center")

    plt.subplot(122)
    plt.hist(ndiffs_float, bins=nbinbounds, color="xkcd:dark blue", **{'alpha': 0.3})
    # sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")

    plt.axvline(x=mndiff_float, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    # plt.axvline(x=cndiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Current Districts")
    plt.axvline(x=cndiff_float, color="purple", ls='--', lw=2.5, ymax=0.75, label="Proposed Districts")
    plt.xlim(ndmin - ndbuff, ndmax + ndbuff)

    plt.ylim(0, np.max(nhist) * 1.4)
    if (nbinedge_max - nbinedge_min) <= 12:
        plt.xticks(np.arange(nbinedge_min, nbinedge_max, 2))
    else:
        plt.xticks(np.arange(nbinedge_min, nbinedge_max, 4))
    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.title('"Partisan Bias" Score')
    plt.xlabel("Seat Difference at Equal Votes (D-R)")
    # plt.ylabel("Frequency")
    plt.legend(loc="upper center")

    plt.tight_layout()
    return myfigure


def partisan_metrics_hist2D(ensemble: np.ndarray, instance: np.ndarray, comparison_label: str) -> Any:  # , statewide=None):
    # get overall voteshare for party A
    # overall_result = 0
    # if statewide != None:
    #    overall_result = statewide
    # else:
    #    overall_result = np.mean(np.mean(ensemble))

    # get shape of data
    chainlength, districts = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0

    # logic for odd or even numbers of districts
    # odd_number_of_districts = (districts % 2 != 0)
    odd_midindex = round(hmid) - 1
    evl_midindex = round(hmid - 0.5) - 1
    evr_midindex = round(hmid + 0.5) - 1

    # information for the instance
    actual_race_results = sorted(instance, reverse=True)
    actual_mean_voteshare = np.mean(actual_race_results)
    actual_seatsvotes1 = actual_mean_voteshare - np.array(actual_race_results) + 0.5
    actual_seatsvotes2 = np.flip(1 - actual_seatsvotes1)

    # What is the percentage nec. for a majority?
    actual_majority_vs1 = actual_seatsvotes1[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
            actual_seatsvotes1[round(hmid - 0.5) - 1] + actual_seatsvotes1[round(hmid - 0.5) - 1])
    actual_majority_vs2 = actual_seatsvotes2[round(hmid) - 1] if districts % 2 != 0 else 0.5 * (
            actual_seatsvotes2[round(hmid - 0.5) - 1] + actual_seatsvotes2[round(hmid - 0.5) - 1])
    cvdiff = actual_majority_vs1 - actual_majority_vs2

    # What is the percentage nec. for a majority?
    actual_number_seats1 = np.count_nonzero(actual_seatsvotes1 <= 0.5)
    actual_number_seats2 = np.count_nonzero(actual_seatsvotes2 <= 0.5)
    cndiff = actual_number_seats1 - actual_number_seats2

    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats = np.zeros((2, chainlength))

    # seats votes curve for the ensemble
    for j in range(0, chainlength):
        if j % 100000 == 0:
            print(j)

        race_results = sorted(ensemble[j, :], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1 - seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0, j] = seatsvotes1[odd_midindex] if districts % 2 != 0 else 0.5 * (
                seatsvotes1[evl_midindex] + seatsvotes1[evr_midindex])
        majority_vs[1, j] = seatsvotes2[odd_midindex] if districts % 2 != 0 else 0.5 * (
                seatsvotes2[evl_midindex] + seatsvotes2[evr_midindex])

        # What is the number of seats arising from a split vote?
        lastseat1 = np.count_nonzero(seatsvotes1 <= 0.5) - 1
        v0 = seatsvotes1[lastseat1]
        v1 = seatsvotes1[lastseat1 + 1]
        number_seats[0, j] = lastseat1 + 1 + (0.5 - v0) / (v1 - v0)

        lastseat2 = np.count_nonzero(seatsvotes2 <= 0.5) - 1
        v0 = seatsvotes2[lastseat2]
        v1 = seatsvotes2[lastseat2 + 1]
        number_seats[1, j] = lastseat2 + 1 + (0.5 - v0) / (v1 - v0)

    #
    #   Voteshares
    #
    # things we need to plot
    vdiffs = majority_vs[0, :] - majority_vs[1, :]
    # mvdiff = np.median(vdiffs)

    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max + .01, .01)

    # vdbuff = (vdmax - vdmin) * .2

    np.histogram(vdiffs, bins=vbinbounds)
    print("MM Enacted Plan Percentile = %8.6f" % stats.percentileofscore(vdiffs, cvdiff))

    #
    #   Number of Seats
    #
    # things we need to plot
    ndiffs = number_seats[0, :] - number_seats[1, :]
    # mndiff = np.median(ndiffs)

    # for making it look nice
    ndmin = min(ndiffs)
    ndmax = max(ndiffs)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min - 0.5, nbinedge_max + 0.5)
    # ndbuff = (ndmax - ndmin) * .2

    np.histogram(ndiffs, bins=nbinbounds)
    print("PB Enacted Plan Percentile = %8.6f" % stats.percentileofscore(ndiffs, cndiff))

    # 2D histogram

    myfigure = plt.figure(figsize=(6.5, 5))
    # Basic 2D density plot
    sns.set_style("white")
    print("KDE Plot Start")
    figure = sns.kdeplot(x=vdiffs, y=ndiffs, cmap="Blues", shade=True, cbar=True,
                         levels=[.0001, .001, .01, .05, .25, .5, .75, 1.0])  # , bw_adjust=.5
    plt.xticks(figure.get_xticks(), map(lambda x: format(100 * x, ".0f"), figure.get_xticks()))
    # plt.plot([cvdiff], [cndiff], 'rs', label="Current Plan")
    print("KDE Plot End")
    plt.plot([cvdiff], [cndiff], 'rs', label=comparison_label)
    plt.xlabel("Mean-Median %")
    plt.ylabel("Partisan Bias")
    plt.title("2D Histogram of Ensemble Partisan Metrics")
    plt.legend(loc='best')
    plt.tight_layout()

    return myfigure


def mean_median_distribution(ensemble: np.ndarray, instance: np.ndarray,
                             title: str = "Distribution of Mean-Median Scores", xlabel: str = None) -> Any:
    mme = np.mean(ensemble, axis=1) - np.median(ensemble, axis=1)
    mmi = np.mean(instance) - np.median(instance)

    mmmedian = np.median(mme)
    mmmean = np.mean(mme)
    mmstd = np.std(mme)
    mmbinedges = np.linspace(mmmean - 5 * mmstd, mmmean + 5 * mmstd, 51)

    # for making it look nice
    # mmmin = np.min([np.min(mme), mmi])
    # mmmax = np.max([np.max(mmi), mmi])
    # mmbinedge_min = np.floor(mmmin)
    # mmbinedge_max = np.ceil(mmmax)
    # nbinbounds = np.arange(nbinedge_min-0.5, nbinedge_max+0.5)
    # ndbuff = (ndmax - ndmin)*.2

    print("Percentile Score:  %8.6f." % (stats.percentileofscore(mme, mmi)))

    mmhist, mmedges = np.histogram(mme, bins=mmbinedges)

    myfigure = plt.figure(figsize=(6.5, 3.5))

    plt.hist(mme, bins=mmbinedges, color="xkcd:dark blue", **{'alpha': 0.3})
    # sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mmmedian, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    # plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Current Districts")
    plt.axvline(x=mmi, color="purple", ls='--', lw=2.5, ymax=0.75, label="Proposed Districts")
    plt.xlim(mmmean - 4 * mmstd, mmmean + 4 * mmstd)

    plt.ylim(0, np.max(mmhist) * 1.4)
    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.legend(loc="upper center")
    plt.xlabel(xlabel)
    plt.ylabel("Relative Frequency")
    plt.title(title)

    return myfigure


def racial_vs_political_deviations(ensemble_p: np.ndarray, instance_p: np.ndarray, ensemble_r: np.ndarray,
                                   instance_r: np.ndarray, title: str) -> Any:
    # get shape of data
    chainlength, districts = np.shape(ensemble_p)
    dist_list = np.arange(1, districts + 1)

    # sort the instance_p, and record the district numbers
    sorted_districts_p = np.argsort(instance_p) + 1
    sorted_voteshares_p = sorted(instance_p)
    median_voteshares_p = np.median(ensemble_p)
    vote_diffs_by_rank = sorted_voteshares_p - median_voteshares_p
    vote_diffs_by_dist = [vote_diffs_by_rank[ii] for ii in np.argsort(sorted_districts_p)]

    # sort the instance_r, and record the district numbers
    sorted_districts_r = np.argsort(instance_r) + 1
    sorted_popshares_r = sorted(instance_r)
    median_popshares_r = np.median(ensemble_r)
    pop_diffs_by_rank = sorted_popshares_r - median_popshares_r
    pop_diffs_by_dist = [pop_diffs_by_rank[ii] for ii in np.argsort(sorted_districts_r)]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)
    ax.plot(pop_diffs_by_dist, vote_diffs_by_dist, 'bs')
    ax.plot([0, 0], [-.3, .3], 'k--')
    ax.plot([-.2, .4], [0, 0], 'k--')
    ax.set_xlabel('Minority Population Deviation')
    ax.set_ylabel('Democratic Voter Deviation')

    for i, txt in enumerate(dist_list):
        ax.annotate(txt, (pop_diffs_by_dist[i] + .005, vote_diffs_by_dist[i] + .005))

    return fig
