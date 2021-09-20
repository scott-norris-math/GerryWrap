import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats



def vote_vector_ensemble_comps(ensemble, title, pc_thresh=0.01,have_actual=True, \
                        comp_plans=False, comp_plans_vv=[], comp_plans_names=[], comp_plans_colors=[], \
                        comp_plans_pnums=[]):
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
    vbuffer = .02
        
    # get shape of data
    districts, chainlength = np.shape(ensemble)
    
    # list of integers (for plotting)
    seats = np.arange(districts)+1

    # create Seats/votes curve for enacted plan
    if have_actual:
        vs_actual = np.array(sorted(ensemble[:,0]))

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5*chainlength), chainlength, 10)
    vs_ensemble = [ ensemble[ii,samples_to_plot] for ii in range(districts) ]
    vs_lower  = [ np.percentile(vse, 100*pc_thresh) for vse in vs_ensemble ]
    vs_median = [ np.percentile(vse, 50) for vse in vs_ensemble ]
    vs_upper  = [ np.percentile(vse, 100*(1-pc_thresh)) for vse in vs_ensemble ]
    
    if have_actual:
        print(np.sum([a-b for a,b in zip(vs_actual,vs_median)]))
    
    # identify stuffed, packed, and cracked districts
    # ONLY makes sense if actual plan present
    if have_actual:
        dmin = 0
        dmax = districts
        stuffed = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] < .5) ]
        #if len(stuffed) > 0:  dmin = stuffed[-1]
        cracked = [ii for ii in range(dmin,dmax) if (vs_actual[ii] < vs_lower[ii] and vs_actual[ii] < .5 and vs_actual[ii] > .3) ]
        #if len(cracked) > 0:  dmin = cracked[-1]
        packed  = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5) ]
    else:
        stuffed = []
        cracked = []
        packed = []


    # -----------------------------------------------

    
    # Create the plot
    myplot = plt.figure(figsize=(6.5, 3.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6, quantiles=[[pc_thresh, 1-pc_thresh] for ii in seats])
    plt.plot(seats, vs_median, 'bo', markersize=2, label="Median of Ensemble")
    if have_actual:
        plt.plot(seats, vs_actual, 'ro', markersize=4, label="Actual Vote Shares")
    plt.plot(seats, 0*seats+0.5, 'k--', lw=0.75, label="Needed to Win")

    # ---------------------------------------------
    
    # labelling stuffed districts
    if len(stuffed) > 2:
        smin_loc = 1+stuffed[0]-0.5
        smax_loc = 1+stuffed[-1]+0.5
        smin_val = vs_actual[stuffed[0]] - vbuffer
        smax_val = vs_actual[stuffed[-1]] + vbuffer
        plt.fill([smin_loc, smax_loc, smax_loc, smin_loc],[smin_val, smin_val, smax_val, smax_val], 'y', alpha=0.3)
        plt.text(0.5*(smin_loc+smax_loc), smax_val+vbuffer, '"Stuffing"', **{'ha':'center', 'weight':'bold'})
        
    # labelling cracked districts
    if len(cracked) > 2:
        cmin_loc = 1+cracked[0]-0.5
        cmax_loc = 1+cracked[-1]+0.5
        cmin_val = vs_actual[cracked[0]] - vbuffer
        cmax_val = vs_actual[cracked[-1]] + vbuffer
        plt.fill([cmin_loc, cmax_loc, cmax_loc, cmin_loc],[cmin_val, cmin_val, cmax_val, cmax_val], 'y', alpha=0.3)
        plt.text(0.5*(cmin_loc+cmax_loc), cmin_val-2*vbuffer, '"Cracking"', **{'ha':'center', 'weight':'bold'})
    
    # labelling packed districts
    if len(packed) > 2:
        pmin_loc = 1+packed[0]-0.5
        pmax_loc = 1+packed[-1]+0.5
        pmin_val = vs_actual[packed[0]] - vbuffer
        pmax_val = vs_actual[packed[-1]] + vbuffer
        plt.fill([pmin_loc, pmax_loc, pmax_loc, pmin_loc],[pmin_val, pmin_val, pmax_val, pmax_val], 'y', alpha=0.3)
        plt.text(0.5*(pmin_loc+pmax_loc), pmax_val-2*vbuffer, '"Packing"', **{'ha':'center', 'weight':'bold'})

    # ---------------------------------------------------------    
        
    # "Comparison Plans", if applicable
    if comp_plans:
        # Determine if district number should be plotted
        true_pnums = [i for i, x in enumerate(comp_plans_pnums) if x]
        whpnum     = true_pnums[0]
        
        # Order of districts for x-axis
        dnumvec    = np.argsort(comp_plans_vv[whpnum])
        
        for i in np.arange(len(comp_plans_vv)):
            #print(comp_plans_vv[i])
            #print(comp_plans_colors[i])
            #print(comp_plans_names[i])
            y1 = np.array(sorted(comp_plans_vv[i]))
            #print(y1)
            #plt.plot(seats, y1, 'ro', markersize=4, label=comp_plans_names[i])
            plt.plot(seats, y1, color=comp_plans_colors[i], marker='o',markersize=4, ls='',label=comp_plans_names[i])
        
        
    plt.xlim(0, districts+1)
    # Set y axis limits based on data; in particular actual/compaison plans
    if have_actual:
        plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)
    elif comp_plans:
        # Get min and max of comparison plans 
        ymin = 0.5
        ymax = 0.5
        for i in np.arange(len(comp_plans_vv)):
            y1 = np.array(sorted(comp_plans_vv[whpnum]))
            ymin = np.min([y1[0],ymin])
            ymax = np.max([y1[-1],ymax])
        plt.ylim(ymin - vbuffer, ymax + vbuffer)
    else:
        plt.ylim(vs_lower[0]-vbuffer, vs_upper[-1]+vbuffer)
        
    # Replace 1-n with actual district numbers
    if comp_plans:
        # Use dnumvec computed above
        dnumstr= [str(x+1) for x in dnumvec]
        
        print(dnumstr)
        
        #plt.rc('xtick', labelsize=6)   # Fontsize of tick labels
        plt.xticks(np.arange(len(dnumvec))+1,dnumstr)
         
    plt.xlabel('District Number')
    plt.ylabel('Democratic vote share')
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot



def vote_vector_ensemble(ensemble, title, pc_thresh=0.01,have_actual=True):
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
    seats = np.arange(districts)+1

    # create Seats/votes curve for enacted plan
    if have_actual:
        vs_actual = np.array(sorted(ensemble[:,0]))

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5*chainlength), chainlength, 10)
    vs_ensemble = [ ensemble[ii,samples_to_plot] for ii in range(districts) ]
    vs_lower  = [ np.percentile(vse, 100*pc_thresh) for vse in vs_ensemble ]
    vs_median = [ np.percentile(vse, 50) for vse in vs_ensemble ]
    vs_upper  = [ np.percentile(vse, 100*(1-pc_thresh)) for vse in vs_ensemble ]
    
    if have_actual:
        print(np.sum([a-b for a,b in zip(vs_actual,vs_median)]))
    
    # identify stuffed, packed, and cracked districts
    # ONLY makes sense if actual plan present
    if have_actual:
        dmin = 0
        dmax = districts
        stuffed = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] < .5) ]
        #if len(stuffed) > 0:  dmin = stuffed[-1]
        cracked = [ii for ii in range(dmin,dmax) if (vs_actual[ii] < vs_lower[ii] and vs_actual[ii] < .5 and vs_actual[ii] > .3) ]
        #if len(cracked) > 0:  dmin = cracked[-1]
        packed  = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5) ]
    else:
        stuffed = []
        cracked = []
        packed = []


    # -----------------------------------------------

    
    # Create the plot
    myplot = plt.figure(figsize=(6.5, 3.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6, quantiles=[[pc_thresh, 1-pc_thresh] for ii in seats])
    plt.plot(seats, vs_median, 'bo', markersize=2, label="Median of Ensemble")
    if have_actual:
        plt.plot(seats, vs_actual, 'ro', markersize=4, label="Actual Vote Shares")
    plt.plot(seats, 0*seats+0.5, 'k--', lw=0.75, label="Needed to Win")

    # labelling stuffed districts
    if len(stuffed) > 2:
        smin_loc = 1+stuffed[0]-0.5
        smax_loc = 1+stuffed[-1]+0.5
        smin_val = vs_actual[stuffed[0]] - vbuffer
        smax_val = vs_actual[stuffed[-1]] + vbuffer
        plt.fill([smin_loc, smax_loc, smax_loc, smin_loc],[smin_val, smin_val, smax_val, smax_val], 'y', alpha=0.3)
        plt.text(0.5*(smin_loc+smax_loc), smax_val+vbuffer, '"Stuffing"', **{'ha':'center', 'weight':'bold'})
        
    # labelling cracked districts
    if len(cracked) > 2:
        cmin_loc = 1+cracked[0]-0.5
        cmax_loc = 1+cracked[-1]+0.5
        cmin_val = vs_actual[cracked[0]] - vbuffer
        cmax_val = vs_actual[cracked[-1]] + vbuffer
        plt.fill([cmin_loc, cmax_loc, cmax_loc, cmin_loc],[cmin_val, cmin_val, cmax_val, cmax_val], 'y', alpha=0.3)
        plt.text(0.5*(cmin_loc+cmax_loc), cmin_val-2*vbuffer, '"Cracking"', **{'ha':'center', 'weight':'bold'})
    
    # labelling packed districts
    if len(packed) > 2:
        pmin_loc = 1+packed[0]-0.5
        pmax_loc = 1+packed[-1]+0.5
        pmin_val = vs_actual[packed[0]] - vbuffer
        pmax_val = vs_actual[packed[-1]] + vbuffer
        plt.fill([pmin_loc, pmax_loc, pmax_loc, pmin_loc],[pmin_val, pmin_val, pmax_val, pmax_val], 'y', alpha=0.3)
        plt.text(0.5*(pmin_loc+pmax_loc), pmax_val-2*vbuffer, '"Packing"', **{'ha':'center', 'weight':'bold'})

    plt.xlim(0, districts+1)
    if have_actual:
        plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)
    else:
        plt.ylim(vs_lower[0]-vbuffer, vs_upper[-1]+vbuffer)
        
    plt.xlabel('District Number')
    plt.ylabel('Democratic vote share')
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot




def seats_votes_varying_maps_2(ensemble, title, pc_thresh=0.05):
    
    # for making things look nice
    vbuffer = .02
        
    # get shape of data
    districts, chainlength = np.shape(ensemble)
    
    # list of integers (for plotting)
    seats = np.arange(districts)+1

    # create Seats/votes curve for enacted plan
    vs_actual = np.array(sorted(ensemble[:,0]))

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5*chainlength), chainlength, 10)
    vs_ensemble = [ ensemble[ii,samples_to_plot] for ii in range(districts) ]
    vs_median = np.array([ np.percentile(vse, 50) for vse in vs_ensemble ])
    vs_lower  = np.array([ np.percentile(vse, 100*pc_thresh) for vse in vs_ensemble ])
    vs_upper  = np.array([ np.percentile(vse, 100*(1-pc_thresh)) for vse in vs_ensemble ])
    
    # identify stuffed, packed, and cracked districts
    inrange = np.array([ii for ii in range(districts) if vs_actual[ii] < vs_upper[ii] and vs_actual[ii] > vs_lower[ii] ], dtype=np.int32)
    packed  = np.array([ii for ii in range(districts) if vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5 ], dtype=np.int32)
    stuffed = np.array([ii for ii in range(districts) if vs_actual[ii] > vs_upper[ii] and vs_actual[ii] < .5 ], dtype=np.int32)
    cracked = np.array([ii for ii in range(districts) if vs_actual[ii] < vs_lower[ii] and vs_actual[ii] < .5 ], dtype=np.int32)


    # -----------------------------------------------

    
    # Create the plot
    myplot = plt.figure(figsize=(6.5,4.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6, quantiles=[[pc_thresh, 1-pc_thresh] for ii in seats])
    plt.plot(seats, vs_median, 'ko', markersize=2, label="Median of Ensemble")
    plt.scatter(1+inrange, vs_actual[inrange], s=5, color="black", label='Enacted: in Range')
    plt.scatter(1+packed,  vs_actual[packed] , s=8, color="red", label='Enacted: "Packed"')
    plt.scatter(1+cracked, vs_actual[cracked], s=8, color="blue", label='Enacted: "Cracked"')
    plt.scatter(1+stuffed, vs_actual[stuffed], s=8, color="orange", label='Enacted: "Stuffed"')
    plt.plot(seats, 0*seats+0.5, 'k--', lw=0.75, label="Needed to Win")

    plt.xlim(0, districts+1)
    plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)

    plt.xlabel('District Number')
    plt.ylabel('Democratic vote share')
    plt.title(title)
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    return myplot





def seats_votes_ensemble(ensemble, title, statewide=None, have_actual=True):
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
        actuals = np.array(sorted(ensemble[:,0], reverse=True))
        actual_seats = np.sum(actuals > .5)
        actual_votes = np.mean(actuals) if statewide == None else statewide
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
    avg_range = range(int(chainlength/2), chainlength)
    avg_seatsvotes = 0*np.array(sorted(ensemble[:,0], reverse=True))
    avg_seats = 0
    avg_votes = 0
    for step in avg_range:
        tmp_race_results = np.array(sorted(ensemble[:,step], reverse=True))
        tmp_mean  = np.mean
        tmp_seats = np.sum(tmp_race_results > .5)
        tmp_votes = np.mean(tmp_race_results) if statewide == None else statewide
        tmp_seatsvotes  = [tmp_votes - r + 0.5 for r in tmp_race_results]

        avg_seatsvotes += tmp_seatsvotes
        avg_seats += tmp_seats
        avg_votes += tmp_votes
        
    avg_seatsvotes /= len(avg_range)
    avg_votes /= len(avg_range)
    avg_seats /= len(avg_range)

    # Convert to arrays, reflect seats-votes curve about (.5, .5)
    avg_seatsvotes1 = np.array(avg_seatsvotes)
    avg_seatsvotes2 = np.flip(1-avg_seatsvotes1)


    #     plot them together in the same figure
    # -------------------------------------------------
    
    seats = range(1,districts+1)
    
    myfigure = plt.figure(figsize=(6.5,5))

    plt.subplot(211)
    plt.title(title)

    # panel 1: the ensemble
    # -------------------
    plt.plot(seats, avg_seatsvotes1, 'b',lw=2,label="Democrats")
    plt.plot(seats, avg_seatsvotes2, 'r',lw=2,label="Republicans")
    plt.fill_between(seats, avg_seatsvotes1, avg_seatsvotes2, where=(avg_seatsvotes1>avg_seatsvotes2), interpolate=True, **{'alpha':.2})

    # labeling
    hbuff = 0.25
    vbuff = 0.01
    dpos  = [ii for ii in range(districts) if avg_seatsvotes1[ii] > 0.5]
    first = dpos[0]
    start = avg_seatsvotes1[first-1]
    final = avg_seatsvotes1[first]
    slope = final - start
    hmin  = 1 + (first-1) + (0.5 - start) / slope
    hmid = 1 + (districts - 1) / 2.0
    hmax = hmin + 2*(hmid-hmin)
    vmax = avg_seatsvotes1[round(hmid)-1] if districts % 2 != 0 else 0.5*(avg_seatsvotes1[round(hmid-0.5)-1]+avg_seatsvotes1[round(hmid+0.5)-1])
    vmid = 0.5
    vmin = vmax - 2*(vmax-vmid)
    plt.plot([hmin-hbuff, hmax+hbuff],[vmid, vmid],'--', lw=2, color="green",label="Equal Voteshares")
    plt.plot([hmid, hmid],[vmin-vbuff, vmax+vbuff],'--', lw=2, color="gray", label="Majority Seat")
    
    # Should "actual" result be the location on the curve at the actual statewide percentage?
    # vs. the 
    plt.plot([avg_seats+0.5], [avg_votes], 'bs', lw=2, label="Actual Dem. Result")
    plt.plot([districts-avg_seats+0.5], [1-avg_votes], 'rs', label="Actual Rep. Result")

             
    print("Vote Needed for Majority (D-R) -- Ensemble:   %4.4f" % (vmax-vmin))
    print("Seats at 50%% Voteshare   (D-R) -- Ensemble:     %4d" % (hmax-hmin))

    # cleaning up
    plt.xlim(0, districts+1)
    plt.ylim(0.25,0.75)
    plt.text(hmid, .7, "Average of \n Sampled Plans", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
    plt.ylabel("Statewide Vote Share")
    #plt.xlabel("Number of Seats")
    plt.legend(**{'fontsize':8})

    if (have_actual):
        # panel 2
        # -------------------
        plt.subplot(212)
    
        plt.title(title)
        
        plt.plot(seats, seatsvotes1, 'b', lw=2,label="Democrats")
        plt.plot(seats, seatsvotes2, 'r', lw=2,label="Republicans")
        plt.fill_between(seats, seatsvotes1, seatsvotes2, where=(seatsvotes1>seatsvotes2), interpolate=True, **{'alpha':.2})

        # labeling
        dpos  = [ii for ii in range(districts) if seatsvotes1[ii] > 0.5]
        first = dpos[0]
        start = seatsvotes1[first-1]
        final = seatsvotes1[first]
        slope = final - start
        
        hmin  = 1 + (first-1) + (0.5 - start) / slope
        hmid = 1 + (districts - 1) / 2.0
        hmax = hmin + 2*(hmid-hmin)
    
        vmax = seatsvotes1[round(hmid)-1] if districts % 2 != 0 else 0.5*(seatsvotes1[round(hmid-0.5)-1]+seatsvotes1[round(hmid+0.5)-1])
        vmid = 0.5
        vmin = vmax - 2*(vmax-vmid)

        plt.plot([hmin-hbuff, hmax+hbuff],[vmid, vmid],'--', lw=2, color="green",label="Equal Voteshares")
        plt.plot([hmid, hmid],[vmin-vbuff, vmax+vbuff],'--', lw=2, color="gray", label="Majority Seat")
        plt.plot([actual_seats+0.5], [actual_votes], 'bs', lw=2, label="Actual Dem. Result")
        plt.plot([districts-actual_seats+0.5], [1-actual_votes], 'rs', label="Actual Rep. Result")

        print("Vote Needed for Majority (D-R) -- Actual:     %4.4f" % (vmax-vmin))
        print("Seats at 50%% Voteshare   (D-R) -- Actual:       %4d" % (hmax-hmin))

        # cleaning up
        plt.xlim(0, districts+1)
        plt.ylim(0.25,0.75)
        plt.ylabel("Percent Vote Share")
        plt.xlabel("Number of Seats")
        plt.text(hmid, .7, "Enacted Plan", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
        plt.legend(**{'fontsize':8})
        plt.tight_layout()
        
    return myfigure





def mean_median_partisan_bias(ensemble, statewide=None, have_actual=True):
    

    # get overall voteshare for party A
    overall_result = 0
    if statewide != None:
        overall_result = statewide
    else:
        overall_result = np.mean(np.mean(ensemble))
    
    # get shape of data
    districts, chainlength = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0
    
    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats = np.zeros((2, chainlength))
    for j in range(0,chainlength):
        race_results = sorted(ensemble[:,j], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1-seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0,j] = seatsvotes1[round(hmid)-1] if districts % 2 != 0 else 0.5*(seatsvotes1[round(hmid-0.5)-1] + seatsvotes1[round(hmid-0.5)-1])
        majority_vs[1,j] = seatsvotes2[round(hmid)-1] if districts % 2 != 0 else 0.5*(seatsvotes2[round(hmid-0.5)-1] + seatsvotes2[round(hmid-0.5)-1])
        
        # What is the number of seats you get at 50%?
        number_seats[0,j] = np.sum(seatsvotes1 <= 0.5)
        number_seats[1,j] = np.sum(seatsvotes2 <= 0.5)

    # things we need to plot
    vdiffs = majority_vs[0,:] - majority_vs[1,:]
    mvdiff = np.median(vdiffs)
    cvdiff = vdiffs[0]
    
    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max+.01, .01)
    vdbuff = (vdmax - vdmin)*.025
    
    vhist, vedges = np.histogram(vdiffs, bins=vbinbounds)
    if have_actual:
        print("MM Enacted Plan Percentile = ", stats.percentileofscore(vdiffs, cvdiff))
    
    
    
    # things we need to plot
    ndiffs = number_seats[0,:] - number_seats[1,:]
    mndiff = np.median(ndiffs)
    cndiff = ndiffs[0] 
    
    # for making it look nice
    ndmin = min(ndiffs)
    ndmax = max(ndiffs)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min-0.5, nbinedge_max+0.5)
    ndbuff = (ndmax - ndmin)*.01
    
    nhist, nedges = np.histogram(ndiffs, bins=nbinbounds)
    if have_actual:
        print("PB Enacted Plan Percentile = ", stats.percentileofscore(ndiffs, cndiff))

    
    
    
    myfigure = plt.figure(figsize=(6.5, 3.5))
    plt.subplot(121)
    
    
    plt.hist(vdiffs, bins=vbinbounds, color="xkcd:dark blue", **{'alpha':0.3})
    #sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mvdiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    if have_actual:
        plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Enacted Districts")
    plt.xlim(vdmin-vdbuff, vdmax+vdbuff)
    
    plt.ylim(0, np.max(vhist)*1.4)
    plt.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        right=False,       # ticks along the top edge are off
        labelleft=False)   # labels along the bottom edge are off
    plt.xlabel("Voteshare Difference for Majority (D-R)")
    plt.ylabel("Relative Frequency")
    plt.title('"Mean-Median" Score')
    plt.legend(loc="upper center")

    
    
    plt.subplot(122)
    plt.hist(ndiffs, bins=nbinbounds, color="xkcd:dark blue", **{'alpha':0.3})
    #sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mndiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    if (have_actual):
        plt.axvline(x=cndiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Enacted Districts")
    plt.xlim(ndmin-ndbuff, ndmax+ndbuff)
    plt.ylim(0, np.max(nhist)*1.4)
    plt.xticks(np.arange(nbinedge_min, nbinedge_max, 2))
    plt.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        right=False,       # ticks along the top edge are off
        labelleft=False)   # labels along the bottom edge are off
    plt.title('"Partisan Bias" Score')    
    plt.xlabel("Seat Difference at Equal Votes (D-R)")
    #plt.ylabel("Frequency")
    plt.legend(loc="upper center")
    
    
    plt.tight_layout()
    return myfigure