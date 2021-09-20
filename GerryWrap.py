import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats








def vote_vector_ensemble(ensemble, instance=None, title="Vote Vector", pc_thresh=0.01):
    
    # make sure it is sorted
    vs_actual = sorted(instance)
    
    # for making things look nice
    vbuffer = .02
        
    # get shape of data
    chainlength, districts = np.shape(ensemble)
    
    # list of integers (for plotting)
    seats = np.arange(districts)+1

    # collect distributions of results across ensemble
    samples_to_plot = range(int(0.5*chainlength), chainlength, 10)
    vs_ensemble = [ list(ensemble[samples_to_plot,ii]) for ii in range(districts) ]
    vs_lower  = [ np.percentile(vse, 100*pc_thresh) for vse in vs_ensemble ]
    vs_median = [ np.percentile(vse, 50) for vse in vs_ensemble ]
    vs_upper  = [ np.percentile(vse, 100*(1-pc_thresh)) for vse in vs_ensemble ]
    
    # identify stuffed, packed, and cracked districts
    dmin = 0
    dmax = districts
    stuffed = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] < .5) ]
    #if len(stuffed) > 0:  dmin = stuffed[-1]
    cracked = [ii for ii in range(dmin,dmax) if (vs_actual[ii] < vs_lower[ii] and vs_actual[ii] < .5 and vs_actual[ii] > .3) ]
    #if len(cracked) > 0:  dmin = cracked[-1]
    packed  = [ii for ii in range(dmin,dmax) if (vs_actual[ii] > vs_upper[ii] and vs_actual[ii] > .5) ]


    # -----------------------------------------------

    
    # Create the plot
    myplot = plt.figure(figsize=(6.5, 3.5))

    plt.violinplot(vs_ensemble, seats, showextrema=False, widths=0.6, quantiles=[[pc_thresh, 1-pc_thresh] for ii in seats])
    plt.plot(seats, vs_median, 'bo', markersize=2, label="Median of Ensemble")
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
    plt.ylim(vs_actual[0] - vbuffer, vs_actual[-1] + vbuffer)

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





def seats_votes_ensemble(ensemble, instance, title="Seats/Votes Curve", statewide=None):
    
    # get shape of data
    chainlength, districts = np.shape(ensemble)
    
    # get the actual outcomes
    actuals = np.array(sorted(instance, reverse=True))
    actual_seats = np.sum(actuals > .5)
    actual_votes = np.mean(actuals) if statewide == None else statewide
    
    # seats / votes curve, assuming uniform swing
    seatsvotes1 = actual_votes - actuals + 0.5
    
    # apply reflection about (.5, .5)
    seatsvotes2 = np.flip(1 - seatsvotes1)
    

    # now compute an average seats/votes over many samples
    # ------------------------------------------------
    avg_range = range(int(chainlength/2), chainlength)
    avg_seatsvotes = 0*seatsvotes1
    avg_seats = 0
    avg_votes = 0
    for step in avg_range:
        tmp_race_results = np.array(sorted(ensemble[step,:], reverse=True))
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


    # panel 1
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

    # panel 2
    # -------------------
    plt.subplot(212)
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
    #plt.text(hmid, .7, "Enacted Plan", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
    plt.text(hmid, .7, "Proposed Plan", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
    plt.legend(**{'fontsize':8})
    plt.tight_layout()
    return myfigure




def votes_seats_ensemble(ensemble, instance, title="Votes/Seats Curve", statewide=None):
    
    # get shape of data
    chainlength, districts = np.shape(ensemble)
    
    # get the actual outcomes
    actuals = np.array(sorted(instance, reverse=True))
    actual_seats = np.sum(actuals > .5)
    actual_votes = np.mean(actuals) if statewide == None else statewide
    
    # seats / votes curve, assuming uniform swing
    seatsvotes1 = actual_votes - actuals + 0.5
    
    # apply reflection about (.5, .5)
    seatsvotes2 = np.flip(1 - seatsvotes1)
    

    # now compute an average seats/votes over many samples
    # ------------------------------------------------
    avg_range = range(int(chainlength/2), chainlength)
    avg_seatsvotes = 0*seatsvotes1
    avg_seats = 0
    avg_votes = 0
    for step in avg_range:
        tmp_race_results = np.array(sorted(ensemble[step,:], reverse=True))
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


    # panel 1
    # -------------------
    plt.plot(avg_seatsvotes1, seats, 'b',lw=2,label="Democrats")
    plt.plot(avg_seatsvotes2, seats, 'r',lw=2,label="Republicans")
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
    
    plt.plot([vmid, vmid],[hmin, hmax],'--', lw=2, color="green",label="Equal Voteshares")
    plt.plot([vmin, vmax],[hmid, hmid],'--', lw=2, color="gray", label="Majority Seat")
    plt.plot([avg_votes], [avg_seats+0.5], 'bs', lw=2, label="Actual Dem. Result")
    plt.plot([1-avg_votes], [districts-avg_seats+0.5], 'rs', label="Actual Rep. Result")

    
             
    print("Vote Needed for Majority (D-R) -- Ensemble:   %4.4f" % (vmax-vmin))
    print("Seats at 50%% Voteshare   (D-R) -- Ensemble:     %4d" % (hmax-hmin))

    # cleaning up
    plt.ylim(0, districts+1)
    plt.xlim(0.25,0.75)
    #plt.text(hmid, .7, "Average of \n Sampled Plans", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
    plt.ylabel("Number of Seats")
    #plt.xlabel("Number of Seats")
    plt.legend(**{'fontsize':8})

    # panel 2
    # -------------------
    plt.subplot(212)
    plt.plot(seatsvotes1, seats, 'b', lw=2,label="Democrats")
    plt.plot(seatsvotes2, seats, 'r', lw=2,label="Republicans")
    
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

    plt.plot([vmid, vmid],[hmin, hmax],'--', lw=2, color="green",label="Equal Voteshares")
    plt.plot([vmin, vmax],[hmid, hmid],'--', lw=2, color="gray", label="Majority Seat")
    plt.plot([actual_votes], [actual_seats+0.5], 'bs', lw=2, label="Actual Dem. Result")
    plt.plot([1-actual_votes], [districts-actual_seats+0.5], 'rs', label="Actual Rep. Result")

    print("Vote Needed for Majority (D-R) -- Actual:     %4.4f" % (vmax-vmin))
    print("Seats at 50%% Voteshare   (D-R) -- Actual:       %4d" % (hmax-hmin))

    
    # cleaning up
    plt.ylim(0, districts+1)
    plt.xlim(0.25,0.75)
    plt.xlabel("Percent Vote Share")
    plt.ylabel("Number of Seats")
    #plt.text(hmid, .7, "Enacted Plan", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})
    
    #plt.text(hmid, .7, "Proposed Plan", **{'va':'top', 'ha':'center', 'size':10, 'weight':'bold'})

    plt.legend(**{'fontsize':8})
    plt.tight_layout()
    return myfigure







def partisan_metrics_histpair(ensemble, instance, statewide=None):
    

    # get overall voteshare for party A
    overall_result = 0
    if statewide != None:
        overall_result = statewide
    else:
        overall_result = np.mean(np.mean(ensemble))
    

    
    # get shape of data
    chainlength, districts = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0

    # logic for odd or even numbers of districts
    odd_number_of_districts = (districts % 2 != 0)
    odd_midindex = round(hmid)-1
    evl_midindex = round(hmid-0.5)-1
    evr_midindex = round(hmid+0.5)-1
    

    # information for the instance
    actual_race_results = sorted(instance, reverse=True)
    actual_mean_voteshare = np.mean(actual_race_results)
    actual_seatsvotes1 = actual_mean_voteshare - np.array(actual_race_results) + 0.5
    actual_seatsvotes2 = np.flip(1 - actual_seatsvotes1)

    # What is the percentage nec. for a majority?
    actual_majority_vs1 = actual_seatsvotes1[round(hmid)-1] if districts % 2 != 0 else 0.5*(actual_seatsvotes1[round(hmid-0.5)-1] + actual_seatsvotes1[round(hmid-0.5)-1])
    actual_majority_vs2 = actual_seatsvotes2[round(hmid)-1] if districts % 2 != 0 else 0.5*(actual_seatsvotes2[round(hmid-0.5)-1] + actual_seatsvotes2[round(hmid-0.5)-1])
    cvdiff = actual_majority_vs1 - actual_majority_vs2

    # What is the percentage nec. for a majority?
    actual_number_seats1 = np.sum(actual_seatsvotes1 <= 0.5)
    actual_number_seats2 = np.sum(actual_seatsvotes2 <= 0.5)
    cndiff = actual_number_seats1 - actual_number_seats2

   
    
    
    
    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats = np.zeros((2, chainlength))
    
    # seats votes curve for the ensemble
    for j in range(0,chainlength):
        race_results = sorted(ensemble[j,:], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1 - seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0,j] = seatsvotes1[odd_midindex] if districts % 2 != 0 else 0.5*(seatsvotes1[evl_midindex] + seatsvotes1[evr_midindex])
        majority_vs[1,j] = seatsvotes2[odd_midindex] if districts % 2 != 0 else 0.5*(seatsvotes2[evl_midindex] + seatsvotes2[evr_midindex])
        
        # What is the number of seats arising from a split vote?
        lastseat1 = np.sum(seatsvotes1 <= 0.5)-1
        v0 = seatsvotes1[lastseat1]
        v1 = seatsvotes1[lastseat1+1]
        number_seats[0,j] = lastseat1 + 1 + (0.5 - v0) / (v1 - v0)
        
        lastseat2 = np.sum(seatsvotes2 <= 0.5)-1
        v0 = seatsvotes2[lastseat2]
        v1 = seatsvotes2[lastseat2+1]
        number_seats[1,j] = lastseat2 + 1 + (0.5 - v0) / (v1 - v0)

        


    #
    #   Voteshares
    #
    # things we need to plot
    vdiffs = majority_vs[0,:] - majority_vs[1,:]
    mvdiff = np.median(vdiffs)
        
    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max+.01, .01)
    
    vdbuff = (vdmax - vdmin)*.2
    
    vhist, vedges = np.histogram(vdiffs, bins=vbinbounds)
    print("MM Enacted Plan Percentile = %8.6f" % stats.percentileofscore(vdiffs, cvdiff))
    
    
    #
    #   Number of Seats
    #    
    # things we need to plot
    ndiffs = number_seats[0,:] - number_seats[1,:]
    mndiff = np.median(ndiffs)

    
    # for making it look nice
    ndmin = min(ndiffs)
    ndmax = max(ndiffs)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min-0.5, nbinedge_max+0.5)
    ndbuff = (ndmax - ndmin)*.2
    
    nhist, nedges = np.histogram(ndiffs, bins=nbinbounds)
    print("PB Enacted Plan Percentile = %8.6f" % stats.percentileofscore(ndiffs, cndiff))

    

    # ----------------------------------------
    #   Show the 1D distributions separately
    # ----------------------------------------
    myfigure = plt.figure(figsize=(6.5, 3.5))
    plt.subplot(121)
    
    
    plt.hist(vdiffs, bins=vbinbounds, color="xkcd:dark blue", **{'alpha':0.3})
    #sns.histplot(vdiffs, kde=True, bins=binbounds, color="xkcd:dark blue")
    plt.axvline(x=mvdiff, color="green", ls='--', lw=2.5, ymax=0.75, label="Ensemble Median")
    #plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Current Districts")
    plt.axvline(x=cvdiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Proposed Districts")
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
    #plt.axvline(x=cndiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Current Districts")
    plt.axvline(x=cndiff, color="purple", ls='--', lw=2.5, ymax=0.75, label="Proposed Districts")
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


    

    
def partisan_metrics_hist2D(ensemble, instance, statewide=None):
    

    # get overall voteshare for party A
    overall_result = 0
    if statewide != None:
        overall_result = statewide
    else:
        overall_result = np.mean(np.mean(ensemble))
    

    
    # get shape of data
    chainlength, districts = np.shape(ensemble)
    hmid = 1 + (districts - 1) / 2.0

    # logic for odd or even numbers of districts
    odd_number_of_districts = (districts % 2 != 0)
    odd_midindex = round(hmid)-1
    evl_midindex = round(hmid-0.5)-1
    evr_midindex = round(hmid+0.5)-1
    

    
    
    # information for the instance
    actual_race_results = sorted(instance, reverse=True)
    actual_mean_voteshare = np.mean(actual_race_results)
    actual_seatsvotes1 = actual_mean_voteshare - np.array(actual_race_results) + 0.5
    actual_seatsvotes2 = np.flip(1 - actual_seatsvotes1)

    # What is the percentage nec. for a majority?
    actual_majority_vs1 = actual_seatsvotes1[round(hmid)-1] if districts % 2 != 0 else 0.5*(actual_seatsvotes1[round(hmid-0.5)-1] + actual_seatsvotes1[round(hmid-0.5)-1])
    actual_majority_vs2 = actual_seatsvotes2[round(hmid)-1] if districts % 2 != 0 else 0.5*(actual_seatsvotes2[round(hmid-0.5)-1] + actual_seatsvotes2[round(hmid-0.5)-1])
    cvdiff = actual_majority_vs1 - actual_majority_vs2

    # What is the percentage nec. for a majority?
    actual_number_seats1 = np.sum(actual_seatsvotes1 <= 0.5)
    actual_number_seats2 = np.sum(actual_seatsvotes2 <= 0.5)
    cndiff = actual_number_seats1 - actual_number_seats2

   
    
    
    
    # space allocation
    majority_vs = np.zeros((2, chainlength))
    number_seats = np.zeros((2, chainlength))
    
    # seats votes curve for the ensemble
    for j in range(0,chainlength):
        race_results = sorted(ensemble[j,:], reverse=True)
        mean_voteshare = np.mean(race_results)
        seatsvotes1 = mean_voteshare - np.array(race_results) + 0.5
        seatsvotes2 = np.flip(1 - seatsvotes1)

        # What is the percentage nec. for a majority?
        majority_vs[0,j] = seatsvotes1[odd_midindex] if districts % 2 != 0 else 0.5*(seatsvotes1[evl_midindex] + seatsvotes1[evr_midindex])
        majority_vs[1,j] = seatsvotes2[odd_midindex] if districts % 2 != 0 else 0.5*(seatsvotes2[evl_midindex] + seatsvotes2[evr_midindex])
        
        # What is the number of seats arising from a split vote?
        lastseat1 = np.sum(seatsvotes1 <= 0.5)-1
        v0 = seatsvotes1[lastseat1]
        v1 = seatsvotes1[lastseat1+1]
        number_seats[0,j] = lastseat1 + 1 + (0.5 - v0) / (v1 - v0)
        
        lastseat2 = np.sum(seatsvotes2 <= 0.5)-1
        v0 = seatsvotes2[lastseat2]
        v1 = seatsvotes2[lastseat2+1]
        number_seats[1,j] = lastseat2 + 1 + (0.5 - v0) / (v1 - v0)

        


    #
    #   Voteshares
    #
    # things we need to plot
    vdiffs = majority_vs[0,:] - majority_vs[1,:]
    mvdiff = np.median(vdiffs)
        
    # for making it look nice
    vdmin = min(vdiffs)
    vdmax = max(vdiffs)
    vbinedge_min = np.floor(vdmin)
    vbinedge_max = np.ceil(vdmax)
    vbinbounds = np.arange(vbinedge_min, vbinedge_max+.01, .01)
    
    vdbuff = (vdmax - vdmin)*.2
    
    vhist, vedges = np.histogram(vdiffs, bins=vbinbounds)
    print("MM Enacted Plan Percentile = %8.6f" % stats.percentileofscore(vdiffs, cvdiff))
    
    
    #
    #   Number of Seats
    #    
    # things we need to plot
    ndiffs = number_seats[0,:] - number_seats[1,:]
    mndiff = np.median(ndiffs)

    
    # for making it look nice
    ndmin = min(ndiffs)
    ndmax = max(ndiffs)
    nbinedge_min = np.floor(ndmin)
    nbinedge_max = np.ceil(ndmax)
    nbinbounds = np.arange(nbinedge_min-0.5, nbinedge_max+0.5)
    ndbuff = (ndmax - ndmin)*.2
    
    nhist, nedges = np.histogram(ndiffs, bins=nbinbounds)
    print("PB Enacted Plan Percentile = %8.6f" % stats.percentileofscore(ndiffs, cndiff))

    

    # 2D histogram
    
    myfigure = plt.figure(figsize=(6.5, 5))
    # Basic 2D density plot
    import seaborn as sns
    sns.set_style("white")
    sns.kdeplot(x=vdiffs, y=ndiffs, cmap="Blues", shade=True, cbar=True, levels=[.0001, .001, .01, .05, .25, .5, .75, 1.0]) #, bw_adjust=.5
    #plt.plot([cvdiff], [cndiff], 'rs', label="Current Plan")
    plt.plot([cvdiff], [cndiff], 'rs', label="Proposed Plan")
    plt.xlabel("Mean-Median")
    plt.ylabel("Partisan Bias")
    plt.title("2D Histogram of Ensemble Partisan Metrics")
    plt.legend(loc='best')
    plt.tight_layout()
       
    
    return myfigure