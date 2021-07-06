#Expected command line call: MCMC_for_Texas_v2_4Bash Output_Directory Number_of_Runs Number_of_Skips Save_period
#
#Output_Directory : Name of directory in which to save output. Does not need to already exist.
#Number_of_Runs : Number of ensemble maps to be created.
#Number_of_Skips : If Number_of_Runs is a large number, then for sake of memory not all maps are saved. 
#Save_period : How often to save the the ensemble data during the run. 


# standard imports
import os
import sys
from functools import partial

# data-related imports
import csv
import pickle
import pandas as pd
import geopandas as gpd

# GerryChain imports
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain, proposals, metrics, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.random import random



#For reproducibility, a random seed is incorporated
random.seed(12345678)



# command line arguments
outdir = "./"+sys.argv[1]+"/"
nmaps  = int(sys.argv[2])
mod1   = int(sys.argv[3])
mod2   = int(sys.argv[4])




# --------------------------------
#    read shape files
# --------------------------------


#Read in precinct-shape files for Texas. Eventually, this should become a variable
df = gpd.read_file("TX_vtds_extra/TX_vtds_extra.shp")
df.geometry = df.geometry.buffer(0)

#Replace NAs with 0
for col in df.columns:
    df[col] = df[col].fillna(0)
    
#df['GOV18R']=df['GOV18R'].fillna(0)
#df['GOV18D']=df['GOV18D'].fillna(0)
#df['SEN18D']=df['SEN18D'].fillna(0)
#df['SEN18R']=df['SEN18R'].fillna(0)
#df['18TotalPop']=df['18TotalPop'].fillna(0)
#df['18TotalVR']=df['18TotalVR'].fillna(0)
#df['18TotalTO']=df['18TotalTO'].fillna(0)
#df['HISP18']=df['HISP18'].fillna(0)


# convert to a GerryChain Graph object
df.geometry = df.geometry.buffer(0)
graph = Graph.from_geodataframe(df) 



#Make directory to store data. 
#newdir = "./Outputs_Long_505/"										#Need to make this a variable
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")

    

#Labeling the Elections
elections = [
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("SEN12",  {"Democratic": "SEN12D",  "Republican": "SEN12R"}),
    Election("GOV14",  {"Democratic": "GOV14D",  "Republican": "GOV14R"}),
    Election("SEN14",  {"Democratic": "SEN14D",  "Republican": "SEN14R"}),
    Election("PRES16", {"Democratic": "PRES16D", "Republican": "PRES16R"}),
    Election("GOV18",  {"Democratic": "GOV18D",  "Republican": "GOV18R"}),
    Election("SEN18",  {"Democratic": "SEN18D",  "Republican": "SEN18R"})
]
num_elections = len(elections)



#Updater for MCMC
my_updaters = {"countysplits":  updaters.county_splits("countysplits", "COUNTY"),
               "population":    updaters.Tally("TOTPOP", alias="population"),
               "VAP":           updaters.Tally("VAP"),
               "WVAP":          updaters.Tally("WVAP"),
               "BVAP":          updaters.Tally("BVAP")
               "HVAP":          updaters.Tally("HISPVAP")
}
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)



#When running through all races, we use the Partitions variable
Partitions = {
    "TXSenate"   : GeographicPartition(graph, assignment='TXSN', updaters=my_updaters),
    "TXHouse"    : GeographicPartition(graph, assignment='TXHD', updaters=my_updaters),
    "USCongress" : GeographicPartition(graph, assignment='USCD', updaters=my_updaters)
}



# ----------------------------------------- 
#     define map ideals and constraints
# -----------------------------------------

# empty dictionaries for storing per-chamber map-drawing rules
Tolerance = {}
Proposals = {}
Compactness = {}
Splits = {}



#Proposals for each assignment (Senate, House, and Congress)
for chamber in Partitions:
    print(chamber)
    
    # calculate the ideal population per district
    ideal_population = sum(Partitions[chamber]["population"].values()) / len(Partitions[chamber])


    # define a maximum tolerated deviation from the ideal population
    p = []
    for x in Partitions[chamber]["population"]: 
        p.append(round(100*(1-Partitions[chamber]["population"][x]/ideal_population),2))
    max_deviation = round((max(abs(min(p)),abs(max(p))))/100, 3)
    print("Max deviation from ideal population: " + str(max_deviation) + "%")
    Tolerance[chamber] = max_deviation + .05  # add a five percent buffer?
    

    # define a constraint (?) on the population per district
    proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.01,
                   node_repeats=2
                  )
    Proposals[chamber] = proposal

    
    # define a constraint on the maximum number of cut edges
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2*len(Partitions[chamber]["cut_edges"])
    )
    Compactness[chamber] = compactness_bound

    
    # define a constraint on the maximum number of split counties
    split_bound = constraints.UpperBound(
        lambda p: len(p["countysplits"]),
        len(Partitions[chamber]["countysplits"])
    )
    Splits[chamber] = split_bound    


    
#The actual program
#print("About to enter the Markov Chain")
#for chamber in Partitions:
#chamber = "House"
chamber = "Senate"
chain = MarkovChain(
    proposal=Proposals[chamber],
    constraints=[
        # District populations must stay within 4% of equality
        constraints.within_percent_of_ideal_population(Partitions[chamber], Tolerance[chamber]),
        Compactness[chamber],
        Splits[chamber],
    ],
    accept=accept.always_accept,
    initial_state=Partitions[chamber],
    total_steps=nmaps
)



# -----------------------------------------------------
#     allocate storage for metrics of each map
# -----------------------------------------------------

WVAP_data = pandas.DataFrame([], range(0,nmaps), ["WVAP"])
BVAP_data = pandas.DataFrame([], range(0,nmaps), ["BVAP"])
HVAP_data = pandas.DataFrame([], range(0,nmaps), ["HVAP"])
MVAP_data = pandas.DataFrame([], range(0,nmaps), ["MVAP"])

pops_data = pandas.DataFrame([], range(0,nmaps), ["POPS"])
cuts_data = pandas.DataFrame([], range(0,nmaps), ["CUTS"])

VOTE_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])
wins_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])
egap_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])
mmed_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])
prbs_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])
gini_data = pandas.DataFrame([], range(0,nmaps), [election.name for election in elections])







# ----------------------
# main loop
# ----------------------

i=0
for partition in chain:#.with_progress_bar():    
    #plots and prints every 100th map, helpful also for tracking progress

    mod_rat = int(mod2/mod1)
    
    if(i%mod1==0):
        index = int(i/mod1)
        
        # metrics based on the geometry of the map
        pop_vec.append(sorted(list(partition["population"].values())))
        cut_vec.append(len(partition["cut_edges"]))

        # percentages of White, Black, Hispanic, and combined minority voters in each district
        WVAP_data.loc[index] = {partition["WVAP"][key]/partition["VAP"][key] for key in partition["population"] }
        BVAP_data.loc[index] = {partition["BVAP"][key]/partition["VAP"][key] for key in partition["population"] }
        HVAP_data.loc[index] = {partition["HVAP"][key]/partition["VAP"][key] for key in partition["population"] }
        MVAP_data.loc[index] = {(partition["VAP"][key] - partition["WVAP"][key])/partition["VAP"][key] for key in partition["population"]}
        
        # overall vote share -- must be calculated separately for each election
        VOTE_data.loc[index] = {election.name: sorted(partition[election.name].percents("Democratic")) for election in elections}
                
        # derived partisan metrics; also must be calculated for each election
        wins_data.loc[index] = [partition[election.name].wins("Democratic") for election in elections]
        egap_data.loc[index] = [metrics.efficiency_gap(partition[election.name]) for election in elections]
        mmed_data.loc[index] = [metrics.mean_median   (partition[election.name]) for election in elections]
        prbs_data.loc[index] = [metrics.partisan_bias (partition[election.name]) for election in elections]
        gini_data.loc[index] = [metrics.partisan_gini (partition[election.name]) for election in elections]

            
    #if(i%mod2==0):
    #    #print("i = "+str(i))
    #    partition.plot()
    #    print("i = "+str(i))
    #    plt.savefig(newdir+chamber+"_mapstep"+str(i)+".png")
    #    plt.close()
    #    if i==0:
    #        election_data[0:1].to_csv(newdir + "Election_Data_" + chamber + "_Running" + ".csv")
    #    else:
    #        upper_index = int((i/mod2)*mod_rat)+1
    #        lower_index = upper_index - mod_rat
    #        election_data[lower_index:upper_index].to_csv(newdir + "Election_Data_" + chamber + "_Running" + .csv",mode='a', header=False)
    
    i = i+1




# ----------------------------                                          
#    Write out results
# ----------------------------

outfile = 

pickle.dump()




#election results
election_data.to_csv(newdir + "Election_Data_" + chamber + "_Final" + ".csv")

#proportion non-white VAP in each district
NWVAP_data.to_csv(newdir + "NWVAP_Data_" + chamber + "_Final" + ".csv")

#calculate the proportion Hispanic VAP in each district
HVAP_data.to_csv(newdir + "HVAP_Data" + "_Final" + ".csv")

#calculate the proportion Black VAP in each district
BVAP_data.to_csv(newdir + "BVAP_Data" + "_Final" + ".csv")


# metrics   
with open(newdir + "gini_" + chamber + "_Final" + ".csv", "w") as tf1:
    writer = csv.writer(tf1, lineterminator="\n")
    writer.writerows(gini)                                                          
                                                          
with open(newdir + "mms_" + chamber + "_Final" + ".csv", "w") as tf1:
    writer = csv.writer(tf1, lineterminator="\n")
    writer.writerows(mms)

with open(newdir + "egs_" + chamber + "_Final" + ".csv", "w") as tf1:
    writer = csv.writer(tf1, lineterminator="\n")
    writer.writerows(egs)

with open(newdir + "hmss_" + chamber + "_Final" + ".csv", "w") as tf1:
    writer = csv.writer(tf1, lineterminator="\n")
    writer.writerows(hmss)

#Other data generated
with open(newdir + "pop_vec_" + chamber + "_Final" + ".csv", "w") as f:
    writer = csv.writer(f,lineterminator="\n")
    writer.writerows(pop_vec)

with open(newdir + "cut_vec_" + chamber + "_Final" + ".csv", "w") as f:
    writer = csv.writer(f,lineterminator="\n")
    writer.writerow(cut_vec)  


print("finished")    
