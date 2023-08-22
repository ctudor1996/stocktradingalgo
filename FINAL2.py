##########################################################################################################################
##########################################################################################################################
# Section 0 -Introduction
##########################################################################################################################
##########################################################################################################################

# Hi
# This is an attempt to create an AI that will determine the best time to buy / short a stock.
# It establishes degrees of freedom and variables to analyze stock price and volume historical data from yahoo finance.
#


#high level overview
#The AI will search through paramaters in a range in certain increments
#It will report and plot a graph of the range of Profit and Losses based on the range of parameters
#Thus, it will find some optimal parameter

#The next problem is: these optimal parameters only account for the whole period of stocks explored
#What if there are different optimal parameters for each different state of a stock.

#Thus, how to describe and find states of stocks.
#Economic fundamental data: recession, unemployment
#stock specific news (company, country, etc.)

#Could write a rule that every time there is sharp change in the state of a stock, you punctuate that period and
#search for new optimal parameters for the duration of that period.

######
#!!!!OUTPUT range of indexes to make the AI remember periods in a stock's history when there have been large historic events (and large associated indexes) so that it can:
#maybe partially account for black swan events at least with one degree of freedom!!!!!!!!
#####



###FUNDAMENTAL ECON VARIABLES:
    #Inflation
    #Employment Cost Index
    #Unemployment Rate
    #GDP
    #Purchasing Manager's index

##########################################################################################################################
##########################################################################################################################
# Section 1 - Imports
##########################################################################################################################
##########################################################################################################################


#imports
import pandas as pd
from IPython.display import display
import math
import numpy as np
import itertools
import statistics as stat
import matplotlib
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.signal import find_peaks
import csv

#display all columns in the dataframe
pd.set_option('display.max_columns', 1000)

######################## DATA and USER INPUTS ####################
##################################################################

stock_files = ['file:///C:/Users/Asus/Desktop/STOCK DATA TO MINE/MINE/BAC.csv'] 

df = pd.read_csv('file:///C:/Users/Asus/Desktop/STOCK DATA TO MINE/MINE/BAC.csv' )


##########################################################################################################################
##########################################################################################################################
# Section 2 - DEGREES OF FREEDOM / VARIABLES 
##########################################################################################################################
##########################################################################################################################


##########################################################################################################################
# Section 3.1 - Algorithm DoF / Vars
##########################################################################################################################


#alpha is the bias in the index calculation, just a base value that is the equivalent of x0 in a regression
alpha = 0

#the moving average factor
avrgfactor = 5

average_factor_sensitivity = 0.7171


#determines the number of past positive (or negative respectively) periods the current period is compared to
#! future plans to maybe make smart AI try to account for black swan events
lookback = 4


#Index Critical value past which to buy
rule = 1.1827


index_sensitivity_factor = 0.9350

#Adjusts how much the volume support influences the adjusted price change. Past days volume is weighed more than current day volume
#it is a power, should be between 0.1 and 0.9
VAD_influence_factor = 0.6854

#adjusts how much volume support influences the price period vector used in the index
#it is a power, should be between 0.1 and 0.9
vap_influence_factor = 0.501893916



lookback_influence_factor = 0.935001326



appv_influence_factor = 0.535001326

entire_stock_duration_positive_period_counter_sensitivity_factor = 5


##########################################################################################################################
# Section 3.2 - P&L/BuyRule 
##########################################################################################################################

sharenr = 100


####CHANGES TO BUY RULE

#nr of negative periods in current decline to activate the buy rule
nr_neg_p_to_buy = 4

#nr of positive periods in current rise to activate the buy rule
nr_pos_p_to_buy = 4

#T/F 1=the moving average change in price must also be in accordance with the decline/rise of the period for the buy rule to be activated, 0= not
on_off_avrgdelta_pnl = 1


#cp = 30


##########################################################################################################################
##########################################################################################################################
# Section 3 - Algorithm
##########################################################################################################################
##########################################################################################################################

total_infolist = []

def Algofunc(avrgfactor,
             average_factor_sensitivity,
             lookback,
             rule,
             index_sensitivity_factor,
             VAD_influence_factor,
             vap_influence_factor,
             lookback_influence_factor,
             appv_influence_factor,
             entire_stock_duration_positive_period_counter_sensitivity_factor,
             alpha,
             df):
    
    
    
    #vwap estimate for the day
    df['VWAP Estimate'] = (df['High']*0.25 + df['Close']*0.5 + df['Low']*0.25)
    
    p_list = df['VWAP Estimate'].tolist()
    pricedelta = []
    
        
    for i in range(1, len(p_list)):
        pricedelta.append(p_list[i]-p_list[i-1])
    
    pricedelta.insert(0,0)
    
    df['pricedelta'] = pd.Series(pricedelta)
    
    
    volumelist = df['Volume'].tolist()
    avrgdelta = []
    avrgvolume = []
    
    ######################################## average volume
    
    for i in range(avrgfactor, len(volumelist)+1):
        avrgvolume.append(sum(volumelist[i-avrgfactor:i])/avrgfactor)
    
    for i in range(avrgfactor-1):
        avrgvolume.insert(0,0)
        
    df['average volume'] = pd.Series(avrgvolume)
    
    for i in range(avrgfactor+1, len(pricedelta)+1):
        avrgdelta.append(sum(pricedelta[i-avrgfactor:i])/avrgfactor**average_factor_sensitivity)
    
    for i in range(avrgfactor):
        avrgdelta.insert(0,0)
    
    df['average delta'] = pd.Series(avrgdelta)
    
    ######################################### volume support
    df['VS']=df['Volume']/df['average volume']
   
    
    ######################################### volume adjusted average price delta
    
    vs_list_for_factor= []
    vs_list_for_factor = df['VS'].tolist()
    
    vs_list_adj = [num ** VAD_influence_factor for num in vs_list_for_factor]
    df['VS_adj']=pd.Series(vs_list_adj)
    
    df['VAD'] = (df['VS_adj'])*(df['average delta'])


    ############################################################################################
    ########################## complex operations ##############################################
    ############################################################################################
    
    
    
    ############### COUNTERS ###################
    ############################################
    
    ####################################### OVERALL positive and negative periods counters and their lists
    PP_counter = 0
    PP_countlist = []
    
    NP_counter = 0
    NP_countlist = []
    
    
    ###################################### within-period counters and their lists
    pos_counter = 0
    pos_countlist = []
    
    neg_counter = 0
    neg_countlist = []
    
    
    ###################################### general lists and variables
        
    d = df['VAD'].tolist()
        
    v = df['VS'].tolist()
    vap = 0
    vap_list=[]
    ppv = 0
    ppv_list = []
    appv = 0
    appv_list=[]
    index = 0
    index_list = []
    
    
    ######################## index and price-period vector calculation ########################
    ###########################################################################################
    
    for i in range(0, len(d)):
        if d[i] < 0:                                          #### negative periods calculations
            dc = abs(d[i])
            pos_counter = 0
            pos_countlist.append(0)
            neg_counter = neg_counter + 1
            neg_countlist.append(neg_counter)
            
            if neg_counter == 1:                               #### volume adjusted period calculation
                vap = v[i]
                vap_list.append(vap)
            elif neg_counter > 1:
                vap = v[i]+vap_list[i-1]
                vap_list.append(vap)
                
            ppv = vap**vap_influence_factor * dc                                    ##### price period vector calculation
            ppv_list.append(ppv)
            
            if neg_counter == 1:                             ###### average price period vector calculation
                appv = ppv_list[i]
                appv_list.append(appv)
                if pos_countlist[i-1] > entire_stock_duration_positive_period_counter_sensitivity_factor:
                    PP_counter = PP_counter+1
                    PP_countlist.append(appv_list[i-1])
            elif neg_counter > 1:
                appv = sum(ppv_list[i - neg_counter + 1:i+1]) / (neg_counter)
                appv_list.append(appv)
            
            
            
            ############# first instance of the index calculation
            
            if (NP_counter >= 1) and (lookback <= 1 or NP_counter<=lookback):                               
                index = alpha + abs(appv**appv_influence_factor / (NP_countlist[NP_counter-1]**lookback_influence_factor))**index_sensitivity_factor
                index_list.append(index)
            elif (lookback > 1)  and (NP_counter > lookback):
                index = alpha + abs(appv**appv_influence_factor / ((sum(NP_countlist[NP_counter-lookback:NP_counter])/lookback)**lookback_influence_factor))**index_sensitivity_factor
                index_list.append(index)
                
        elif d[i] > 0:                                        ################ positive periods calculations
            dc = d[i]
            neg_counter = 0
            neg_countlist.append(0)
            pos_counter = pos_counter + 1
            pos_countlist.append(pos_counter)
            
            if pos_counter == 1:
                vap = v[i]
                vap_list.append(vap)
            elif pos_counter > 1:
                vap = v[i]+vap_list[i-1]
                vap_list.append(vap)
                
            ppv = vap * dc
            ppv_list.append(ppv)
            
            if pos_counter == 1:
                appv = ppv_list[i]
                appv_list.append(appv)
                if neg_countlist[i-1] > entire_stock_duration_positive_period_counter_sensitivity_factor:
                    NP_counter = NP_counter+1
                    NP_countlist.append(appv_list[i-1])
            elif pos_counter > 1:
                appv = sum(ppv_list[i - pos_counter+1:i+1]) / (pos_counter)
                appv_list.append(appv)
            
            if PP_counter > 1 and (lookback <= 1 or PP_counter<=lookback):
                index = alpha + abs(appv**appv_influence_factor / (PP_countlist[PP_counter-1]**lookback_influence_factor))**index_sensitivity_factor
                index_list.append(index)
            elif (lookback > 1) and (PP_counter > lookback):
                index = alpha + abs((appv**appv_influence_factor) / ((sum(PP_countlist[PP_counter-lookback:PP_counter])/(lookback))**lookback_influence_factor))**index_sensitivity_factor
                index_list.append(index)
    
        else:                                                   ################# fringe cases
            pos_countlist.append(0)
            neg_countlist.append(0)
            vap_list.append(0)
            ppv_list.append(0)
            appv_list.append(0)
    
    for i in range(len(d)-len(index_list)):
        index_list.insert(0,0)
    
    df['index'] = index_list
    
    return(vap_list, 
           ppv_list, 
           appv_list, 
           index_list, 
           pos_countlist, 
           neg_countlist, 
           avrgdelta, 
           p_list,
           neg_countlist,
           pos_countlist)

        





##########################################################################################################################

##########################################################################################################################

##########################################################################################################################
##########################################################################################################################
# Section 4 - P&L Function / Buy Rule Algorithm
##########################################################################################################################
##########################################################################################################################





def pnlfunc(sharenr,
            nr_neg_p_to_buy,
            on_off_avrgdelta_pnl,
            nr_pos_p_to_buy):
    
    #Algo Outut from invoking algofunc
    algo_output = Algofunc(avrgfactor,
                 average_factor_sensitivity,
                 lookback,
                 rule,
                 index_sensitivity_factor,
                 VAD_influence_factor,
                 vap_influence_factor,
                 lookback_influence_factor,
                 appv_influence_factor,
                 entire_stock_duration_positive_period_counter_sensitivity_factor,
                 alpha,
                 df)


    #vap_list, ppv_list, appv_list, index_list, pos_countlist, neg_countlist, avrgdelta, p_list

    vap_list = algo_output[0]
    ppv_list = algo_output[1]
    appv_list = algo_output[2]
    index_list = algo_output[3]
    pos_countlist = algo_output[4]
    neg_countlist = algo_output[5]
    avrgdelta = algo_output[6]
    p_list = algo_output[7]
    neg_countlist = algo_output[8]
    pos_countlist = algo_output[9]
    
    #establishing new lists to track pnl values    
    pnlcp = 0
    pnlcp_list = []
    runningtotal = 0
    rt_list = []
    sharecount = 0
    sc_list = []
    closed_position_list = []
    traded = 0
    pnl_per_trade_list = []


       

    average_ROI = 0
    ROI_list = []                            ## average return on investment across time


    for i in range(0, len(index_list)):
        
        if pos_countlist[i] + neg_countlist[i] == 1:
            traded = 0
         
        #BUY if index over indexrule and the price is going down
        if index_list[i] > rule and traded == 0 and neg_countlist[i] > nr_neg_p_to_buy and (on_off_avrgdelta_pnl*(avrgdelta[i]))<0:
            traded = 1
            pnlcp = p_list[i]
            pnlcp_list.append(pnlcp)
            #if you buy or sell you sell your entire share holding
            if sharecount == 0:
                runningtotal = runningtotal + pnlcp * sharenr
                sharecount = sharecount + sharenr
            else:
                runningtotal = runningtotal + pnlcp * sharecount
                sharecount = 0    
            rt_list.append(runningtotal)
            sc_list.append(sharecount)
            
        #SELL/SHORT if index over indexrule and the price is going up
        elif index_list[i] > rule and traded == 0 and pos_countlist[i] > nr_pos_p_to_buy and (on_off_avrgdelta_pnl*(-avrgdelta[i]))<0:
            traded = 1
            pnlcp = p_list[i]
            pnlcp_list.append(pnlcp)
            #if you buy or sell you sell your entire share holding 
            if sharecount == 0:
                runningtotal = runningtotal - pnlcp * sharenr
                sharecount = sharecount + sharenr
            else:
                runningtotal = runningtotal - pnlcp * sharecount
                sharecount = 0                
            rt_list.append(runningtotal)
            sc_list.append(sharecount)
                
    
    for i in range(0, len(sc_list)):
        if sc_list[i] == 0:
            closed_position_list.append(rt_list[i])


    #list of value gained or value lost in each trade
    for i in range(1,len(closed_position_list)):    
        pnl_per_trade_list.append(closed_position_list[i]-closed_position_list[i-1])

    return(pnlcp_list, 
           rt_list, 
           sc_list, 
           closed_position_list, 
           sharecount, 
           runningtotal,
           pnl_per_trade_list)

    




##########################################################################################################################
##########################################################################################################################
# Section FINAL - Running the AI
##########################################################################################################################
##########################################################################################################################

#pnl_output = pnlfunc(sharenr,
#                     nr_neg_p_to_buy,
#                     on_off_avrgdelta_pnl,
#                     nr_pos_p_to_buy)


iteration=0
iteration_list=[]
iteration_list.insert(0,0)

#make a list and matrix of ALL combinations for ALL files

total_tested_param_matrix = []
total_optimal_param_matrix = []

total_closing_profit_list = []
total_nr_trades_list = []
total_stock_ID_list =[]
total_local_max_explored_list =[]

#as the algo goes through csv files, this list keeps track of the algo combinations that yielded the highest returns
tested_param_matrix_combolist=[]


#KEEP TRACK AT WHICH ITERATION THE CSV FILES SWAP
csv_swap_tracker_iterationnr = []
csv_swap_tracker_iterationnr.insert(0,0)



#LOOP
# Specify the path of the folder containing the CSV files
folder_path = "C:/Users/Asus/Desktop/STOCK DATA TO MINE/MINE"

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]


# Iterate through the list of CSV files and read each file into a Pandas DataFrame
file_counter = 0
file_countlist = []



for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print('Currently working on file:')
    print(file)
    file_counter = file_counter + 1
    file_countlist.append(file_counter)
    df = pd.read_csv(file_path)
    tested_param_matrix = []
    closing_profit_list = []
    nr_trades_list = []
    local_index_list = []
    param_combo_local_maxima = []
    
    # Do something with the DataFrame
    for avrgfactor in range (3,6,1):
        for average_factor_sensitivity in range (60,71,10):
            average_factor_sensitivity = average_factor_sensitivity/100
            for lookback in range(3,5,1):
                for rule in range (80,141,20):
                    rule = rule/100
                    for index_sensitivity_factor in range (90,91,10):
                        index_sensitivity_factor = index_sensitivity_factor/100
                        for VAD_influence_factor in range(60,71,10):
                            VAD_influence_factor = VAD_influence_factor/100
                            for vap_influence_factor in range(40,51,10):
                                vap_influence_factor = vap_influence_factor/100
                                for lookback_influence_factor in range (90,91,10):
                                    lookback_influence_factor = lookback_influence_factor/100
                                    for appv_influence_factor in range(50,51,10):
                                        appv_influence_factor = appv_influence_factor/100
                                        for entire_stock_duration_positive_period_counter_sensitivity_factor in range (3,6,1):
                                            for nr_neg_p_to_buy in range (4,6,1):
                                                for nr_pos_p_to_buy in range (4,6,1):
                                                        alpha = 0
                                                                                                                
                                                        #let's start it!
                                                        
                                                        iteration = iteration + 1
                                                        iteration_list.append(iteration)
                                                        
                                                        file_countlist.append(file_counter)
                                                        #record the parameter vector currently being tested
                                                        param_vector = []
                                                        param_vector.append(avrgfactor)
                                                        param_vector.append(average_factor_sensitivity)
                                                        param_vector.append(lookback)
                                                        param_vector.append(rule)
                                                        param_vector.append(index_sensitivity_factor)
                                                        param_vector.append(VAD_influence_factor)
                                                        param_vector.append(vap_influence_factor)
                                                        param_vector.append(lookback_influence_factor)
                                                        param_vector.append(appv_influence_factor)
                                                        param_vector.append(entire_stock_duration_positive_period_counter_sensitivity_factor)
                                                        param_vector.append(nr_neg_p_to_buy)
                                                        param_vector.append(nr_pos_p_to_buy)
                                                        param_vector.append(alpha)
                                                        #append the vector to the total list
                                                        total_tested_param_matrix.append(param_vector)
                                                        
                                                        
                                                        
                                                        tested_param_matrix.append(param_vector)
                                                        
                                                        #Run algo
                                                        algo_output = Algofunc(avrgfactor,
                                                                     average_factor_sensitivity,
                                                                     lookback,
                                                                     rule,
                                                                     index_sensitivity_factor,
                                                                     VAD_influence_factor,
                                                                     vap_influence_factor,
                                                                     lookback_influence_factor,
                                                                     appv_influence_factor,
                                                                     entire_stock_duration_positive_period_counter_sensitivity_factor,
                                                                     alpha,
                                                                     df)
                                                        
                                                        #Run P&L Calc
                                                        pnl_output = pnlfunc(sharenr,
                                                                             nr_neg_p_to_buy,
                                                                             on_off_avrgdelta_pnl,
                                                                             nr_pos_p_to_buy)
                                                        
                                                        
                                                        
                                                        #identify the final profit or loss from the parameter vector
                                                        runningtotal = pnl_output[5]
                                                        rt_list = pnl_output[1]
                                                        
                                                        #record the final profit or loss in the file-wide list
                                                        
                                                        closing_profit_list.append(runningtotal)   
                                                        
                                                        nr_trades_list.append(len(rt_list))
                                                        
                                                        #record the final profit or loss in the master list
                                                        total_closing_profit_list.append(runningtotal)            
                                                        total_nr_trades_list.append(len(rt_list))
                                                        
                                                        
                                                        
                                                        #print something based on in period counter?
                                                        if iteration % 100 == 0:
                                                            print(iteration)
                                                        
                                                        #REPORTING INCREMENTALLY    
                                                        if file_countlist[iteration]-file_countlist[iteration-1] !=0:
                                                            
                                                            csv_swap_tracker_iterationnr.append(iteration)
                                                                                                                        
                                                            length_exploration_stock_n = csv_swap_tracker_iterationnr[len(csv_swap_tracker_iterationnr)-1]
                                                            length_exploration_stock_n_minus1 = csv_swap_tracker_iterationnr[len(csv_swap_tracker_iterationnr)-2]
                                                            diff = length_exploration_stock_n - length_exploration_stock_n_minus1
                                                            
                                                            closing_profit_list = total_closing_profit_list[-diff:]
                                                                                                                        
                                                            tested_param_matrix = total_tested_param_matrix[-diff:]
                                                            
                                                            print('Info Output From Previous File Tested')
                                                            
                                                            max_pnl = max(closing_profit_list)
                                                            index_of_max_pnl = closing_profit_list.index(max_pnl)

                                                            print('parameter combination of highest profit gained iteration')
                                                            print(tested_param_matrix[index_of_max_pnl])
                                                            
                                                            #here we append the combination of parameters that yielded the highest profits
                                                            tested_param_matrix_combolist.append(tested_param_matrix[index_of_max_pnl])
                                                            print('highest profit gained')
                                                            print(max_pnl)

                                                            #make the tested parameter matrix indo a pandas dataframe
                                                            param_matrix = pd.DataFrame(tested_param_matrix)

                                                            #output periodic lists of all the 
                                                            avrgfactor_x = param_matrix[0].tolist()
                                                            average_factor_sensitivity_x = param_matrix[1].tolist()
                                                            lookback_x = param_matrix[2].tolist()
                                                            rule_x = param_matrix[3].tolist()
                                                            index_sensitivity_factor_x = param_matrix[4].tolist()
                                                            VAD_influence_factor_x = param_matrix[5].tolist()
                                                            vap_influence_factor_x = param_matrix[6].tolist()
                                                            lookback_influence_factor_x = param_matrix[7].tolist()
                                                            appv_influence_factor_x = param_matrix[8].tolist()
                                                            entire_stock_duration_positive_period_counter_sensitivity_factor_x = param_matrix[9].tolist()
                                                            nr_neg_p_to_buy_x = param_matrix[10].tolist()
                                                            nr_pos_p_to_buy_x = param_matrix[11].tolist()
                                                            alpha_x = param_matrix[12].tolist() 



                                                            
                                                            param_matrix_output = [avrgfactor_x,
                                                                 average_factor_sensitivity_x,
                                                                 lookback_x,
                                                                 rule_x,
                                                                 index_sensitivity_factor_x,
                                                                 VAD_influence_factor_x,
                                                                 vap_influence_factor_x,
                                                                 lookback_influence_factor_x,
                                                                 appv_influence_factor_x, 
                                                                 entire_stock_duration_positive_period_counter_sensitivity_factor_x,
                                                                 nr_neg_p_to_buy_x,
                                                                 nr_pos_p_to_buy_x,
                                                                 alpha_x]
                                                            pnl_output = [closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list, closing_profit_list]
                                                                                               
                                                            #x=[1,2,3,1,2,3,1,2,3]
                                                            #y=[20,30,40,50,60,70,50,80,40]
                                                            
                                                            
                                                            titles = ['avrgfactor_x',
                                                                 'average_factor_sensitivity_x',
                                                                 'lookback_x',
                                                                 'rule_x',
                                                                 'index_sensitivity_factor_x',
                                                                 'VAD_influence_factor_x',
                                                                 'vap_influence_factor_x',
                                                                 'lookback_influence_factor_x',
                                                                 'appv_influence_factor_x', 
                                                                 'entire_stock_duration_positive_period_counter_sensitivity_factor_x',
                                                                 'nr_neg_p_to_buy_x',
                                                                 'nr_pos_p_to_buy_x',
                                                                 'alpha_x']
                                                            
                                                            for i in range(13):
                                                                print(f'{titles[i]}')
                                                                x = param_matrix_output[i]
                                                                y = pnl_output[i]
                                                                # calculate sum and count of y values for each x value
                                                                sum_y = defaultdict(list)
                                                                count_y = defaultdict(int)
                                                                for x_val, y_val in zip(x, y):
                                                                    sum_y[x_val].append(y_val)
                                                                    count_y[x_val] += 1

                                                                # calculate average and median of y values for each x value
                                                                avg_y = {}
                                                                med_y = {}
                                                                for k in sum_y:
                                                                    avg_y[k] = sum(sum_y[k]) / count_y[k]
                                                                    sorted_y = sorted([y[i] for i in range(len(x)) if x[i] == k])
                                                                    med_y[k] = stat.median(sorted_y)

                                                                # print the results
                                                                for k in avg_y:
                                                                    print(f"Average of y values for x = {k}: {avg_y[k]}")
                                                                    print(f"Median of y values for x = {k}: {med_y[k]}")
                                                            
                                                            
                                                            
                                                            
                                                            #MAIN OPTIMIZATION ALGORITHM -->> FINDING PEAKS
                                                            peak_func_input = np.array(closing_profit_list)

                                                            # Smooth the data using a moving average
                                                            window_size = 6
                                                            smooth_peak_func_input = np.convolve(peak_func_input, np.ones(window_size)/window_size, mode='valid')
                                                            
                                                            # Find the peaks in the smoothed revenue data
                                                            peaks, _ = find_peaks(smooth_peak_func_input,prominence=20000, distance=4)
                                                            
                                                            print('Number of Local Maxima Identified')
                                                            print(len(smooth_peak_func_input[peaks]))


                                                            # Set printing options for NumPy arrays
                                                            np.set_printoptions(suppress=True, precision=2)

                                                            # Print the peak values
                                                            for i, peak_value in enumerate(smooth_peak_func_input[peaks]):
                                                                index = peaks[i]
                                                                local_index_list.append(index)
                                                                total_optimal_param_matrix.append(tested_param_matrix[index])
                                                                total_stock_ID_list.append(file)
                                                                total_local_max_explored_list.append(smooth_peak_func_input[peaks][i])
                                                            
                                                            
                                                            
                                                            # Open the CSV file for writing
                                                            with open("C:/Users/Asus/Desktop/STOCK DATA TO MINE/OUTPUT/OUTPUT.csv", mode='w', newline='') as file:

                                                                # Create a CSV writer
                                                                writer = csv.writer(file)
                                                                
                                                                # Write the two lists to the file
                                                                writer.writerow(['List 1'] + total_stock_ID_list)
                                                                writer.writerow(['List 2'] + total_local_max_explored_list)
                                                                
                                                                # Write the matrix to the file
                                                                writer.writerows(total_optimal_param_matrix)



#tested_param_list = []
#closing_profit_list = []
#nr_trades_list = []


#plt.scatter(iteration_list,closing_profit_list)

#
length_of_last = csv_swap_tracker_iterationnr[-1] - csv_swap_tracker_iterationnr[-2]
closing_profit_list = total_closing_profit_list[-length_of_last:]
max_pnl = max(closing_profit_list)
index_of_max_pnl = closing_profit_list.index(max_pnl)

print('Report for the last file checked')
print('parameter combination of highest profit gained iteration')
print(tested_param_matrix[index_of_max_pnl])
print('highest profit gained')
print(max_pnl)


#total_optimal_param_matrix.append(tested_param_matrix[index])
#total_stock_ID_list.append(file)
#total_local_max_explored_list.append(smooth_peak_func_input[peaks][i])


    
#make the tested parameter matrix indo a pandas dataframe

print(iteration)
print(iteration_list)
iteration_list.insert(0,0)

#make a list and matrix of ALL combinations for ALL files



print('#########################################')
print('#########################################')
print('#########################################')
print('Nr. of Parameters Tested')
print(len(total_tested_param_matrix))
print('#########################################')
print('Nr. of Optimal Parameters Found')
print(len(total_optimal_param_matrix))
print('#########################################')
print('Average TOTAL profit of all trades')
print(stat.mean(total_closing_profit_list))
print('#########################################')
print('Average Nr. of Trades')
print(stat.mean(total_nr_trades_list))
print('#########################################')






#KEEP TRACK AT WHICH ITERATION THE CSV FILES SWAP
csv_swap_tracker_iterationnr = []
param_matrix = pd.DataFrame(total_optimal_param_matrix)

#output periodic lists of all the 
avrgfactor_x = param_matrix[0].tolist()
average_factor_sensitivity_x = param_matrix[1].tolist()
lookback_x = param_matrix[2].tolist()
rule_x = param_matrix[3].tolist()
index_sensitivity_factor_x = param_matrix[4].tolist()
VAD_influence_factor_x = param_matrix[5].tolist()
vap_influence_factor_x = param_matrix[6].tolist()
lookback_influence_factor_x = param_matrix[7].tolist()
appv_influence_factor_x = param_matrix[8].tolist()
entire_stock_duration_positive_period_counter_sensitivity_factor_x = param_matrix[9].tolist()
nr_neg_p_to_buy_x = param_matrix[10].tolist()
nr_pos_p_to_buy_x = param_matrix[11].tolist()
alpha_x = param_matrix[12].tolist() 




param_matrix_output = [avrgfactor_x,
     average_factor_sensitivity_x,
     lookback_x,
     rule_x,
     index_sensitivity_factor_x,
     VAD_influence_factor_x,
     vap_influence_factor_x,
     lookback_influence_factor_x,
     appv_influence_factor_x, 
     entire_stock_duration_positive_period_counter_sensitivity_factor_x,
     nr_neg_p_to_buy_x,
     nr_pos_p_to_buy_x,
     alpha_x]


pnl_output = [total_local_max_explored_list,
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list, 
              total_local_max_explored_list]
                                   
#x=[1,2,3,1,2,3,1,2,3]
#y=[20,30,40,50,60,70,50,80,40]


titles = ['avrgfactor_x',
     'average_factor_sensitivity_x',
     'lookback_x',
     'rule_x',
     'index_sensitivity_factor_x',
     'VAD_influence_factor_x',
     'vap_influence_factor_x',
     'lookback_influence_factor_x',
     'appv_influence_factor_x', 
     'entire_stock_duration_positive_period_counter_sensitivity_factor_x',
     'nr_neg_p_to_buy_x',
     'nr_pos_p_to_buy_x',
     'alpha_x']

for i in range(13):
    print(f'{titles[i]}')
    x = param_matrix_output[i]
    y = pnl_output[i]
    # calculate sum and count of y values for each x value
    sum_y = defaultdict(list)
    count_y = defaultdict(int)
    for x_val, y_val in zip(x, y):
        sum_y[x_val].append(y_val)
        count_y[x_val] += 1

    # calculate average and median of y values for each x value
    avg_y = {}
    med_y = {}
    for k in sum_y:
        avg_y[k] = sum(sum_y[k]) / count_y[k]
        sorted_y = sorted([y[i] for i in range(len(x)) if x[i] == k])
        med_y[k] = stat.median(sorted_y)

    # print the results
    for k in avg_y:
        print(f"Average of y values for x = {k}: {avg_y[k]}")
        print(f"Median of y values for x = {k}: {med_y[k]}")

# Create multiple scatter plots in a loop
for i in range(13):
    plt.figure(i+1)
    plt.scatter(param_matrix_output[i], pnl_output[i])
    plt.title(f'{titles[i]}')
    plt.xlabel('Parameter Values Explored')
    plt.ylabel('Money Earned Investing over Time (1000s)')




   
# Show the plots
plt.show()


#NOTES AND SHIT
#Make the index inflexion level determine a larger amount of shares to be bought????                        
       