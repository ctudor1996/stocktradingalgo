
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
import yfinance as yf
import datetime
import csv


######################## DATA and USER INPUTS ####################
##################################################################

#df = pd.read_csv('file:///C:/Users/Asus/Desktop/STOCK DATA TO MINE/MINE/STEM.csv' )



### YFINANCE DOWNLOADS
stem = yf.Ticker('stem')

stem_hist = stem.history(period='max', interval='1d')
df=stem_hist
df
df = df.reset_index()
#df['date'] = df['date'].apply(str_to_datetime)


##########################################################################################################################
##########################################################################################################################
# Section 2 - DEGREES OF FREEDOM / VARIABLES 
##########################################################################################################################
##########################################################################################################################


##########################################################################################################################
# Section 3.1 - Algorithm DoF / Vars
##########################################################################################################################

#the moving average factor
avrgfactor = 4

average_factor_sensitivity = 1.1671



#determines the number of past positive (or negative respectively) periods the current period is compared to
#! future plans to maybe make smart AI try to account for black swan events
lookback = 5


#Index Critical value past which to buy
rule = 0.8827



index_sensitivity_factor = 1.135


#Adjusts how much the volume support influences the adjusted price change. Past days volume is weighed more than current day volume
#it is a power, should be between 0.1 and 0.9
VAD_influence_factor = 0.685377572


#adjusts how much volume support influences the price period vector used in the index
#it is a power, should be between 0.1 and 0.9
vap_influence_factor = 0




lookback_influence_factor = 1.185




appv_influence_factor = 0.285


entire_stock_duration_positive_period_counter_sensitivity_factor = 4

alpha = 0
##########################################################################################################################
# Section 3.2 - P&L/BuyRule 
##########################################################################################################################

sharenr = 100


####CHANGES TO BUY RULE

#nr of negative periods in current decline to activate the buy rule
nr_neg_p_to_buy = 3

#nr of positive periods in current rise to activate the buy rule
nr_pos_p_to_buy = 8

#T/F 1=the moving average change in price must also be in accordance with the decline/rise of the period for the buy rule to be activated, 0= not
on_off_avrgdelta_pnl = 1

##########INPUT TOTAL PARAMS TO TEST SINGLE TARGET STOCK
total_param_init = [1.055,	0.767,	0.921,	0.688,	0.506,	1.011,	0.460,	4,	5,	4,	4,	6,]

rule = total_param_init[0]
average_factor_sensitivity = total_param_init[1]
index_sensitivity_factor = total_param_init[2]
VAD_influence_factor = total_param_init[3]
vap_influence_factor = total_param_init[4]
lookback_influence_factor = total_param_init[5]
appv_influence_factor = total_param_init[6]

entire_stock_duration_positive_period_counter_sensitivity_factor = total_param_init[7]
avrgfactor = total_param_init[8]
lookback = total_param_init[9]
nr_neg_p_to_buy = total_param_init[10]
nr_pos_p_to_buy = total_param_init[11]



total_param_runit = [rule,
               average_factor_sensitivity,
               index_sensitivity_factor,
               VAD_influence_factor,
               vap_influence_factor,
               lookback_influence_factor,
               appv_influence_factor,
               ###
               entire_stock_duration_positive_period_counter_sensitivity_factor,
               avrgfactor,
               lookback,
               nr_neg_p_to_buy,
               nr_pos_p_to_buy]

    
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
             entire_stock_duration_positive_period_counter_sensitivity_factor):
    
    
    
    #vwap estimate for the day
    df['VWAP Estimate'] = (df['High']*0.15 + df['Close']*0.7 + df['Low']*0.15)
    
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
           pos_countlist,
           avrgdelta)

        





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
                 entire_stock_duration_positive_period_counter_sensitivity_factor
                 )


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
# Section FINAL - Running the Single target analysis
##########################################################################################################################
##########################################################################################################################


##########################################################################################################################
# Section Final.1 - Information Prints (FINAL P&L VALUES)
##########################################################################################################################





##########################################################################################################################
# Section Final.2 - Graph Plots of Results
##########################################################################################################################




##########################################################################################################################
# Section Final.3 - User Inputs for current day calculations
##########################################################################################################################




#######################USER INPUT FOR ANALYSIS
#print('choose whether to analyze stock for current day price')
#user_input = input("Do you want to proceed? (y/n):")

#if user_input == "y":
    # Execute the actions if the user wants to proceed
    #print("Proceeding with current moment inputs")
    #print('######')
    #print('insert current price')
    #cp = input()
    
    # Add more actions here
#else:
    # Execute alternative actions if the user doesn't want to proceed
    #print("Not prompting current moment inputs. End")
    # Add more alternative actions here

##########################################################################################################################
# Section Final.4 - function combining p&l rule and algo
##########################################################################################################################

def Algo_total_func(rule,
               average_factor_sensitivity,
               index_sensitivity_factor,
               VAD_influence_factor,
               vap_influence_factor,
               lookback_influence_factor,
               appv_influence_factor,
               entire_stock_duration_positive_period_counter_sensitivity_factor,
               avrgfactor,
               lookback,
               nr_neg_p_to_buy,
               nr_pos_p_to_buy):
    
    
    #Algo Outut from invoking algofunc
    param_for_algo = [avrgfactor,
                 average_factor_sensitivity,
                 lookback,
                 rule,
                 index_sensitivity_factor,
                 VAD_influence_factor,
                 vap_influence_factor,
                 lookback_influence_factor,
                 appv_influence_factor,
                 entire_stock_duration_positive_period_counter_sensitivity_factor]
    algo_output = Algofunc(*param_for_algo)


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
    
    param_for_pnl = [sharenr,
                nr_neg_p_to_buy,
                on_off_avrgdelta_pnl,
                nr_pos_p_to_buy]
    
    pnl_output = pnlfunc(*param_for_pnl)
    
    profit_list = pnl_output[3]
    
    #OUTPUTS:
        
    c_index = index_list[-1:]
    curr_ind = c_index[0]
    
    sharecount = pnl_output[4]
    runningtotal= pnl_output[5]
    if sharecount != 0:
    
        if sharecount < 0:
            exit_value = runningtotal - p_list[len(p_list)-1]*sharecount
        
        elif sharecount > 0:
            exit_value = p_list[len(p_list)-1]*sharecount + runningtotal
        

    if sharecount == 0:
        exit_value = runningtotal
    
    proximity_to_rule = abs(rule - curr_ind)
    
    if len(p_list)<365:
        years = 1
    else:
        years = int(len(p_list)/365)
    start_price = p_list[0]
    start_inv = start_price * sharenr
    out_inv = exit_value
    ROI = (out_inv/start_inv)**(1/years)
    
    rt_list = pnl_output[1]
    
    nr_trades = len(rt_list)
    
    
    if avrgdelta[-1] < 0:
        buy_sell = 'buy'
    else:
        buy_sell = 'sell'
    
    return(curr_ind, proximity_to_rule, exit_value, ROI, nr_trades, buy_sell)

##########################################################################################################################
# Section Final.5 - MARKET SURVEY
##########################################################################################################################

print('test algo output')
print(Algo_total_func(*total_param_runit))

stock_list_df = pd.read_csv('file:///C:/Users/Asus/Desktop/Economic Data/ALL REVOLUT STOCKS LIST FOR SEARCH - 1 SHORTENED.csv' )


ticker_list = stock_list_df['Symbol'].tolist()


ticker_test = ticker_list
print('Market Survey for the following tickers:')
print(len(ticker_test))
print(ticker_test)

curr_ind_list=[]
proximity_to_rule_list = []
exit_value_list = []
ROI_list =[]
nr_trades_list = []
buy_sell_list = []
error_counter = 0
ticker_counter = 0

for ticker in ticker_test:
    try:
        ticker_counter = ticker_counter + 1
        
        if ticker_counter % 50 == 0:
            print(ticker_counter)
            
        local_ticker = yf.Ticker(ticker)

        ticker_hist = local_ticker.history(period='max', interval='1d')
        df=ticker_hist
        df = df.reset_index()
        local_calc = Algo_total_func(*total_param_runit)
        
        local_curr_ind = local_calc[0]
        local_proximity_to_rule = local_calc[1]
        local_exit_value = local_calc[2]
        local_ROI = local_calc[3]
        local_nr_trades = local_calc[4]
        local_buy_sell = local_calc[5]
        
        curr_ind_list.append(local_curr_ind)
        proximity_to_rule_list.append(local_proximity_to_rule)
        exit_value_list.append(local_exit_value)
        ROI_list.append(local_ROI)
        nr_trades_list.append(local_nr_trades)
        buy_sell_list.append(local_buy_sell)
        
    except:
        print('error occured for:')
        print(ticker)
        error_counter = error_counter + 1
    
    
# Open the CSV file for writing
with open("C:/Users/Asus/Desktop/Economic Data/Market Survey Output.csv", mode='w', newline='') as file:

    # Create a CSV writer
    writer = csv.writer(file)
    
    # Write the two lists to the file
    writer.writerow(['Ticker'] + ticker_test)
    writer.writerow(['Current Day Index'] + curr_ind_list)
    writer.writerow(['Proximity to Index Buy/Sell Rule'] + proximity_to_rule_list)
    writer.writerow(['Exit Value'] + exit_value_list)
    writer.writerow(['Nr. of Trades'] + nr_trades_list)
    writer.writerow(['Buy/Sell depending on whether the avrgdelta is pos or neg'] + buy_sell_list)
    writer.writerow(['ROI'] + ROI_list)
    
print('total number of errors')
print(error_counter)