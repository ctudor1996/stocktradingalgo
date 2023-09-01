
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



df = pd.read_csv('file:///C:/Users/Asus/Desktop/STOCK DATA TO MINE/MINE/HSBC.csv' )

df.head()

##########################################################################################################################
##########################################################################################################################
# Section 2 - DEGREES OF FREEDOM / VARIABLES 
##########################################################################################################################
##########################################################################################################################


##########################################################################################################################
# Section 3.1 - Algorithm DoF / Vars
##########################################################################################################################

#the moving average factor
avrgfactor = 5

average_factor_sensitivity = 0.717093112



#determines the number of past positive (or negative respectively) periods the current period is compared to
#! future plans to maybe make smart AI try to account for black swan events
lookback = 4


#Index Critical value past which to buy
rule = 1.182701737



index_sensitivity_factor = 0.935001326


#Adjusts how much the volume support influences the adjusted price change. Past days volume is weighed more than current day volume
#it is a power, should be between 0.1 and 0.9
VAD_influence_factor = 0.685377572


#adjusts how much volume support influences the price period vector used in the index
#it is a power, should be between 0.1 and 0.9
vap_influence_factor = 0.501893916




lookback_influence_factor = 0.935001326




appv_influence_factor = 0.535001326


entire_stock_duration_positive_period_counter_sensitivity_factor = 5

alpha = 0
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


##########################################################################################################################
##########################################################################################################################
# Section 3 - Algorithm
##########################################################################################################################
##########################################################################################################################

total_infolist = []

def Algofunc(rule,
             average_factor_sensitivity,
             index_sensitivity_factor,
             VAD_influence_factor,
             vap_influence_factor,
             lookback_influence_factor,
             appv_influence_factor,
             entire_stock_duration_positive_period_counter_sensitivity_factor,
             alpha,
             avrgfactor,
             lookback):
    
    
    
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
                index = alpha + abs(appv**appv_influence_factor) / ((sum(NP_countlist[NP_counter-lookback:NP_counter])/lookback)**lookback_influence_factor)**index_sensitivity_factor
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
    algo_output = Algofunc(rule,
                           average_factor_sensitivity,
                         index_sensitivity_factor,
                         VAD_influence_factor,
                         vap_influence_factor,
                         lookback_influence_factor,
                         appv_influence_factor,
                         entire_stock_duration_positive_period_counter_sensitivity_factor,
                         alpha,
                         avrgfactor,
                         lookback)


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


def Algo_total_func(rule,
             average_factor_sensitivity,
             index_sensitivity_factor,
             VAD_influence_factor,
             vap_influence_factor,
             lookback_influence_factor,
             appv_influence_factor,
             entire_stock_duration_positive_period_counter_sensitivity_factor,
             alpha,
             avrgfactor,
             lookback,
             nr_neg_p_to_buy,
             nr_pos_p_to_buy):
    
    
    #Algo Outut from invoking algofunc
    algo_output = Algofunc(rule,
                 average_factor_sensitivity,
                 index_sensitivity_factor,
                 VAD_influence_factor,
                 vap_influence_factor,
                 lookback_influence_factor,
                 appv_influence_factor,
                 entire_stock_duration_positive_period_counter_sensitivity_factor,
                 alpha,
                 avrgfactor,
                 lookback)


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
    
    
    pnl_output = pnlfunc(sharenr,
                         nr_neg_p_to_buy,
                         on_off_avrgdelta_pnl,
                         nr_pos_p_to_buy)
    
    profit_list = pnl_output[3]
    
    profit = profit_list[-1]
    
    return(profit)





profit_outputted = Algo_total_func(average_factor_sensitivity,
             index_sensitivity_factor,
             VAD_influence_factor,
             vap_influence_factor,
             lookback_influence_factor,
             appv_influence_factor,
             entire_stock_duration_positive_period_counter_sensitivity_factor,
             alpha,
             avrgfactor,
             lookback,
             rule,
             nr_neg_p_to_buy,
             nr_pos_p_to_buy)

print(profit_outputted)

 

# Define the gradient descent function.

def gradient_descent(Algo_total_func, learning_rate, num_iterations):

    # Start with a random set of input values.

    inputs = [rule,
              average_factor_sensitivity,
              index_sensitivity_factor,
              VAD_influence_factor,
              vap_influence_factor,
              lookback_influence_factor,
              appv_influence_factor,
              entire_stock_duration_positive_period_counter_sensitivity_factor,
              alpha,
              avrgfactor,
              lookback,              
              nr_neg_p_to_buy,
              nr_pos_p_to_buy]

    float_inputs = inputs[0:9]
    integer_inputs = inputs[9:13]
 

    for i in range(num_iterations):

        # Calculate the profit for the current set of input values.

        profit = Algo_total_func(*inputs)


        # Calculate the gradient of the profit with respect to each input.

        gradients = []

        for j in range(9):

            input_plus = float_inputs

            input_plus[j] += 0.001

            profit_plus = Algo_total_func(*input_plus)

 

            input_minus = float_inputs

            input_minus[j] -= 0.001

            profit_minus = Algo_total_func(*input_minus)
            
            gradients.append((profit_plus - profit_minus) / 0.002)

        for j in range(4):
            input_plus = integer_inputs

            input_plus[j] += 1

            profit_plus = Algo_total_func(*input_plus)

 

            input_minus = integer_inputs

            input_minus[j] -= 1

            profit_minus = Algo_total_func(*input_minus)
            
            gradients.append((profit_plus - profit_minus) / 2)
 

        # Update the input values in the direction that increases the profit.

        for j in range(len(inputs)):

            if inputs.index(inputs[j]) <=8 :

                inputs[j] += learning_rate * gradients[j]

            else:

                inputs[j] = int(inputs[j] + learning_rate * gradients[j])

 

        

    # Return the final set of input values and profit.

    return inputs, Algo_total_func(*inputs)

 

# Run gradient descent with a learning rate of 0.01 and 1000 iterations.

inputs, profit = gradient_descent(Algo_total_func(avrgfactor,
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
             nr_neg_p_to_buy,
             nr_pos_p_to_buy), 
                                  0.01, 50)

 

print("Optimal inputs:", inputs)

print("Profit:", profit)



