# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 04:06:03 2020

this module implements methods for maintaining network level processes
such as:
    1- moving across slots
    2- initializing the relationships between regions and o-d pairs
    3- aggregating results
    
@author: cesny
"""
from utils import readCSV, addTimeStamp, getODdata, getOrderedService, getLambdaMLE
from odpair import odpair
from region import region
import numpy as np


def getSavings(probs, beta_c, beta_d):
    '''
    after running peak-load-pricing across time, get the 
    savings and the lost revenue
    '''
    savings = dict()  # dict storing savings per slot, region
    lostRev = dict()  # dict storing lost revenue per slot across regions
    for slot in probs:
        savings[slot]=dict()
        regRev = list()  # stores revenue per region
        for reg in list(np.arange(1,numRegions+1,1)):
            optRev = list()  # stores revenue for each option in region
            savings[slot][reg] = list()
            for key, prk in enumerate(probs[slot][reg]):
                sav = (1.0/beta_c) * (np.log(prk) - np.log(probs[slot][reg][0]) - beta_d*(key)  )
                savings[slot][reg].append(sav)
                optRev.append(sav*prk)
            regRev.append(np.sum(optRev))
        lostRev[slot] = np.mean(regRev)
    return savings, lostRev


def avgNewz(newz):
    '''
    for plotting

    '''
    out=dict()
    for slot in newz:
        temp=list()
        for key in newz[slot]:
            temp.append(newz[slot][key])
        out[slot] = np.mean(temp)
    return out
    
    

def processOutput(probs, zs):
    '''
    changes probs to lists instead of numpy array
    '''
    for slot in probs:
        for reg in probs[slot]:
            probs[slot][reg] = list(probs[slot][reg][:,0])
            zs[slot][reg] = zs[slot][reg][0]
    return probs,zs


if __name__ == '__main__':
    dDict, head = readCSV('data/ridesLyftMHTN14.csv')
    dDict = addTimeStamp(dDict, slotInMinutes=10)
    slotInMinutes=10
    numRegions = max(dDict['region']) - min(dDict['region']) + 1  # 4
    firstTimePt = min(dDict['TimeIn'])  # 1
    maxTimePt = max(dDict['TimeIn']) + 1  # 37
    windowLengthSlots = 5  # each window is 5*5 = 25 minutes (6 possible departure times: now, 5 mints, 10 mints, 15 mints, 20 mints, 25 mints)
    windowInMinutes = windowLengthSlots * slotInMinutes
    lastTimePt = maxTimePt - windowLengthSlots
    listofSlots = list()
    listofWindows = list()
    for timePt in list(np.arange(firstTimePt, lastTimePt, 1)):
        listofSlots.append((timePt, timePt+1))
    for timePt in list(np.arange(firstTimePt, lastTimePt, 1)):
        listofWindows.append((timePt+1, timePt+1+windowLengthSlots ))
    
    # optimization paramaters and results
    vot = 8.0/(60.0/slotInMinutes)  # x dollars per hour is VOT, divide that by 12 to get dollar per slot
    beta_c = 1
    beta_d = -vot*beta_c
    weight = 1
    probs = dict()  # stores the values of the probabilities from the optimization problem across time
    zs = dict()  # stores the values of z from the optimization problem across time
    status=dict()  # check if found optimal val
    opval = dict()  # stores optimization optimal val
    loadProc = dict()
    PS = dict()
    OS = dict()
    PE = dict()
    OE = dict()
    for slot in listofSlots:  # the optimization is stored per slot and for each region, that's how we store the data!
        probs[slot] = dict()
        zs[slot]=dict()
        status[slot] = dict()
        opval[slot] = dict()
        loadProc[slot] = dict()
        PS[slot] = dict()
        OS[slot] = dict()
        PE[slot] = dict()
        OE[slot] = dict()
        
    prevStarts = dict()  # maintains starts across time windows, this is the cumulative starts *since slot[1]* (beginning of window) till the end of time such that the requests were received prior to slot[0]
    prevEnds = dict()  # maintains ends across time windows, this is the cumulative ends *since slot[1]* (beginning of window) onwards such that the requests were received prior to slot[0]
    # note that we discount starts or ends that occur prior time slot[1], i.e., no longer in the picture, we are only concerned with cumulative starts/ends that appear <b> after the beginning of the time window</b> given that the request was received prior to slot[0]
    for origin in list(np.arange(1,numRegions+1,1)):
        for dest in list(np.arange(1,numRegions+1,1)):
            prevStarts[(origin, dest)] = dict()
            prevEnds[(origin, dest)] = dict()
            for timePt in list(np.arange(firstTimePt, maxTimePt+1, 1)):
                prevStarts[(origin, dest)][timePt] = 0
                prevEnds[(origin, dest)][timePt] = 0
    
    for key, slot in enumerate(listofSlots):
        #key = 0
        #slot = listofSlots[key]
        # discount starts or ends that occur prior to time slot[1] (only concerned with what happens since beginning of window)
        for origin in list(np.arange(1,numRegions+1,1)):
            for dest in list(np.arange(1,numRegions+1,1)):
                for timePt in list(np.arange(slot[1], maxTimePt+1, 1)):  # remove what has been previously observed that doesn't matter anymore from the cumulative starts and ends
                    prevStarts[(origin, dest)][timePt] = prevStarts[(origin, dest)][timePt] - prevStarts[(origin, dest)][slot[0]]
                    prevEnds[(origin, dest)][timePt] = prevEnds[(origin, dest)][timePt] - prevEnds[(origin, dest)][slot[0]]
        
        window = listofWindows[key]  # get the current window
        keysToExtract = list(np.arange(window[0], window[1]+1, 1))
        prevStartsWin = dict()
        prevEndsWin = dict()
        for origin in list(np.arange(1,numRegions+1,1)):
            for dest in list(np.arange(1,numRegions+1,1)):
                prevStartsWin[(origin, dest)] = {key: prevStarts[(origin, dest)][key] for key in keysToExtract}
                prevEndsWin[(origin, dest)] = {key: prevEnds[(origin, dest)][key] for key in keysToExtract}
        
        dataDictOD = dict()  # intialize the dict of dicts, stores the data segregated by OD pair
        for orig in list(np.arange(1,numRegions+1,1)):  # fill the data dict
            for dest in list(np.arange(1, numRegions+1, 1)):
                dataDictOD[(orig, dest)] = getODdata(dDict, orig, dest, window)
        orderSerOD = dict()  # get the ordered service rate by OD
        lamMLE = dict()  # stores the maximum likelihood estimator for arrival rates for each O-D pair
        for orig in list(np.arange(1,numRegions+1,1)):  # fill the data dict
            for dest in list(np.arange(1, numRegions+1, 1)):
                orderSerOD[(orig, dest)] = getOrderedService(dataDictOD[(orig, dest)], slotInMinutes)
                lamMLE[(orig, dest)] =  getLambdaMLE(dataDictOD[(orig, dest)], slotInMinutes, windowInMinutes)[0]
        print('... done with initial data processing ...')
        
        # create the dict of odpair classes
        odclasses = dict()  # {(o,d):class, ..}
        for orig in list(np.arange(1,numRegions+1,1)):  # fill the classes
            for dest in list(np.arange(1, numRegions+1, 1)):
                odclasses[(orig, dest)] = odpair(orig, dest, slot, lamMLE[(orig, dest)], window, orderSerOD[(orig, dest)])
                odclasses[(orig, dest)].updateObsStarts(prevStartsWin[(orig, dest)])  # add prev. starts in window
                odclasses[(orig, dest)].updateObsEnds(prevEndsWin[(orig, dest)])  # add prev. ends in window
                odclasses[(orig, dest)].createFutureStarts()  # creates future starts
                odclasses[(orig, dest)].createFutureEnds()  # creates future ends
        print('... initialized odpair classes ...')
        
        # create the regions and implement the optimization
        regclasses = dict()
        for reg in list(np.arange(1,numRegions+1,1)):
            regclasses[reg] = region(reg, slot, window, odclasses)
            regclasses[reg].updateObsStarts()
            regclasses[reg].updateObsEnds()
            regclasses[reg].createFutureStarts()
            regclasses[reg].createFutureEnds()
            loadProc[slot][reg], PS[slot][reg], OS[slot][reg], PE[slot][reg], OE[slot][reg] = regclasses[reg].loadProcess()
            regclasses[reg].nowStart()
            regclasses[reg].nowEnd()
            probs[slot][reg], zs[slot][reg], status[slot][reg], opval[slot][reg] = regclasses[reg].optimize(beta_c, beta_d, weight)
            for key, pk in enumerate(probs[slot][reg][:,0]):  # kills small negative values due to numerical error
                if pk<=0:
                    print('... warning, probabilities are too close to zero! ...')
                    probs[slot][reg][key, 0] = 0.00000001
            probs[slot][reg][0,0]+=1-sum(list(probs[slot][reg][:,0]))  # kills round off errors, makes sure sum to 1
        print('... done creating regions and optimizing ...')
        
        # now use the optimal probabilities to go through observed rides and probabilistically delay each one!
        dataDictSlotOD = dict()
        for orig in list(np.arange(1,numRegions+1,1)):  # find observed rides
            for dest in list(np.arange(1, numRegions+1, 1)):
                dataDictSlotOD[(orig, dest)] = getODdata(dDict, orig, dest, slot)  # get the data that would be observed within the slot
                for ind, timein in enumerate(dataDictSlotOD[(orig, dest)]['TimeIn']):
                    startchoice = np.random.choice(keysToExtract, p=list(probs[slot][orig][:,0]))
                    endchoice = startchoice +  (dataDictSlotOD[(orig, dest)]['TimeOut'][ind] -  timein)
                    for timePt in list(np.arange(startchoice, maxTimePt+1, 1)):
                        prevStarts[(orig, dest)][timePt] += 1
                    for timePt in list(np.arange(endchoice, maxTimePt+1, 1)):
                        prevEnds[(orig, dest)][timePt] += 1
        print('... done updating starts ands ends by time point ...')
    
    # get results
    newprobs, newz = processOutput(probs, zs)
    savings, lostRev = getSavings(newprobs, beta_c, beta_d)
    print('... got results! ...')
                    
                    

