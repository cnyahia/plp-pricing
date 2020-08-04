# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:07:17 2019

This code is for utilities such as managing data

@author: cesny
"""

import csv
import math
import numpy as np
import copy
from datetime import time


#-----------   data processing ---------------
def readCSV(file):
    '''
    ------------------
    read in csv file
    ------------------
    :param file: the csv file
    :return dataDict: a dictionary with the csv data
    :return head: the keys of the dictionary (i.e., first row of csv file)
    ------------------
    dDict= {'region':[1,2,3,4]; 'DOregion':[3,4,5,1], 
            'Dispatching_base_number':[.,.,.,.]}
    
    '''
    with open(file, mode='r') as infile:
        read = csv.reader(infile)
        head = next(read)
        dataDict = dict()
        for header in range(1, 12): dataDict[head[header]] = []  # create headersn the first entry of head is empty
        for row in read:
            for header in range(1, 12):
                if (head[header] == 'region') or (head[header] == 'DOregion'):
                    dataDict[head[header]].append(int(row[header]))
                else:
                    dataDict[head[header]].append(row[header])
    return dataDict, head[1:]


def addTimeStamp(dataDict, slotInMinutes):
    '''
    ------------------
    for each data point assign when it comes in
        based on conventinon that we operate by 
        timeSlots
    ------------------
    :param dataDict: input dictionary
    :param slotInMinutes: duration of a slot discretization (e.g. 5 minutes)
    :return dataDict: dictionary with the TimeIn and TimeOut entries referring
    to the slot number at which ride starts/ends
    ------------------
    data specific: starting hour is 16
        so first time slot is between 16:00:00 and 16:timeslot:00 = slot 1
    timeSlot numbering starts from 1
    ------------------
    Note that the time slot corresponds to between two time points
    t0| -- slot 1 -- |t1| -- slot 2 --|t2| --..
    ------------------
    '''
    dataDict['TimeIn'] = list()
    for val in dataDict['Pickup_DateTime']:
        splitDate = val.split(' ')
        splitTimeIn = splitDate[1].split('-')[0].split(':')
        timeStampMinutes = (float(splitTimeIn[0]) - 16)*60 + float(splitTimeIn[1]) + float(splitTimeIn[2])*(1.0/60)  # time stamp starting from zero at 16:00:00
        timeSlot = math.ceil(timeStampMinutes/slotInMinutes)
        dataDict['TimeIn'].append(timeSlot)
    
    dataDict['TimeOut'] = list()
    for val in dataDict['DropOff_datetime']:
        splitDate = val.split(' ')
        splitTimeOut = splitDate[1].split('-')[0].split(':')
        timeStampMinutes = (float(splitTimeOut[0]) - 16)*60 + float(splitTimeOut[1]) + float(splitTimeOut[2])*(1.0/60)  # time stamp starting from zero at 16:00:00
        timeSlot = math.ceil(timeStampMinutes/slotInMinutes)
        dataDict['TimeOut'].append(timeSlot)
    return dataDict



def getODdata(dataDict, orig, dest, window):
    '''
    returns the dictionary items that are associated with the
    specified od pair and the requests that are going to appear in the
    upcoming window

    ----------
    :param dataDict: the dictionary of trips
    :param orig: origin
    :param dest: desination
    :param window: time window (first timePt, last timePt) tuple
        e.g. (3,6) refers to the following window
        |3| -- slot 3 -- |4| -- slot 4 --|t5| -- slot 5 -- |t6|
    :returns dataDictOD: a dictionary trimmed to the orig-dest inputs
    ----------
    '''
    dataDictOD = copy.deepcopy(dataDict)
    delIndexes = list()
    winSlots = list(np.arange(window[0], window[1], 1))  # the slots of the window
    for key, val in enumerate(dataDict['region']):
        if (val != orig) or (dataDict['DOregion'][key] != dest) or (dataDict['TimeIn'][key] not in winSlots):
            delIndexes.append(key)
    # three conditions: correct origin, correct destination, starts in upcoming window
    for label in dataDictOD:
        for index in sorted(delIndexes, reverse=True):
            del dataDictOD[label][index]  # delete indexes that do not correspond to the od pair
    
    return dataDictOD


def map2time(timePt, slotInMinutes=10, startTime=time(hour=16, minute=00, second=00)):
    '''
    -------------
    maps time point to clock time
    -------------
    startTime: time at which data is collected
    slotInMinutes: length of slot in minutes
    timePt: time point
    -------------
    '''
    # first check if we need to add any hours
    addHour = math.floor(timePt*slotInMinutes/60)
    # check if we need to add any minutes on top of that
    addMinute = timePt*slotInMinutes - addHour*60
    # add to time
    newHour = startTime.hour + addHour
    newMinute = startTime.minute + addMinute
    newTime = time(hour=newHour, minute=newMinute, second=00)
    return str(newTime.hour)+":"+str(newTime.minute)

#------------------------------------------------------------------------





#-----------   arrival rates and empirical distribution ---------------

def getEmpiricalIntegral(t, listOS):
    '''
    ---------------------
    Gets the integration of the CDF of the empirical distribution up to time t
    i.e., computes int_{0}^{t}P(S<=u)du, int_{0}^{t}G(u)du
    ---------------------
    :param t: time point
    :param listOS: list of ordered service times,
        the service times in the list must be in units of slots
    :return integral: integral=int_{0}^{t}G(u)du
    ---------------------
    '''
    # get number of data points
    n = len(listOS)
    # first get the list of all the service times that are less than t in the window of interest
    storeService = list()
    key=0
    NotComplete = True
    while (NotComplete) and (listOS[key] <= t):
        storeService.append(listOS[key])
        key+=1
        if key not in np.arange(0, len(listOS), 1):  # check if index out of range
            NotComplete = False  # i.e., complete
    # add the time point as the last point
    storeService.append(t)
    # get empirical distribution CDF step function up to time t
    x = list()
    x.append(0)
    for elem in storeService[:-1]:  # for all elements up until the last one store them twice to make the step function
        x.extend([elem]*2)
    x.append(storeService[-1])  # add the last element
    y = list()
    y.append(0)  # first number is (zero, zero)
    cuml = 0  # stores cumulative
    for elem in storeService[:-1]:
        y.append(cuml)
        cuml += 1.0/n
        y.append(cuml)
    y.append(cuml)  # at time t you will have the last cumulative count
    # now do the integration
    integral = np.trapz(y,x)
        
    return integral



def evalEmpiricalDist(t, listOS):
    '''
    ---------------------
    evaluates the CDF at time t
    i.e., G(t)=P(S<=t)
    ---------------------
    :param t: time point
    :param listOS: list of ordered service times obtained from getOrderedService
    the service times in the list must be in units of slots
    :return cdf: cdf=G(t)
    ---------------------
    '''
    # get number of data points
    n = len(listOS)
    # first get the list of all the service times that are less than t in the window of interest
    storeService = list()
    key=0
    NotComplete = True
    while (NotComplete) and (listOS[key] <= t):
        storeService.append(listOS[key])
        key+=1
        if key not in np.arange(0, len(listOS), 1):  # check if index out of range
            NotComplete = False  # i.e., complete
    cdf = float(len(storeService))/n
    return cdf



def getOrderedService(dataDict, slotInMinutes):
    '''
    --------------------
    returns the list of service times ordered from smallest to largest
    this will be used to generate the empirical distribution
    
    the list is generated for an od pair and an
    upcoming horizon (as determined by dataDict)
    
    for each od pair, the output is 
    ordServiceTime=[least ser. time, ..., highest service time]
    --------------------
    :param dataDict: input dictionary with data entries of an od pair
    :param slotInMinutes: duration of the discretization slot
    :return ordServiceTime: ordered list of service times in increasing order 
    --------------------
    '''
    ordServiceTime = list()  # dictionary stores for every window soujourn time
    for key, In in enumerate(dataDict['Pickup_DateTime']):
        splitDateIn = In.split(' ')
        splitTimeIn = splitDateIn[1].split('-')[0].split(':')  # get the time you enter
        splitDateOut = dataDict['DropOff_datetime'][key].split(' ')
        splitTimeOut = splitDateOut[1].split('-')[0].split(':')  # get the time you leave
        hourDiff = float(splitTimeOut[0]) - float(splitTimeIn[0])
        minuteDiff = float(splitTimeOut[1]) - float(splitTimeIn[1])
        secondDiff = float(splitTimeOut[2]) - float(splitTimeIn[2])
        totalMinuteDiff = hourDiff*60 + minuteDiff + secondDiff*(1.0/60)  # total difference in minutes between entry and exit time
        totalMinuteDiffSlots = totalMinuteDiff / slotInMinutes  # get the service time in slots
        ordServiceTime.append(totalMinuteDiffSlots)  # add the difference in entry and exit to dictofLists based on entry location
    ordServiceTime.sort()
    return ordServiceTime



def getLambdaMLE(dataDict, slotInMinutes, windowInMinutes):
    '''
    -------------
    returns the maximum likelihood estimator of the Poisson arrival rate
    from the data
    ------------- 
    :param dataDict: input dictionary with data entries of an od pair
    for the upcoming time horizon
    :param slotInMinutes: duration of the discretization
    :param windowInMinutes: duration of the time horizon in minutes
    :return lambdaSlots: MLE arrival rate for the upcoming window in
    units of slots
    :return lambdaMin: MLE arrival rate for the upcoming window in
    units of minutes
    -------------
    '''
    totalArrivals= len(dataDict['Pickup_DateTime'])  # total number of arrivals for upcoming window
    lambdaMin = float(totalArrivals)/windowInMinutes  # rate in arrivals per minute
    numSlots = windowInMinutes/slotInMinutes  # number of slots in upcoming time horizon
    lambdaSlots = float(totalArrivals)/numSlots  # rate in arrivals per slot
    return lambdaSlots, lambdaMin
    


#------------------------------------------------------------------------




if __name__ == '__main__':
    # dDict, head = readCSV('data/ridesLyftMHTN14.csv')
    # dDict = addTimeStamp(dDict, slotInMinutes=5)
    print('utils')