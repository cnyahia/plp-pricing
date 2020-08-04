# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:34:11 2020

creates a base class that defines every origin-destination (od) pair
this class includes methods such as:
    1- gets number of starts for the od pair, previous, now, and future (p-n-f)
    2- gets the ends for each region associated with the od pair, p-n-f
    3- evaluates the empirical CDF associated with the od pair
    4- integrates the empirical CDF up to a specific time point

important:
    number of starts/ends and trip characteristics are computed at the end
    of time points
    for example:
    |t0|--- slot 1 -- |t1|-- slot 2 -- |t2| -- 
    the number of starts/ends are computed at t0, t1, t2, etc.

@author: cesny
"""
import copy as cp
import numpy as np
from utils import getEmpiricalIntegral, evalEmpiricalDist



class odpair:
    '''
    --a class for each origin-destination pair
    --the properties of this class are defined for every [u0,u1] (slot)
    and the upcoming window
    --the class contains methods for computing predicted num. of starts,
    maintaining the list of previously observed starts, predicted ends by
    destination 
    
    '''
    def __init__(self, origin, dest, slotPts, rate, window, ordSer):
        self.origin = origin  # the origin
        self.dest = dest
        self.slotPts = slotPts  # the timePts of the current pricing slot (u0,u1), |u0|--slot--|u1|
        self.rate = rate  # MLE rate for current time window
        self.window = window  # time horizon (first timePt, last timePt) tuple
        self.ordSer = list(ordSer)  # list of ordered service times for O-D pair
        self.obStarts = dict() # starts associated with prev. observed rides
        self.predStarts = dict()  # dict that will contain predicted starts for upcoming time window
        self.obEnds = dict()  # end associated with prev. observed rides 
        self.predEnds = dict()  # end that are predicted for the o-d pair
    
    
    def updateParams(self, rate, slotPts, window):
        '''
        updates the parameters of the class
        ----------
        :param rate: arrival rate
        :param slotPts: end points of new slot
        :param window: end points of time window
        ----------
        '''
        self.rate = rate
        self.slotPts = slotPts
        self.window = window
        return None
    
    
    def updateObsStarts(self, observedStarts):
        '''
        updates the starts dict by removing starts that terminated prior
        (i.e, not relevant anymore), and
        adding starts that were currently observed
        
        recall the the starts are cumulative (i.e., we are concerned with
        the total starts by time t) 
        --> thus, we have to subtract the starts by u1 from the upcoming time
        points since we are only concerned with the cumulative starts for
        time points in the window
        
        this method is called after the pricing happens and the users make 
        their choices, you update the starts before moving on to evaluating
        prices for next slot
        ------
        :param observedStarts: for the users that made the choice, at previous
        time slot, observedStarts contains the cumulative start by time points
        in the time window
        :return None: updates the obStarts dict using information from the 
        observed starts, deletes expired starts
        ------
        '''
        self.obStarts = dict()  # clear things up
        self.obStarts.update(observedStarts)
        return None
    
    
    def updateObsEnds(self, observedEnds):
        '''
        updates the ends dict by removing ends that terminated prior
        (i.e., not relevant anymore) and
        adding ends that were currently observed
        
        this method is called after the pricing happens and the users make 
        their choices, you update the ends before moving on to evaluating
        prices for next slot
        ------
        :param observedEnds: for the users that made the choice, at previous
        time slot, observedEnds contains the number of terminated trips by
        time point t
        :return None: updates the obEnds dict using information from the 
        observed ends, deletes expired ends
        ------
        '''
        self.obEnds = dict()  # clear things up
        self.obEnds.update(observedEnds)
        return None
    
    
    def createFutureStarts(self):
        '''
        creates the future starts for the next window, for each time point,
        we add the expected number of starts prior to that time point!
        '''
        self.predStarts=dict()  # clears things up
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.predStarts[timePt] = self.futureStart(timePt)
        return None
    
    
    def createFutureEnds(self):
        '''
        creates the future ends for the next window, for each time point,
        we add the expected number of ends that terminate prior to the time
        point
        '''
        self.predEnds=dict()  # clears things up
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.predEnds[timePt] = self.futureEnds(timePt)
        return None
    

    def futureStart(self, t):
        '''
        gets the number of future starts that have already started by time
        point t

        -----
        :param t: time point at which you evaluate expected future starts
        :return sf: expected number of future starts
        ------
        '''
        return self.rate*(t - self.slotPts[1])
    
    
    def futureEnds(self, timePt):
        '''
        expected number of ends that terminate by time point
        -------
        :param timePt: time point that we evaluate up to
        :return ef: expected number of future ends
        -------
        '''
        return self.rate*(self.intG(timePt - self.slotPts[1]))
    
    
    def now(self):
        '''
        computes expected number of arrivals in current slot
        
        this will be used later on to determine the additional number of 
        starts the occur before time t given that they are associated with
        ride requests that appear 'now' in the slot
        
        this will also be used later on to determine the additional number
        of ends that terminate prior to time t given that they are associated
        with ride requests that appear 'now' in the slot
        '''
        return self.rate*(self.slotPts[1] - self.slotPts[0])  
        
    
    def getProbEnd(self):
        '''
        computes Grr(t2-t1) which is P(S<t2-t1) to see if rides that started
        at t1 would have terminated by time t2
        
        specifically, for a certain time point \tau_{k}, we construct
        [Grr(\tau_{k}-\tau_{1}), Grr(\tau_{k}-\tau_{2}),..
         ,Grr(\tau_{k}-\tau_{k})]
        and we create a dict of lists that stores the above list for every
        \tau_{k} in the window
        '''
        Glists = dict()
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            Glists[timePt] = list()
            for prevtimePt in list(np.arange(self.window[0], timePt+1, 1)):  # up to and including the time point
                Glists[timePt].append(self.evalG(prevtimePt, timePt))
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            Glists[timePt] = [Glists[timePt]]  # put in list 
            Glists[timePt] = np.array(Glists[timePt])
            Glists[timePt] = Glists[timePt].T
        return Glists
    
    
    def evalG(self, t1, t2):
        '''
        evaluates the empirical service time distribution between two
        time points
        
        this will be used to evalaute the number of 'now' rides that terminate
        prior to some time point t2 given the users choose to depart at 
        time point t1
        ------
        :param t1: first time point
        :param t2: second time point
        :return G(t2-t1): where G(t2-t1)= P(ServiceTime<t2-t1)
        ------
        '''
        return evalEmpiricalDist(t2-t1, self.ordSer)
    
    
    def intG(self, timePt):
        '''
        integrates the service time distribution from
        zero up to the specified time point
        
        this will be used to determine the number of 'future' rides that
        terminate prior to a time point
        -------
        :param timePt: time point to integrate to
        :return int_{0}^{timePt}G:
        -------
        '''
        return getEmpiricalIntegral(timePt, self.ordSer)
        

