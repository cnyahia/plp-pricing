# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 03:52:53 2020

This module includes a class for modeling a region (zone)
This module:
    1- aggregate all the starts-ends (predicted or previously observed) from
    the origin-destination classes associated with this region
    2- implements the optimization for each region and determines the 
    optimal prices!

All methods are implemented at a specific slot and upcoming window!!
    To move across slots and aggregate results across time, check network.py!!
    
    
Note: the optimization problem is defined considering that
        the offered departure times are evenly distributed between 
        timePt[0] and timePt[1] of the window, and the duration between
        any two offered departure time is one slot
        
@author: cesny
"""

import copy as cp
import numpy as np
import cvxpy as cvx


class region:
    '''
    --a class for each region
    --the properties of this class are defined for every [u0,u1] (slot)
    and the upcoming window
    --the class contains methods for aggregating number of starts and ends
    by region
    --the class also contains methods for implementing the optimization 
    per region given past, now, and future of constituent o-d pairs
    
    '''
    def __init__(self, region, slotPts, window, odpairs):
        self.region = region
        self.slotPts = slotPts   # the timePts of the current pricing slot (u0,u1), |u0|--slot--|u1|
        self.window = window  # the end points of the window i.e. time horizon (first timePt, last timePt) tuple
        self.obStarts = dict() # starts associated with prev. observed rides
        self.predStarts = dict()  # dict that will contain predicted starts for upcoming time window
        self.obEnds = dict()  # end associated with prev. observed rides 
        self.predEnds = dict()  # end that are predicted for the region
        self.inOD = dict()  # a dictionary of the OD pairs that have the region as the destination
        self.outOD = dict()  # a dictionary of the OD pairs that have the region as the origin
        self.nowSt = 0  # total number of starts for users that appear 'now'
        self.nowE = 0  # total number of ends for users that appear 'now'
        self.Glists = dict()
        self.load = dict()  # the load process indicating change in cumulative starts and ends across time
        self.initializeODs(cp.deepcopy(odpairs))  # creates the ODs associated with the region (self.inOD, self.outOD)
        
    
    def updateParams(self, slotPts, window):
        '''
        updates parameters for new window
        ---------
        :param slotPts: end points of slot
        :param window: end points of window
        ---------
        '''
        self.slotPts = slotPts
        self.window = window
        return None
    
    
    def initializeODs(self, odpairs):
        '''
        for a dict of all odpairs and their classes, gets the od pairs
        that are relevant for the region, populate self.inOD, self.outODs
        ------------
        :param odpairs: {(1,1): class, (1,2): class, etc.}
        '''
        # re-zero
        self.inOD = dict()
        self.outOD = dict()
        # first initialize what's going out!
        for odp in odpairs:
            if self.region == odp[0]:
                self.outOD[odp] = odpairs[odp]
        # next initialize what's going in!
        for odp in odpairs:
            if self.region == odp[1]:
                self.inOD[odp] = odpairs[odp]
        return None
    
    
    def updateObsStarts(self):
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
        self.obStarts = dict()
        # initialize to the current window
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.obStarts[timePt] = 0
        # add the starts of the constituent od pairs
        for odp in self.outOD:  # for outgoing od-pair corresponding to the region
            for key in self.outOD[odp].obStarts:
                self.obStarts[key] = self.obStarts[key] + self.outOD[odp].obStarts[key]
        return None
    
    
    def updateObsEnds(self):
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
        # initialize to the current window
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.obEnds[timePt] = 0
        # add the ends of the constituent od pairs
        for odp in self.inOD:  # for incoming od-pair corresponding to the region
            for key in self.inOD[odp].obEnds:
                self.obEnds[key] = self.obEnds[key] + self.inOD[odp].obEnds[key]
        return None
    
    
    def createFutureStarts(self):
        '''
        creates the future starts for the next window, for each time point,
        we add the expected number of starts prior to that time point!
        '''
        self.predStarts=dict()  # clears things up
        # initialize to the current window
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.predStarts[timePt] = 0
        # add the future starts of the consituent pairs
        for odp in self.outOD:  # for outgoing od-pair corresponding to the region
            for key in self.outOD[odp].predStarts:
                self.predStarts[key] = self.predStarts[key] + self.outOD[odp].predStarts[key]
        return None
    
    
    def createFutureEnds(self):
        '''
        creates the future ends for the next window, for each time point,
        we add the expected number of ends that terminate prior to the time
        point
        '''
        self.predEnds=dict()  # clears things up
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.predEnds[timePt] = 0
        # add the ends of the constituent od pairs
        for odp in self.inOD:  # for incoming od-pair corresponding to the region
            for key in self.inOD[odp].predEnds:
                self.predEnds[key] = self.predEnds[key] + self.inOD[odp].predEnds[key]
        return None
    
    
    def loadProcess(self):
        '''
        generates the load process from starts and ends associated with
        'previously' observed rides in addition to starts and ends 
        associated with 'future' rides 
        '''
        self.load = dict()
        for timePt in list(np.arange(self.window[0], self.window[1]+1, 1)):
            self.load[timePt] = self.predStarts[timePt] + self.obStarts[timePt] - self.predEnds[timePt] - self.obEnds[timePt]
        return self.load, self.predStarts, self.obStarts, self.predEnds, self.obEnds
        
    
    def nowStart(self):
        '''
        gets the total expected number of starts for 'now' users
        note that this is just the total number, doesn't account
        for when do they choose to depart
        i.e. sum_{j\in R}\lambda_{rj}(u1-u0)
        '''
        self.nowSt = 0
        for odp in self.outOD:  # for outgoing od-pair corresponding to the region
            self.nowSt = self.nowSt + self.outOD[odp].now()
        return None
    
    
    def nowEnd(self):
        '''
        gets the total expected number of ends for 'now' users
        note that this is just the total number, doesn't account
        for when do they choose to depart
        \lambda_{rr}(u1-u0)
        also creates a dict of dicts that evaluates G(t-\tau_{k}) for each
        possible departure time and future time Point 
        '''
        self.nowE = 0
        for odp in self.inOD:
            if (self.region == odp[0]) and (self.region == odp[1]):
                self.nowE =  self.inOD[odp].now()
                self.Glists = self.inOD[odp].getProbEnd()
        return None
    
    
    def optimize(self, beta_c, beta_d, weight):
        '''
        creates the objective function of the optimization problem!
        '''
        n = self.window[1] - self.window[0] + 1
        p = cvx.Variable((n,1))
        z = cvx.Variable(1)
        d=np.array([list(np.arange(1, n, 1))]).T
        expr = (1.0/beta_c) * (cvx.sum(-1*cvx.entr(p[1:,[0]]) - beta_d*cvx.multiply(d,p[1:,[0]])  )  ) - (1.0/beta_c)*(cvx.log(p[0,[0]]) + cvx.entr(p[0,[0]]) ) + weight*z
        obj = cvx.Minimize(expr)
        constraints = [cvx.sum(p) == 1, p >= 0, p <= 1, z >= 0]
        for key, timePt in enumerate(list(np.arange(self.window[0], self.window[1], 1))):
            deltaPost = self.nowSt*cvx.sum( p[:key+1+1,[0]]) - self.nowE*cvx.sum(cvx.multiply(self.Glists[timePt+1], p[:key+1+1,[0]] )  ) 
            deltaPre = self.nowSt*cvx.sum( p[:key+1,[0]]) - self.nowE*cvx.sum(cvx.multiply(self.Glists[timePt], p[:key+1,[0]] )  ) 
            cexp = (self.load[timePt+1] +  deltaPost) - (self.load[timePt] + deltaPre)
            constraints = constraints + [cexp <= z]
        for key, timePt in enumerate(list(np.arange(self.window[0]+1, self.window[1]+1, 1))):
            cexp2 = p[key+1,[0]] - np.exp(beta_d*(key+1))*p[0,[0]]
            constraints = constraints + [cexp2 >= 0]
            
        prob = cvx.Problem(obj,constraints)
        prob.solve()
        
        return p.value, z.value, prob.status, prob.value

    
    