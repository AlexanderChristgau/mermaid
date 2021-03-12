"""
Simple (explicit) Runge-Kutta integrators to forward integrate dynamic forward models
"""
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod

import torch
from . import utils
import numpy as np
from future.utils import with_metaclass

class RKIntegrator(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for Runge-Kutta integration: x' = f(x(t),u(t),t)
    """

    def __init__(self,f,u,pars,params):
        """
        Constructor
        
        :param f: function to be integrated 
        :param u: input to the function
        :param pars: parameters to be passed to the integrator
        :param params: general ParameterDict() parameters for setting
        """

        self.nrOfTimeSteps_perUnitTimeInterval = params[('number_of_time_steps', 10,
                                                'Number of time-steps to per unit time-interval integrate the PDE')]
        """number of time steps for the integration"""
        self.f = f
        """Function to be integrated"""

        if pars is None:
            self.pars = []
        else:
            self.pars = pars
            """parameters for the integrator"""

        if u is None:
            self.u = lambda t,pars,vo: []
        else:
            self.u = u
            """input for integration"""

    def set_pars(self,pars):
        self.pars = pars

    def set_number_of_time_steps(self, nr):
        """
        Sets the number of time-steps per unit time-interval the integration
        
        :param nr: number of timesteps per unit time-interval
        """
        self.nrOfTimeSteps_perUnitTimeInterval = nr

    def get_number_of_time_steps(self):
        """
        Returns the number of time steps per unit time-interval that are used for the integration
        
        :return: number of time steps per unit time-interval
        """
        return self.nrOfTimeSteps_perUnitTimeInterval

    def get_dt(self):
        """
        Returns the time-step
        :return: timestep dt
        """

        dt = 1.0/self.nrOfTimeSteps_perUnitTimeInterval
        return dt

    def solve(self,x,fromT,toT,variables_from_optimizer=None):
        """
        Solves the differential equation.
        
        :param x: initial condition for state of the equation
        :param fromT: time to start integration from 
        :param toT: time to end integration
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: Returns state, x, at time toT
        """
        # arguments need to be list so we can pass multiple variables at the same time
        assert type(x)==list

        dT = toT-fromT
        nr_of_timepoints = int(round(self.nrOfTimeSteps_perUnitTimeInterval*dT))

        timepoints = np.linspace(fromT, toT, nr_of_timepoints + 1)
        dt = timepoints[1]-timepoints[0]
        currentT = fromT
        #iter = 0
        for i in range(0, nr_of_timepoints):
            #print('RKIter = ' + str( iter ) )
            #iter+=1
            x = self.solve_one_step(x, currentT, dt, variables_from_optimizer)
            currentT += dt
        #print( x )
        return x

    def _xpyts(self, x, y, v):
        # x plus y times scalar
        return [a+b*v for a,b in zip(x,y)]

    def _xts(self, x, v):
        # x times scalar
        return [a*v for a in x]

    def _xpy(self, x, y):
        return [a+b for a,b in zip(x,y)]

    @abstractmethod
    def solve_one_step(self, x, t, dt, variables_from_optimizer=None):
        """
        Abstract method to be implemented by the different Runge-Kutta methods to advance one step. 
        Both x and the output of f are expected to be lists 
        
        :param x: initial state 
        :param t: initial time
        :param dt: time increment
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: returns the state at t+dt 
        """
        pass

class EulerForward(RKIntegrator):
    """
    Euler-forward integration
    """

    def solve_one_step(self, x, t, dt, vo=None):
        """
        One step for Euler-forward
        
        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        #xp1 = [a+b*dt for a,b in zip(x,self.f(t,x,self.u(t)))]
        xp1 = self._xpyts(x, self.f(t, x, self.u(t, self.pars, vo), self.pars, vo), dt)
        return xp1

class RK4(RKIntegrator):
    """
    Runge-Kutta 4 integration
    """
    def debugging(self,input,t,k):
        x = utils.checkNan(input)
        if np.sum(x):
            print("find nan at {} step".format(t))
            print("flag m: {}, location k{}".format(x[0],k))
            print("flag phi: {}, location k{}".format(x[1],k))
            raise ValueError("nan error")

    def solve_one_step(self, x, t, dt, vo=None):
        """
        One step for Runge-Kutta 4
        
        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        k1 = self._xts(self.f(t, x, self.u(t, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k1,t,1)
        k2 = self._xts(self.f(t + 0.5 * dt, self._xpyts(x, k1, 0.5), self.u(t + 0.5 * dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k2, t, 2)
        k3 = self._xts(self.f(t + 0.5 * dt, self._xpyts(x, k2, 0.5), self.u(t + 0.5 * dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k3, t, 3)
        k4 = self._xts(self.f(t + dt, self._xpy(x, k3), self.u(t + dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k4, t, 4)

        # now iterate over all the elements of the list describing state x
        xp1 = []
        for i in range(len(x)):
            xp1.append( x[i] + k1[i]/6. + k2[i]/3. + k3[i]/3. + k4[i]/6. )

        return xp1


class SDEIntegrator(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for Runge-Kutta integration: x' = f(t,X_t)dt + g(t,X_t)odW_t
    """

    def __init__(self, f, g, pars, params):
        """
        Constructor

        :param f: function to be integrated
        :param u: input to the function
        :param pars: parameters to be passed to the integrator
        :param params: general ParameterDict() parameters for setting
        """

        self.nrOfTimeSteps_perUnitTimeInterval = params[('number_of_time_steps', 50,
                                                         'Number of time-steps to per unit time-interval integrate the PDE')]
        """number of time steps for the integration"""
        self.f = f
        self.g = g
        """Function to be integrated"""

        if pars is None:
            self.pars = []
        else:
            self.pars = pars
            """parameters for the integrator"""

    def set_pars(self, pars):
        self.pars = pars

    def set_number_of_time_steps(self, nr):
        """
        Sets the number of time-steps per unit time-interval the integration

        :param nr: number of timesteps per unit time-interval
        """
        self.nrOfTimeSteps_perUnitTimeInterval = nr

    def get_number_of_time_steps(self):
        """
        Returns the number of time steps per unit time-interval that are used for the integration

        :return: number of time steps per unit time-interval
        """
        return self.nrOfTimeSteps_perUnitTimeInterval

    def get_dt(self):
        """
        Returns the time-step
        :return: timestep dt
        """

        dt = 1.0 / self.nrOfTimeSteps_perUnitTimeInterval
        return dt

    def solve(self, x, fromT, toT, variables_from_optimizer=None):
        """
        Solves the differential equation.

        :param x: initial condition for state of the equation
        :param fromT: time to start integration from
        :param toT: time to end integration
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: Returns state, x, at time toT
        """
        # arguments need to be list so we can pass multiple variables at the same time
        assert type(x) == list

        dT = toT - fromT
        nr_of_timepoints = int(round(self.nrOfTimeSteps_perUnitTimeInterval * dT))

        timepoints = np.linspace(fromT, toT, nr_of_timepoints + 1)
        dt = timepoints[1] - timepoints[0]
        currentT = fromT

        dW = np.random.normal(0, np.sqrt(dt), nr_of_timepoints)
        # iter = 0
        for i in range(0, nr_of_timepoints):
            # print('RKIter = ' + str( iter ) )
            # iter+=1
            x = self.solve_one_step(x, currentT, dt, dW[i], variables_from_optimizer)
            currentT += dt
        # print( x )
        return x


    def _xts(self, x, v):
        # x times scalar
        return [a*v for a in x]

    def _xpy(self, x, y):
        return [a+b for a,b in zip(x,y)]

    def _xpyts(self, x, y, v):
        # x plus y times scalar
        return [a+b*v for a,b in zip(x,y)]

    def _xpytspzts(self, x, y, v, z, w):
        # x plus y times scalar plus z times scalar
        return [a + b * v + c * w for a, b, c in zip(x, y, z)]

    @abstractmethod
    def solve_one_step(self, x, t, dt, dW, variables_from_optimizer=None):
        """
        Abstract method to be implemented by the different Runge-Kutta methods to advance one step.
        Both x and the output of f are expected to be lists

        :param x: initial state
        :param t: initial time
        :param dt: time increment
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: returns the state at t+dt
        """
        pass

class EulerMaruyama(SDEIntegrator):
    """
    EulerMaruyama-forward integration:
    x(t + dt) = x(t) + f(x(t))dt + g(x(t))dW
    """

    def solve_one_step(self, x, t, dt, dW, vo=None):
        """
        One step for Euler-forward

        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        xp1 = self._xpytspzts(x, self.f(t, x, self.pars, vo), dt, self.g(t,x,self.pars,vo),dW)
        return xp1

class EulerHeun(SDEIntegrator):
    """
    EulerHeun-forward integration:
    x(t + dt) = x(t) + f(x(t))dt + g(x(t))dW
    """

    def solve_one_step(self, x, t, dt, dW, vo=None):
        """
        One step for Euler-forward

        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        gn = g(t,x,self.pars,vo)
        gbar = g(t,self._xpyts(x,gn,dW),self.parts,vo)

        xp1 = self._xpytspzts(x, self.f(t, x, self.pars, vo), dt,
                              self._xpy(gn,gbar),0.5 * dW)
        return xp1

class MultiSDEIntegrator(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for multidimensional drift SDE's:
    dX_t = f(t,X_t)dt + g_a(X_t) o dW_t^a
    """

    def __init__(self, f, g, pars, params):
        """
        Constructor

        :param f: function to be integrated
        :param u: input to the function
        :param pars: parameters to be passed to the integrator
        :param params: general ParameterDict() parameters for setting
        """

        self.nrOfTimeSteps_perUnitTimeInterval = params[('number_of_time_steps', 50,
                                                         'Number of time-steps to per unit time-interval integrate the PDE')]
        """number of time steps for the integration"""
        self.f = f
        self.g = g
        """Function to be integrated"""

        if pars is None:
            self.pars = []
        else:
            self.pars = pars
            self.n_sigma = pars["n_sigma"]
            """parameters for the integrator"""

    def set_pars(self, pars):
        self.pars = pars

    def set_number_of_time_steps(self, nr):
        """
        Sets the number of time-steps per unit time-interval the integration

        :param nr: number of timesteps per unit time-interval
        """
        self.nrOfTimeSteps_perUnitTimeInterval = nr

    def get_number_of_time_steps(self):
        """
        Returns the number of time steps per unit time-interval that are used for the integration

        :return: number of time steps per unit time-interval
        """
        return self.nrOfTimeSteps_perUnitTimeInterval

    def get_dt(self):
        """
        Returns the time-step
        :return: timestep dt
        """

        dt = 1.0 / self.nrOfTimeSteps_perUnitTimeInterval
        return dt

    def solve(self, x, fromT, toT, variables_from_optimizer=None):
        """
        Solves the differential equation.

        :param x: initial condition for state of the equation
        :param fromT: time to start integration from
        :param toT: time to end integration
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: Returns state, x, at time toT
        """
        # arguments need to be list so we can pass multiple variables at the same time
        assert type(x) == list

        dT = toT - fromT
        nr_of_timepoints = int(round(self.nrOfTimeSteps_perUnitTimeInterval * dT))

        timepoints = np.linspace(fromT, toT, nr_of_timepoints + 1)
        dt = timepoints[1] - timepoints[0]
        currentT = fromT

        normals = np.random.normal(0, np.sqrt(dt), nr_of_timepoints * self.n_sigma)
        W = torch.Tensor((normals).reshape(nr_of_timepoints, self.n_sigma))
        W = torch.unsqueeze(W, dim=2)
        W = torch.unsqueeze(W, dim=2)
        W = torch.unsqueeze(W, dim=2)
        # iter = 0
        for i in range(0, nr_of_timepoints):
            # print('RKIter = ' + str( iter ) )
            # iter+=1
            x = self.solve_one_step(x, currentT, dt, W[i], variables_from_optimizer)
            currentT += dt
        # print( x )
        return x


    def _xts(self, x, v):
        # x times scalar
        return [a*v for a in x]

    def _xpy(self, x, y):
        return [a+b for a,b in zip(x,y)]

    def _xpyts(self, x, y, v):
        # x plus y times scalar
        return [a+b*v for a,b in zip(x,y)]

    def _xpytspz(self, x, y, v, z):
        # x plus y times scalar plus z
        return [a + b * v + c for a, b, c in zip(x, y, z)]

    def _xpytspzts(self, x, y, v, z, w):
        # x plus y times scalar plus z times a scaler
        return [a + b * v + c*w for a, b, c in zip(x, y, z)]

    def _xpytspzpw(self, x, y, v, z, w):
        return [a + b * v + c + d for a, b, c, d in zip(x, y, z, w)]



    def _sumBtdW(self,x,W):
        return [torch.sum(b * W,dim=0,keepdim=True) for b in x]

    @abstractmethod
    def solve_one_step(self, x, t, dt, dW, variables_from_optimizer=None):
        """
        Abstract method to be implemented by the different Runge-Kutta methods to advance one step.
        Both x and the output of f are expected to be lists

        :param x: initial state
        :param t: initial time
        :param dt: time increment
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: returns the state at t+dt
        """
        pass


class MultiEulerMaruyama(MultiSDEIntegrator):
    """
    EulerMaruyama-forward integration:
    x(t + dt) = x(t) + f(x(t))dt + g_a(x(t))dW_t^a
    """

    def solve_one_step(self, x, t, dt, dW, vo=None):
        """
        One step for Euler-forward

        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        noise = self._sumBtdW(self.g(t,x,self.pars,vo), dW)
        xp1 = self._xpytspz(x, self.f(t, x, self.pars, vo), dt,noise)
        return xp1


class Milstein(MultiSDEIntegrator):
    """
    Derivative free Milstein forward integration of:
    x(t + dt) = x(t) + f(x(t))dt + g_a(x(t))dW_t^a
    """

    def solve_one_step(self, x, t, dt, dW, vo=None):
        """
        One step for Euler-forward

        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        g_val = self.g(t,x,self.pars,vo)
        f_val = self.f(t,x,self.pars,vo)
        x_bar = self._xpytspzts(x,f_val,dt,g_val,dt**0.5)
        g_bar = self.g(t,x_bar,self.pars,vo)
        noise1 = self._sumBtdW(g_val, dW)
        noise2 = self._sumBtdW(self._xpyts(g_bar,g_val,-1),torch.square(dW) /(2*(dt**0.5)))
        xp1 = self._xpytspzpw(x,f_val,dt,noise1,noise2)
        return xp1