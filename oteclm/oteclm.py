import numpy as np
import openturns as ot
import openturns.viewer as otView
import math as math
from time import time
import tqdm

class ECLM(object):
    r"""
    Provides a class for the Extended Common Load Model (ECLM).

    Parameters
    ---------- 
    totalImpactVector : :class:`~openturns.Indices`
        The total impact vector of the common cause failure (CCF) group.
    integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
        The integration algorithm used to compute the integrals.

    Notes
    -----
    The Extended Common Load Model (ECLM) is detailed in *Extended Common Load Model: a tool for dependent failure modelling in highly redundant structures*, T. Mankamo, Report number: 2017:11, ISSN: 2000-0456, available at www.stralsakerhetsmyndigheten.se, 2017.

    We consider a common cause failure (CCF) group of :math:`n` components supposed to be independent and identical.

    We denote by :math:`S` the load and :math:`R_i` the resistance of the component :math:`i` of the CCF group. We assume that :math:`(R_1, \dots, R_n)` are independent and identically distributed according to :math:`R`.

    We assume that the load :math:`S` is modelled as a mixture of two normal distributions:

        - the base part  with mean and variance :math:`(\mu_b, \sigma_b^2)`
        - the extreme load parts  with mean and variance :math:`(\mu_x, \sigma_x^2)`
        - the respective weights :math:`(\pi, 1-\pi)`.

    Then the density of :math:`S` is written as:

    .. math::

        f_S(s) = \pi \dfrac{1}{\sigma_b} \varphi \left(\dfrac{s-\mu_b}{\sigma_b}\right) + (1-\pi) \dfrac{1}{\sigma_x} \varphi \left(\dfrac{s-\mu_x}{\sigma_x}\right)\quad \forall s \in \mathbb{R}

    We assume that the resistance :math:`R` is modelled as a normal distribution with  mean and variance :math:`(\mu_R, \sigma_R^2)`. We denote by :math:`p_R` and :math:`F_R` its density and its cumulative density function.

    We define the ECLM probabilities of a CCF group of size :math:`n`, for :math:`0 \leq k \leq n` are the following:

        - :math:`\mathrm{PSG}(k|n)`: probability that a  specific set of :math:`k` components fail.
        - :math:`\mathrm{PEG}(k|n)`: probability that a specific set of :math:`k` components fail  while the other :math:`(n-k)` survive.
        - :math:`\mathrm{PES}(k|n)`: probability that some set of :math:`k` components fail while the other :math:`(n-k)` survive.
        - :math:`\mathrm{PTS}(k|n)`: probability that  at least some specific set of :math:`k` components fail.


    Then the  :math:`\mathrm{PEG}(k|n)`  probabilities are defined as:

    .. math::

        \begin{array}{rcl}
          \mathrm{PEG}(k|n) & = & \mathbb{P}\left[S>R_1, \dots, S>R_k, S<R_{k+1}, \dots, S<R_n\right] \\
                          & = &  \int_{s\in  \mathbb{R}} f_S(s) \left[F_R(s)\right]^k \left[1-F_R(s)\right]^{n-k} \, ds
        \end{array}

    and the  :math:`\mathrm{PSG}(k|n)` probabilities are defined as:

    .. math::

        \begin{array}{rcl}
          \mathrm{PSG}(k|n) & = & \mathbb{P}\left[S>R_1, \dots, S>R_k\right]\\
                            & = &  \int_{s\in  \mathbb{R}} f_S(s) \left[F_R(s)\right]^k\, ds
        \end{array}

    We get the :math:`\mathrm{PES}(k|n)` probabilities and  :math:`\mathrm{PTS}(k|n)` with the relations:

        .. math::
            :label: PES_red

            \mathrm{PES}(k|n) = C_n^k \, \mathrm{PEG}(k|n)

    and

        .. math::
            :label: PTS_red

             \mathrm{PTS}(k|n) = \sum_{i=k}^n  \mathrm{PES}(i|n)


    We use the following set of parameters called *general parameter* :

    .. math::
        :label: generalParam

        \vect{\theta} = (\pi, d_b, d_x, d_R, y_{xm})

    defined by:

    .. math::

        \begin{array}{rcl}
         d_{b} & = & \dfrac{\sigma_b}{\mu_R-\mu_b}\\
         d_{x} & = & \dfrac{\sigma_x}{\mu_R-\mu_b}\\
         d_{R} & = & \dfrac{\sigma_R}{\mu_R-\mu_b}\\
         y_{xm} & = & \dfrac{\mu_x-\mu_b}{\mu_R-\mu_b}
        \end{array}

    Then the  :math:`\mathrm{PEG}(k|n)` probabilities are written as:

    .. math::
        :label: PEG_red

             \mathrm{PEG}(k|n)  =    \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right] \left[\Phi\left(\dfrac{y-1}{d_R}\right)\right]^k \left[1-\Phi\left(\dfrac{y-1}{d_R}\right)\right]^{n-k} \, dy

    And the  :math:`\mathrm{PSG}(k|n)` probabilities are written as:

    .. math::
        :label: PSG_red

        \mathrm{PSG}(k|n)  =    \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right] \left[\Phi\left(\dfrac{y-1}{d_R}\right)\right]^k  \, dy

    Note that for :math:`k=1`, the integral can be computed explicitly:

    .. math::
        :label: PSG1_red

        \begin{array}{lcl}
           \mathrm{PSG}(1|n) & = & \displaystyle \int_{-\infty}^{+\infty}\left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) \right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy +  \int_{-\infty}^{+\infty} \left[  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy \\
                 & = & \pi \left[1-\Phi\left(\dfrac{1}{\sqrt{d_b^2+d_R^2}}\right)\right] +  (1-\pi) \left[1-\Phi\left(\dfrac{1-y_{xm}}{\sqrt{d_x^2+d_R^2}}\right)\right]
        \end{array}

    The computation of the :math:`\mathrm{PEG}(k|n)`  and :math:`\mathrm{PSG}(k|n)` probabilities is done with a quadrature method provided at the creation of the class. We advice the :class:`~openturns.GaussLegendre` quadrature with 50 points.


    **The probabilistic model:**

    We denote by  :math:`N^n` the random variable that counts the number of failure events in the CCF group under one test or demand. Then the range of :math:`N^n` is :math:`[0, n]` and its probability law is :math:`\mathbb{P}\left[N^n=k\right] =  \mathrm{PES}(k|n)`.

    Under :math:`N` test or demands, we denote by :math:`N^{n,N}_t` the random variable that counts the number of times when :math:`k` failure events have occured, for each :math:`k` in :math:`[0, n]`:

    .. math::
        :label: NnNt

        N^{n,N}_t = \sum_{k=1}^N N^n_k

    where the random variables :math:`(N^n_1, \dots, N^n_N)` are independent and identically distributed as :math:`N^n`. The :math:`N^{n,N}_t` follows a Multinomial distribution parameterized by :math:`(N,(p_0, \dots, p_n))` with:

    .. math::

        (p_0, \dots, p_n) =  (\mathrm{PES}(0|n), \dots,  \mathrm{PES}(n|n))


    **The data:**

    The data is the total impact vector :math:`V_t^{n,N}` of the CCF group. The component :math:`V_t^{n,N}[k]` for :math:`0 \leq k \leq n` is the number of failure events of multiplicity :math:`k` in the CCF group. In addition, :math:`N` is the number of tests and demands on the whole group. Then we have :math:`N = \sum_{k=0}^n V_t^{n,N}[k]`.

    Then :math:`V_t^{n,N}` is a realization of the random variable :math:`N^{n,N}_t`.

    **Data likelihood:**

    The log-likelihood of the model is defined by:

    .. math::

        \log \mathcal{L}(\vect{\theta}|V_t^{n,N})  =  \sum_{k=0}^n V_t^{n,N}[k] \log \mathrm{PES}(k|n)


    The optimal parameter :math:`\vect{\theta}` maximises the log-likelihoodand is defined as:

    .. math::
        :label: optimGen

        \vect{\theta}_{optim}   =   \arg \max_{\vect{\theta}} \log \mathcal{L}(\vect{\theta}|V_t^{n,N})

    Remark: As we have :eq:`PES_red`, then the log-likelihood can be written as:

    .. math::

         \log \cL(\vect{\theta}_t|V_t^{n,N})   = \sum_{k=0}^n V_t^{n,N}[k] \log   C_n^k  +  \sum_{k=0}^n V_t^{n,N}[k] \log \mathrm{PEG}(k|n)

    Noting that the first term does not depend on the parameter :math:`\vect{\theta}`, then we also have:

    .. math::
        :label: optimGenReduced

        \vect{\theta}_{optim}  =  \arg \max_{\vect{\theta}} \sum_{k=0}^n V_t^{n,N}[k] \log \mathrm{PEG}(k|n)


    **Mankamo method:**

    Mankamo introduces a new set of parameters:  :math:`(P_t, P_x, C_{co}, C_x, y_{xm})` defined from the general parameter :eq:`generalParam` as follows:

    .. math::
        :label: Param2

        \begin{array}{rcl}
             P_t & = & \mathrm{PSG}(1|n) = \displaystyle  \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy \\
             P_x &  = &\displaystyle  \int_{-\infty}^{+\infty} \left[  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy = (1-\pi) \left[1-\Phi\left(\dfrac{1-y_{xm}}{\sqrt{d_x^2+d_R^2}}\right)\right]\\
             c_{co} & = & \dfrac{d_b^2}{d_b^2+d_R^2}\\
             c_x & = & \dfrac{d_x^2}{d_x^2+d_R^2}
        \end{array}

    Mankamo assumes that:

    .. math::
        :label: mankamoHyp

        y_{xm} = 1-d_R


    This assumption means that :math:`\mu_R = \mu_x+\sigma_R`. Then equations :eq:`Param2` simplifies and we get:

    .. math::
        :label: Param2to1Mankamo

        \begin{array}{rcl}
            (1-\pi) & = & -\dfrac{P_x}{\Phi\left(\sqrt{1-c_{x}}\right)}\\
            d_b & = & -\dfrac{\sqrt{c_{co}}}{\Phi^{-1}\left(\dfrac{P_t-P_x}{\pi} \right)}\\
            d_R  & = & -\dfrac{\sqrt{1-c_{co}}}{\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)} \\
            d_x  & = & d_R \sqrt{\dfrac{c_{x}}{1-c_{x}}}
        \end{array}

    We call *Mankamo parameter* the set:

    .. math::
        :label: MankamoParam

        (P_t, P_x, C_{co}, C_x)

    The parameter :math:`P_t` is directly estimated from the total impact vector:

    .. math::
        :label: eqPt

         \hat{P}_t = \sum_{i=1}^n\dfrac{iV_t^{n,N}[i]}{nN}

    Then the optimal :math:`(P_x, C_{co}, C_x)` maximises the log-likelihood of the model and then the expression:

    .. math::
        :label: optimMankamo

         (P_x, C_{co}, C_x)_{optim}  = \arg \max_{(P_x, C_{co}, C_x)}  \sum_{k=0}^n V_t^{n,N}[k] \log \mathrm{PEG}(k|n)

    The optimization is done under the following constraints:

    .. math::

        \begin{array}{l}
             0 \leq P_x  \leq P_t \\
             0 \leq c_{co} \leq c_{x} \leq 1 \\
             0 \leq 1-\pi = \dfrac{P_x}{\Phi\left(- \sqrt{1-c_{x}}\right)}\leq 1 \\
             0 \leq d_b = -\dfrac{\sqrt{c_{co}}}{\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)} \\
              0 \leq  d_R   = -\dfrac{\sqrt{1-c_{co}}}{\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)}
        \end{array}

    Assuming that :math:`P_t \leq \frac{1}{2}`, we can write the constraints as:

    .. math::
        :label: MankamoConstraints

        \begin{array}{l}
            0\leq  P_t \leq \dfrac{1}{2}\\
            0 \leq P_x  \leq \min \left \{P_t, \left (P_t-\dfrac{1}{2}\right ) \left(
            1-\dfrac{1}{2\Phi\left(-\sqrt{1-c_{x}}\right)}\right)^{-1},  \Phi\left(- \sqrt{1-c_{x}}\right) \right \} \\
            0 \leq c_{co} \leq c_{x} \leq 1 
        \end{array}
    """

    
    def __init__(self, totalImpactVector, integrationAlgo=ot.GaussKronrod(), nIntervals=8, verbose=False):
        # set attribute
        self.totalImpactVector = totalImpactVector
        self.integrationAlgo = integrationAlgo
        self.nIntervals = nIntervals
        self.n = self.totalImpactVector.getSize()-1
        self.verbose = verbose
        # Mankamo Param: (P_t, P_x, C_{co}, C_x)
        self.MankamoParameter = None
        # GeneralParam: (pi, d_b, d_x, d_R, y_{xm})
        self.generalParameter = None

        # Estimateur de Pt
        N = sum(self.totalImpactVector)
        Pt = 0.0
        for i in range(1,self.n+1):
            Pt += i*self.totalImpactVector[i]
        Pt /= self.n*N
        self.Pt = Pt
    
        # global parameter
        self.eps = 1e-9

        # Definition domain of the log-likelihood
        formula  = "var terme1 := (Cx >= 1 ? 0.5 - 5e-16 : 0.5 - 0.5 * erf(sqrt(1.0 - Cx) / sqrt(2)));"
        formula += "var terme2 := " + str(Pt - 0.5) + " / (1.0 - 1.0 / (2.0 * terme1));"
        formula += "var Px := exp(logPx);"
        formula += "out0 := " + str(Pt) + " - Px;";
        formula += "out1 := terme1 - Px;"
        formula += "out2 := terme2 - Px;"
        formula += "out3 := Cx;"
        formula += "out4 := 1.0 - Cx;"
        formula += "out5 := Cco;"
        formula += "out6 := 1.0 - Cco;"
        formula += "out7 := Cx - Cco;"
        self.MankamoConstraints = ot.SymbolicFunction(["logPx", "Cco", "Cx"], ["out" + str(i) for i in range(8)], formula)

        # Cache for the PEG
        self.PEGAll = ot.Point(self.n + 1, -1.0)

        # Cache for the PSG
        self.PSGAll = ot.Point(self.n + 1, -1.0)

    def verifyMankamoConstraints(self, X):
        r"""
        Verifies if the point :math:`(P_x, C_{co}, C_x)` verifies the constraints.

        Parameters
        ----------
        inPoint : :class:`~openturns.Point`
            The point :math:`(P_x, C_{co}, C_x)`

        Returns
        -------
        testRes : bool
            True if the point verifies the constraints.

        Notes
        -----
        The constraints are defined in :eq:`MankamoConstraints`  under the Mankamo assumption :eq:`mankamoHyp`. 
        """

        if X[0] <= 0.0:
            return False

        # MankamoConstraints takes log(Px) as the first input component
        x = ot.Point(X)
        x[0] = math.log(X[0])
        res = self.MankamoConstraints(x)
        value = True
        for r in res:
            value = value and (r > 0.0)
        return value

    def computeValidMankamoStartingPoint(self, Cx):
        r"""
        Gives a point :math:`(P_x, C_{co})` given :math:`C_x` and :math:`P_t` verifying the constraints.

        Parameters
        ----------
        Cx : float, :math:`0 < C_x < 1`
            The parameter :math:`C_x`.

        Returns
        -------
        validPoint : :class:`~openturns.Point`
            A valid point  :math:`(P_x, C_{co}, C_x)` verifying the constraints.

        Notes
        -----
        The constraints are defined in :eq:`MankamoConstraints`  under the Mankamo assumption :eq:`mankamoHyp`. The parameter :math:`P_t` is computed from the total impact vector as :eq:`eqPt`. For a given :math:`C_x`, we give the range of possible values for :math:`P_x` and :math:`C_{co}` and we propose a valid point :math:`(P_x, C_{co}, C_x)`.
        """
        
        if Cx < self.eps or Cx > 1.0-self.eps:
            raise('Problem with Cx not in [0, 1] or too close to the bounds!')
        if self.Pt > 0.5-self.eps:
            raise('Problem with Pt not < 0.5 or to close to 0.5!')
        
        terme1 = ot.DistFunc.pNormal(-math.sqrt(1 - Cx))
        terme2 = (self.Pt - 0.5) / (1 - 1 / (2 * terme1))
        terme_min = min(self.Pt, terme1, terme2)

        # X respects the constraints if:
        # Constraint 1 :  Px < terme_min 
        # Constraint 2 : Cx > Cco
        if self.verbose: 
            print('Px must be less than ', terme_min)
            print('Cco must be less than ',  Cx)

        proposedPoint = ot.Point([terme_min / 2, Cx / 2, Cx])
        return proposedPoint


    def logVrais_Mankamo(self, X):
        logPx, Cco, Cx = X

        # X is not in the definition domain of PEG
        if not self.verifyMankamoConstraints((math.exp(logPx), Cco, Cx)):
            if self.verbose:
                print("X=", ot.Point(X), "does not satisfy the Mankamo constraints")
            return [-ot.SpecFunc.LogMaxScalar]

        
        # variables (pi, db, dx, dR, y_xm=1-dR)
        pi_weight, db, dx, dR, y_xm = self.computeGeneralParamFromMankamo([self.Pt, math.exp(logPx), Cco, Cx])
        self.setGeneralParameter([pi_weight, db, dx, dR, y_xm])
        S = 0.0
        for k in range(self.n+1):
            valImpactvect = self.totalImpactVector[k]
            if valImpactvect != 0:
                val = self.computePEG(k)
                log_PEG_k = math.log(val)
                S += self.totalImpactVector[k] * log_PEG_k
        return [S]
    

    def estimateMaxLikelihoodFromMankamo(self, startingPoint, visuLikelihood=False):
        r"""
        Estimates the maximum likelihood (general and Mankamo) parameters under the Mankamo assumption.

        Parameters
        ----------
        startingPoint : :class:`~openturns.Point`
            Starting point :math:`(P_x, C_{co}, C_x)` for the optimization problem.
        visuLikelihood : Bool
            Produces the graph of the log-likelihood function at the optimal point.
            Default value is False.

        Returns
        -------
        paramList : :class:`~openturns.Point`
            The optimal point :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})` where :math:`y_{xm} = 1-d_R`.
        finalLogLikValue : float
            The value of the reduced log-likelihood function :eq:`optimGenReduced` at the optimal point.
        graphList : list of :class:`~openturns.Graph`
            The collection of graphs drawing the log-likelihood function at the optimal point when one or two components are fixed.

        Notes
        -----
        If the starting point is not valid, we computes a valid one witht the function *computeValidMankamoStartingPoint* at the point :math:`c_x = 0.7`. 
        """

        maFctLogVrais_Mankamo = ot.PythonFunction(3, 1, self.logVrais_Mankamo)

        ######################################
        # Maximisation de la vraisemblance

        optimPb = ot.OptimizationProblem(maFctLogVrais_Mankamo)
        optimPb.setMinimization(False)
        # contraintes sur (Cco, Cx): maFct_cont >= 0
        optimPb.setInequalityConstraint(self.MankamoConstraints)
        # bounds sur (logPx, Cco, Cx)
        boundsParam = ot.Interval([-35, self.eps, self.eps], [math.log(self.Pt), 1.0-self.eps, 1.0-self.eps])
        if self.verbose:
            print('boundsParam = ', boundsParam)
        optimPb.setBounds(boundsParam)
        # algo Cobyla pour ne pas avoir les gradients
        myAlgo = ot.Cobyla(optimPb)
        myAlgo.setVerbose(self.verbose)
        if self.verbose:
            ot.Log.Show(ot.Log.ALL)
        #myAlgo.setIgnoreFailure(True)
        myAlgo.setRhoBeg(0.1)
        myAlgo.setMaximumEvaluationNumber(10000)
        myAlgo.setMaximumConstraintError(1e-4)
        myAlgo.setMaximumAbsoluteError(1e-4)
        myAlgo.setMaximumRelativeError(1e-3)
        myAlgo.setMaximumResidualError(1e-4)

        # Point de départ:
        # startingPoint = [Px, Cco, Cx]
        if not self.verifyMankamoConstraints(startingPoint):        
            startingPoint = self.computeValidMankamoStartingPoint(0.7)
            if self.verbose:
                print('Changed starting point : ', startingPoint)
        # startPoint = [logPx, Cco, Cx]
        startPoint = [math.log(startingPoint[0]), startingPoint[1], startingPoint[2]]
        myAlgo.setStartingPoint(startPoint)

        myAlgo.run()

        ######################################
        # Parametrage optimal
        myOptimalPoint = myAlgo.getResult().getOptimalPoint()
        finalLogLikValue = myAlgo.getResult().getOptimalValue()[0]

        logPx_optim = myOptimalPoint[0]
        Px_optim = math.exp(logPx_optim)
        Cco_optim = myOptimalPoint[1]
        Cx_optim = myOptimalPoint[2]

        # Mankamo parameter
        mankamoParam = [self.Pt, Px_optim, Cco_optim, Cx_optim]
        self.setMankamoParameter(mankamoParam)
        # General parameter = (pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim)
        # ==> mis à jour par setMankamoParameter
        generalParam = self.getGeneralParameter()

        ######################################################
        # Pour avoir les graphes de Mankamo au point de Mankamo
        #mankamoParam = [self.Pt, 5.7e-7, 0.51, 0.85]
        #self.setMankamoParameter(mankamoParam)
        #logPx_optim = math.log(5.7e-7)
        #Px_optim = 5.7e-7
        #Cco_optim = 0.51
        #Cx_optim = 0.85
        ######################################################
        
        ######################################
        # Graphes de la log vraisemblance avec point optimal     
        
        g_fixedlogPxCco, g_fixedlogPxCx, g_fixedCcoCx, g_fixedCx, g_fixedCco, g_fixedlogPx = [None]*6

        if visuLikelihood:
            
            def func_contraint_LogPx_Cx(X):
                logPx, Cco, Cx = X
                # pour dessiner la contrainte suivante 
                # (logPx, Cx) --> min(terme1, terme2, terme3) - logPx

                terme1 = ot.DistFunc.pNormal(-math.sqrt(1 - Cx))
                terme2 = (self.Pt - 0.5) / (1 - 1 / (2 * terme1))
                terme3 = self.Pt

                # return un Point
                return [min(math.log(terme1), math.log(terme2), math.log(terme3))-logPx]


            # juste pour les besoins des graphes
            ma_Fct_cont_LogPx_Cx = ot.PythonFunction(3, 1, func_contraint_LogPx_Cx)
            maFct_cont_Cco_Cx = ot.SymbolicFunction(["logPx", "Cco", "Cx"], ["Cx - Cco"])
        
            print('Production of graphs')
            maFct_cont_Cco_Cx_fixedlogPx = ot.ParametricFunction(maFct_cont_Cco_Cx, [0], [logPx_optim])
            
            ma_Fct_cont_LogPx_Cx_fixedCco = ot.ParametricFunction(ma_Fct_cont_LogPx_Cx, [1], [Cco_optim])
            ma_Fct_cont_LogPx_Cx_fixedCcoCx = ot.ParametricFunction(ma_Fct_cont_LogPx_Cx, [1,2], [Cco_optim, Cx_optim])
            ma_Fct_cont_LogPx_Cx_fixedlogPxCco = ot.ParametricFunction(ma_Fct_cont_LogPx_Cx, [0,1], [logPx_optim, Cco_optim])

            maFctLogVrais_Mankamo_fixedlogPx =ot. ParametricFunction(maFctLogVrais_Mankamo, [0], [logPx_optim])
            maFctLogVrais_Mankamo_fixedCco = ot.ParametricFunction(maFctLogVrais_Mankamo, [1], [Cco_optim])
            maFctLogVrais_Mankamo_fixedCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [2], [Cx_optim])
            maFctLogVrais_Mankamo_fixedlogPxCco = ot.ParametricFunction(maFctLogVrais_Mankamo, [0,1], [logPx_optim, Cco_optim])
            maFctLogVrais_Mankamo_fixedCcoCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [1,2], [Cco_optim, Cx_optim])
            maFctLogVrais_Mankamo_fixedlogPxCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [0,2], [logPx_optim, Cx_optim])

            coef = 0.3
            # Care! logPx_optim <0!
            logPx_inf = (1+coef)*logPx_optim
            logPx_sup = (1-coef)*logPx_optim

            #Cco_inf = max((1-coef)*Cco_optim, 0.01)
            Cco_inf = (1-coef)*Cco_optim
            Cco_sup = min((1+coef)*Cco_optim, 0.99)
            
            Cx_inf = max((1-coef)*Cx_optim, 1.05*Cco_optim)
            Cx_sup = min((1+coef)*Cx_optim, 0.99)

            NbPt = 64

            ####################
            # graphe (logPx) pour Cco = Cco_optim et Cx = Cx_optim
            # graphe de la loglikelihood
            print('graph (Cco, Cx) = (Cco_optim, Cx_optim)')
            g_fixedCcoCx = maFctLogVrais_Mankamo_fixedCcoCx.draw(logPx_inf, logPx_sup, 2*NbPt)
            
            # + contrainte sur logPx
            limSup_logPx = ma_Fct_cont_LogPx_Cx_fixedCcoCx([logPx_optim])[0] + logPx_optim
            minValGraph = g_fixedCcoCx.getDrawable(0).getData().getMin()[1]
            maxValGraph = g_fixedCcoCx.getDrawable(0).getData().getMax()[1]
            lineConstraint = ot.Curve([limSup_logPx,limSup_logPx], [minValGraph, maxValGraph], r'$\log P_x \leq f(C_x^{optim})$')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setColor('black')
            g_fixedCcoCx.add(lineConstraint)
            
            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, maFctLogVrais_Mankamo_fixedCcoCx([logPx_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCcoCx.add(myCloud)
            g_fixedCcoCx.setXTitle(r'$\log P_x$')
            g_fixedCcoCx.setYTitle(r'$\log \, \mathcal{L}$')
            g_fixedCcoCx.setTitle(r'Log likelihood at $(C_{co}, C_{x}) = ($'+ format(Cco_optim,'.2E') + ',' +  format(Cx_optim,'.2E') + ')')
            g_fixedCcoCx.setLegendPosition('bottomleft')
            #view = otView.View(g_fixedCcoCx)
            #view.show()
            
            ####################
            # graphe (Cco) pour log Px = log Px_optim et Cx = Cx_optim
            # graphe de la loglikelihood
            print('graph (logPx, Cx) = (logPx_optim, Cx_optim)')
            g_fixedlogPxCx = maFctLogVrais_Mankamo_fixedlogPxCx.draw(Cco_inf, Cco_sup, 2*NbPt)
            # + contrainte sur Cco< Cx
            minValGraph = g_fixedlogPxCx.getDrawable(0).getData().getMin()[1]
            maxValGraph = g_fixedlogPxCx.getDrawable(0).getData().getMax()[1]
            lineConstraint = ot.Curve([Cx_optim, Cx_optim], [minValGraph, maxValGraph], r'$C_{co} \leq C_x^{optim}$')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setColor('black')
            g_fixedlogPxCx.add(lineConstraint)
            
            
            # + point optimal 
            pointOptim = ot.Sample(1, [Cco_optim, maFctLogVrais_Mankamo_fixedlogPxCx([Cco_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPxCx.add(myCloud)
            g_fixedlogPxCx.setXTitle(r'$C_{co}$')
            g_fixedlogPxCx.setYTitle(r'$\log \, \mathcal{L}$')
            g_fixedlogPxCx.setTitle(r'Log likelihood at $(\log P_{x}, C_{x}) = ($'+ format(logPx_optim,'.2E') + ',' +  format(Cx_optim,'.2E') + ')')
            g_fixedlogPxCx.setLegendPosition('topright')
            #view = otView.View(g_fixedlogPxCx)
            #view.show()

            ####################
            # graphe (Cx) pour logPx = logPx_optim et Cco = Cco_optim
            # graphe de la loglikelihood
            print('graph (logPx, Cco) = (logPx_optim, Cco_optim)')
            g_fixedlogPxCco = maFctLogVrais_Mankamo_fixedlogPxCco.draw(Cx_inf, Cx_sup, 2*NbPt)
            # contrainte Cx > Cco
            minValGraph = g_fixedlogPxCco.getDrawable(0).getData().getMin()[1]
            maxValGraph = g_fixedlogPxCco.getDrawable(0).getData().getMax()[1]
            lineConstraint = ot.Curve([Cco_optim, Cco_optim], [minValGraph, maxValGraph], r'$C_{co}^{optim} \leq C_x$')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setColor('black')
            g_fixedlogPxCco.add(lineConstraint)
            
            # + point optimal 
            pointOptim = ot.Sample(1, [Cx_optim, maFctLogVrais_Mankamo_fixedlogPxCco([Cx_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPxCco.add(myCloud)
            g_fixedlogPxCco.setXTitle(r'$C_x$')
            g_fixedlogPxCco.setYTitle(r'$\log \, \mathcal{L}$')
            g_fixedlogPxCco.setTitle(r'Log likelihood at $(\log P_{x}, C_{co}) = ($'+ format(logPx_optim,'.2E') + ',' +  format(Cco_optim,'.2E') + ')')
            g_fixedlogPxCco.setLegendPosition('bottomright')
            #g_fixedlogPxCco.setBoundingBox(ot.Interval([0.9, maFctLogVrais_Mankamo_fixedlogPxCco([0.9])[0]], [0.99, maFctLogVrais_Mankamo_fixedlogPxCco([0.99])[0]]))

            #view = otView.View(g_fixedlogPxCco)
            #view.show()
                        
            ####################
            # graphe (Px, Cco) pour Cx = Cx_optim
            print('graph Cx = Cx_optim')
            # graphe de la loglikelihood
            g_fixedCx = maFctLogVrais_Mankamo_fixedCx.draw([logPx_inf, Cco_inf], [logPx_sup, Cco_sup], [NbPt]*2)            
            # contrainte  Cx > Cco
            lineConstraint = ot.Curve([logPx_inf, logPx_sup], [Cx_optim, Cx_optim],  r'$C_{co} \leq C_x^{optim}$')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setLineWidth(2)
            lineConstraint.setColor('black')
            g_fixedCx.add(lineConstraint)
            # contrainte sur logPx et Cx
            dr =  ma_Fct_cont_LogPx_Cx_fixedCco.draw([logPx_inf, Cco_inf], [logPx_sup, Cco_sup], [NbPt]*2).getDrawable(0)
            dr.setLevels([0.0])
            #dr.setLegend(r'$\log P_x \leq f(C_x)$')
            dr.setLegend('constraint')
            dr.setLineStyle('dashed')
            dr.setColor('black')
            g_fixedCx.add(dr)
                  
            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, Cco_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCx.add(myCloud)
            g_fixedCx.setXTitle(r'$\log P_x$')
            g_fixedCx.setYTitle(r'$C_{co}$')
            g_fixedCx.setTitle(r'Log likelihood at $C_{x} = $'+ format(Cx_optim,'.2E'))
            g_fixedCx.setLegendPosition('topleft')
            
            #view = otView.View(g_fixedCx)
            #view.show()     

            
            ####################
            # graphe (logPx, Cx) pour Cco = Cco_optim
            print('graph Cco = Cco_optim')
            # graphe de la loglikelihood
            g_fixedCco = maFctLogVrais_Mankamo_fixedCco.draw([logPx_inf, Cx_inf], [logPx_sup, Cx_sup], [NbPt]*2)
            # contrainte  Cx > Cco
            #lineConstraint = ot.Curve([logPx_inf, logPx_sup], [Cco_optim, Cco_optim], r'$C_{co}^{optim} \leq C_x$')
            lineConstraint = ot.Curve([logPx_inf, logPx_sup], [Cco_optim, Cco_optim], 'constraint')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setColor('black')
            lineConstraint.setLineWidth(2)
            g_fixedCco.add(lineConstraint)
            # contrainte sur logPx et Cx
            dr = ma_Fct_cont_LogPx_Cx_fixedCco.draw([logPx_inf, Cx_inf], [logPx_sup, Cx_sup], [NbPt]*2).getDrawable(0)
            dr.setLevels([0.0])
            #dr.setLegend(r'$\log P_x \leq f(C_x)$')
            dr.setLegend('constraint')
            dr.setLineStyle('dashed')
            dr.setColor('black')
            g_fixedCco.add(dr)                  
            
            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, Cx_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCco.add(myCloud)
            g_fixedCco.setXTitle(r'$\log P_x$')
            g_fixedCco.setYTitle(r'$C_{x}$')
            g_fixedCco.setTitle(r'Log likelihood at $C_{co} = $'+ format(Cco_optim,'.2E'))
            g_fixedCco.setLegendPosition('bottomleft')

            ####################
            # graphe (Cco, Cx) pour logPx = logPx_optim
            print('graph logPx = logPx_optim')
            # niveau de la loglikelihood
            g_fixedlogPx = maFctLogVrais_Mankamo_fixedlogPx.draw([Cco_inf, Cx_inf], [Cco_sup, Cx_sup], [NbPt]*2)
            # contrainte  Cx > Cco
            lineConstraint = ot.Curve([Cco_inf, Cco_sup], [Cx_inf, Cx_sup], r'$C_{co} \leq C_x$')
            lineConstraint.setLineStyle('dashed')
            lineConstraint.setColor('black')
            lineConstraint.setLineWidth(2)
            g_fixedlogPx.add(lineConstraint)
            # contrainte sur logPx et Cx ==> pas possible de le dessiner! Introduire une nvelle fonction
            # + point optimal
            pointOptim = ot.Sample(1, [Cco_optim, Cx_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPx.add(myCloud)
            g_fixedlogPx.setXTitle(r'$C_{co}$')
            g_fixedlogPx.setYTitle(r'$C_{x}$')
            g_fixedlogPx.setTitle(r'Log likelihood at $\log P_{x} = $'+ format(logPx_optim,'.2E'))
            g_fixedlogPx.setLegendPosition('bottomright')
            
            #view = otView.View(g_fixedlogPx)
            #view.show()     

        return mankamoParam, generalParam, finalLogLikValue, [g_fixedlogPxCco, g_fixedlogPxCx, g_fixedCcoCx, g_fixedCx, g_fixedCco, g_fixedlogPx]

    def computeGeneralParamFromMankamo(self, mankamoParam):
        r"""
        Computes the general parameter from the Mankamo parameter  under the Mankamo assumption.

        Parameters
        ----------
        mankamoParam :  list of float
            The  Mankamo parameter :eq:`MankamoParam`.

        Returns
        -------
        generalParam : list of float
            The general parameter :eq:`generalParam`.

        Notes
        -----
        The general parameter :eq:`generalParam` is computed from the Mankamo parameter :eq:`MankamoParam` under the Mankamo assumption :eq:`mankamoHyp` using equations :eq:`Param2to1Mankamo`.
        """

        Pt, Px, Cco, Cx = mankamoParam
        pi_weight = 1.0 - Px / ot.DistFunc.pNormal(-math.sqrt(1.0 - Cx))
        db = -math.sqrt(Cco) / ot.DistFunc.qNormal((Pt - Px) / pi_weight)
        dR = -math.sqrt(1.0 -Cco) / ot.DistFunc.qNormal((Pt - Px) / pi_weight)
        dx = dR * math.sqrt(Cx / (1.0 - Cx))
        yxm = 1.0 - dR

        return [pi_weight, db, dx, dR, yxm]


    def computePEG(self, k):
        r"""
        Computes the :math:`\mathrm{PEG}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Multiplicity of the common cause failure event.

        Returns
        -------
        peg_kn : float,  :math:`0 \leq  \mathrm{PEG}(k|n) \leq 1`
            The :math:`\mathrm{PEG}(k|n)` probability.

        Notes
        -----
        The  :math:`\mathrm{PEG}(k|n)` probability is computed using :eq:`PEG_red`.
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')

        if self.PEGAll[k] != -1.0:
            return self.PEGAll[k]
        
        pi_weight, db, dx, dR, y_xm = self.generalParameter

        # Numerical range of the  Normal() distribution
        val_min = -7.65
        val_max =  7.65

        # Numerical integration interval
        # base load
        yMin_b = val_min * db
        yMax_b = val_max * db

        # extreme load
        yMin_x = val_min * dx + y_xm
        yMax_x = val_max * dx + y_xm

        inputs   = ["y"]
        outputs  = ["z"]
        preamble  = "var yt := (y - 1.0) / " + str(dR) + ";"
        preamble += "var erf_yt := 0.5 * erf(yt / sqrt(2.0));"
        # Kernel b
        # If 0<k<n, the Phi^k(1-Phi)^{n-k} term is zero if either Phi=0 or Phi=1
        # If k=0, the Phi^k(1-Phi)^{n-k} term is zero if Phi=1
        # If k=n, the Phi^k(1-Phi)^{n-k} term is zero if Phi=0
        factor = ""
        if k > 0:
            factor += " * (erf_yt > -0.5 ? (0.5 + erf_yt)^" + str(k) + " : 0.0)"
        if k < self.n:
            factor += " * (erf_yt < 0.5 ? (0.5 - erf_yt)^" + str(self.n - k) + " : 0.0)"
        formula = preamble + "var phib := " + str(1.0 / (db * math.sqrt(2.0 * math.pi))) + " * exp(-0.5 * y^2 / " + str(db * db) + ");"
        formula += "z := phib"
        formula += factor + ";"
        maFctKernelB = ot.SymbolicFunction(inputs, outputs, formula)

        # Kernel X
        formula = preamble + "var phix := " + str(1.0 / (dx * math.sqrt(2.0 * math.pi))) + " * exp(-0.5 * (y - " + str(y_xm) + ")^2 / " + str(dx * dx) + ");"
        formula += "z := phix"
        formula += factor + ";"
        maFctKernelX = ot.SymbolicFunction(inputs, outputs, formula)

        # base load part integration
        int_b = 0.0
        if yMin_b < yMax_b:
            for i in range(self.nIntervals):
                yMin_i = yMin_b + i * (yMax_b - yMin_b) / self.nIntervals
                yMax_i = yMin_b + (i + 1) * (yMax_b - yMin_b) / self.nIntervals
                interval = ot.Interval(yMin_i, yMax_i)
                int_b += self.integrationAlgo.integrate(maFctKernelB, interval)[0]
            int_b = pi_weight * int_b
        # extreme load part integration
        int_x = 0.0
        if yMin_x < yMax_x:
            for i in range(self.nIntervals):
                yMin_i = yMin_x + i * (yMax_x - yMin_x) / self.nIntervals
                yMax_i = yMin_x + (i + 1) * (yMax_x - yMin_x) / self.nIntervals
                interval = ot.Interval(yMin_i, yMax_i)
                int_x += self.integrationAlgo.integrate(maFctKernelX, ot.Interval(yMin_i, yMax_i))[0]
            int_x = (1-pi_weight) * int_x

        PEG = int_b + int_x

        self.PEGAll[k] = PEG

        return PEG


    def computePEGall(self):
        r"""
        Computes all the :math:`\mathrm{PEG}(k|n)` probabilities for :math:`0 \leq k \leq n`.

        Returns
        -------
        peg_list : seq of float,  :math:`0 \leq  \mathrm{PEG}(k|n) \leq 1`
            The :math:`\mathrm{PEG}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mathrm{PEG}(k|n)` probabilities are computed using :eq:`PEG_red`.
        """

        PEG_list = list()
        for k in range(self.n+1):
            PEG_list.append(self.computePEG(k))
        return PEG_list


    def computePSG1(self):
        r"""
        Computes the :math:`\mathrm{PSG}(1|n)` probability.

        Returns
        -------
        psg_1n : float,  :math:`0 \leq  \mathrm{PSG}(1|n) \leq 1`
            The :math:`\mathrm{PSG}(1|n)`  probability.

        Notes
        -----
        The  :math:`\mathrm{PSG}(1|n)` probability is computed using :eq:`PSG1_red`.
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')

        pi_weight, db, dx, dR, y_xm = self.generalParameter

        # PSG(1|n) = Pb + Px
        val_b = math.sqrt(db * db + dR * dR)
        val_x = math.sqrt(dx * dx + dR * dR)
        Pb = pi_weight * ot.DistFunc.pNormal(-1.0 / val_b)
        Px = (1.0 - pi_weight) * ot.DistFunc.pNormal(-(1.0 - y_xm) / val_x)
        return Pb + Px


    def computePSG(self, k):
        r"""
        Computes the :math:`\mathrm{PSG}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Multiplicity of the common cause failure event.

        Returns
        -------
        psg_kn : float,  :math:`0 \leq  \mathrm{PSG}(k|n) \leq 1`
            The :math:`\mathrm{PSG}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mathrm{PSG}(k|n)` probability is computed using :eq:`PSG_red` for :math:`k !=1` and using :eq:`PSG1_red` for :math:`k=1`.
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')

        if self.PSGAll[k] != -1.0:
            return self.PSGAll[k]

        pi_weight, db, dx, dR, y_xm = self.generalParameter

        # special computation on that case
        if k == 0:
            return 1.0

        # special computation on that case
        if k == 1:
            return self.computePSG1()

        # Numerical range of the Normal() distribution
        val_min = -7.65
        val_max =  7.65

        # Numerical integration interval
        # base load
        yMin_b = val_min * db
        yMax_b = val_max * db

        # extreme load
        yMin_x = val_min * dx + y_xm
        yMax_x = val_max * dx + y_xm

        inputs   = ["y"]
        outputs  = ["z"]
        preamble  = "var yt := (y - 1.0) / " + str(dR) + ";"
        preamble += "var erf_yt := 0.5 * erf(yt / sqrt(2.0));"
        # Kernel b
        # If 0<k<n, the Phi^k(1-Phi)^{n-k} term is zero if either Phi=0 or Phi=1
        # If k=0, the Phi^k(1-Phi)^{n-k} term is zero if Phi=1
        # If k=n, the Phi^k(1-Phi)^{n-k} term is zero if Phi=0
        factor = ""
        if k > 0:
            factor += " * (erf_yt > -0.5 ? (0.5 + erf_yt)^" + str(k) + " : 0.0)"
        formula = preamble + "var phib := " + str(1.0 / (db * math.sqrt(2.0 * math.pi))) + " * exp(-0.5 * y^2 / " + str(db * db) + ");"
        formula += "z := phib"
        formula += factor + ";"
        maFctKernelB = ot.SymbolicFunction(inputs, outputs, formula)

        # Kernel X
        formula = preamble + "var phix := " + str(1.0 / (dx * math.sqrt(2.0 * math.pi))) + " * exp(-0.5 * (y - " + str(y_xm) + ")^2 / " + str(dx * dx) + ");"
        formula += "z := phix"
        formula += factor + ";"
        maFctKernelX = ot.SymbolicFunction(inputs, outputs, formula)

        # base load part integration
        int_b = 0.0
        if yMin_b < yMax_b:
            for i in range(self.nIntervals):
                yMin_i = yMin_b + i * (yMax_b - yMin_b) / self.nIntervals
                yMax_i = yMin_b + (i + 1) * (yMax_b - yMin_b) / self.nIntervals
                interval = ot.Interval(yMin_i, yMax_i)
                int_b += self.integrationAlgo.integrate(maFctKernelB, interval)[0]
            int_b = pi_weight * int_b

        # extreme load part integration
        int_x = 0.0
        if yMin_x < yMax_x:
            for i in range(self.nIntervals):
                yMin_i = yMin_x + i * (yMax_x - yMin_x) / self.nIntervals
                yMax_i = yMin_x + (i + 1) * (yMax_x - yMin_x) / self.nIntervals
                interval = ot.Interval(yMin_i, yMax_i)
                int_x += self.integrationAlgo.integrate(maFctKernelX, ot.Interval(yMin_i, yMax_i))[0]
            int_x = (1-pi_weight) * int_x

        PSG = int_b + int_x

        self.PSGAll[k] = PSG

        return PSG


    def computePSGall(self):
        r"""
        Computes all the :math:`\mathrm{PSG}(k|n)` probabilities for :math:`0 \leq k \leq n`.

        Returns
        -------
        psg_list : seq of float,  :math:`0 \leq  \mathrm{PSG}(k|n) \leq 1`
            The :math:`\mathrm{PSG}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mathrm{PSG}(k|n)` probabilities are computed using :eq:`PSG_red` for :math:`k != 1` and :eq:`PSG1_red` for :math:`k = 1`.
        """

        PSG_list = list()
        for k in range(self.n + 1):
            PSG_list.append(self.computePSG(k))
        return PSG_list


    def computePES(self, k):
        r"""
        Computes the :math:`\mathrm{PES}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Multiplicity of the failure event.

        Returns
        -------
        pes_kn : float,  :math:`0 \leq  \mathrm{PES}(k|n) \leq 1`
            The :math:`\mathrm{PES}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mathrm{PES}(k|n)` probability is computed using  :eq:`PES_red`.
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')

        pi_weight, db, dx, dR, y_xm = self.generalParameter

        PEG = self.computePEG(k)
        PES = math.comb(self.n, k) * PEG
        return PES


    def computePESall(self):
        r"""
        Computes all the :math:`\mathrm{PES}(k|n)` probabilities for :math:`0 \leq k \leq n`.

        Returns
        -------
        pes_list : seq of float,  :math:`0 \leq  \mathrm{PES}(k|n) \leq 1`
            The :math:`\mathrm{PES}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mathrm{PES}(k|n)` probabilities are computed using :eq:`PES_red`.
        """

        PES_list = list()
        for k in range(self.n + 1):
            PES_list.append(self.computePES(k))
        return PES_list


    def computePTS(self, k):
        r"""
        Computes the :math:`\mathrm{PTS}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Multiplicity of the common cause failure event.

        Returns
        -------
        pts_kn : float, :math:`0 \leq \mathrm{PTS}(k|n) \leq 1`
            The :math:`\mathrm{PTS}(k|n)` probability.

        Notes
        -----
        The :math:`\mathrm{PTS}(k|n)` probability is computed using :eq:`PTS_red`  where the :math:`\mathrm{PES}(i|n)` probability is computed using :eq:`PES_red`.
        """

        if k == 0:
            return 1.0
        
        PTS = 0.0
        for i in range(k,self.n + 1):
            PTS += self.computePES(i)
        return PTS


    def  computePTSall(self):
        r"""
        Computes all the :math:`\mathrm{PTS}(k|n)` probabilities for :math:`0 \leq k \leq n`.

        Returns
        -------
        pts_list : seq of float,  :math:`0 \leq  \mathrm{PTS}(k|n) \leq 1`
            The :math:`\mathrm{PTS}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mathrm{PTS}(k|n)` probabilities are computed using :eq:`PSG_red`.
        """

        PTS_list = list()
        for k in range(self.n + 1):
            PTS_list.append(self.computePTS(k))
        return PTS_list


    def jobEstimateBootstrapParamSampleFromMankamo(self, inP):
        point, startingPoint = inP
        vectImpactTotal = ot.Indices([int(round(x)) for x in point])
        self.setTotalImpactVector(vectImpactTotal)
        res = self.estimateMaxLikelihoodFromMankamo(startingPoint, False)
        resMankamo = res[0]
        resGeneral = res[1]
        resFinal = resMankamo + resGeneral
        return resFinal


    def estimateBootstrapParamSampleFromMankamo(self, Nbootstrap, startingPoint, fileNameRes, blockSize=256):
        r"""
        Generates a Bootstrap sample of the (Mankamo and general) parameters under the Mankamo assumption.

        Parameters
        ----------
        Nbootstrap : int
            The size of the sample generated.
        startingPoint : list of float,
            Mankamo starting point :eq:`MankamoParam` for the optimization problem.
        fileNameRes: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})` under the Mankamo assumption :eq:`mankamoHyp`.
        blockSize : int,
            The block size after which the sample is saved. Default value is 256.

        Notes
        -----
        The Mankamo parameter sample is obtained by bootstraping the empirical law of the total impact vector :math:`N_b` times. The total empirical impact vector follows the distribution MultiNomial parameterized by the empirical probabilities :math:`[p_0^{emp},\dots, p_n^{emp}]` where :math:`p_k^{emp} = \dfrac{V_t^{n,N}[k]}{N}` and :math:`N` is the number of tests and demands on the whole group. Then the optimisation problem :eq:`optimMankamo` is solved using the specified starting point.

        The function generates a script *script_bootstrap_ParamFromMankamo.py* that uses the parallelisation of the pool object of the multiprocessing module. It also creates a file *myECLM.xml* that stores the total impact vector to be read by the script. Both files are removed at the end of the execution of the method.

        The computation is saved in the csv file named *fileNameRes* every blockSize calculus. The computation can be interrupted: it will be restarted from the last *filenameRes* saved.
        """
        myStudy = ot.Study('myECLM.xml')
        myStudy.add('integrationAlgo', self.integrationAlgo)
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.add('nIntervalsAsIndices', ot.Indices(1, self.nIntervals))
        myStudy.add('startingPoint', ot.Point(startingPoint))
        myStudy.save()

        import os
        directory_path = os.getcwd()
        fileName = directory_path + "/script_bootstrap_ParamFromMankamo.py"
        if os.path.exists(fileName):
            os.remove(fileName)
        with open(fileName, "w") as f:
            f.write("#############################\n"\
"# Ce script :\n"\
"#    - lance un bootstrap sur la loi Multinomiale paramétrée par le vecteur d'impact total initial sous l'hypothèse de Mankamo\n"\
"#     - calcule les estimateurs de max de vraisemblance:\n"\
"#       de [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim] avec  yxm_optim = 1-dR_optim\n"\
"#     - sauve le sample de dimension 9 dans filenameRes: de type adresse/fichier.csv\n"\
"#    - le fichier fileNameInput contient les arguments: vectImpactTotal, startingPoint: de type adresse/fichier.xml\n"\
"\n"\
"\n"\
"import openturns as ot\n"\
"from oteclm import ECLM\n"\
"\n"\
"from time import time\n"\
"import sys\n"\
"\n"\
"from multiprocessing import Pool\n"\
"# barre de progression\n"\
"import tqdm\n"\
"\n"\
"# Ot Parallelisme desactivated\n"\
"ot.TBB.Disable()\n"\
"\n"\
"# Nombre d'échantillons bootstrap à générer\n"\
"Nbootstrap = int(sys.argv[1])\n"\
"# taille des blocs\n"\
"blockSize = int(sys.argv[2])\n"\
"# Nom du fichier csv qui contiendra le sample des paramètres (P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})\n"\
"fileNameRes = str(sys.argv[3])\n"\
"\n"\
"print('boostrap param : ')\n"\
"print('Nbootstrap, blockSize, fileNameRes  = ', Nbootstrap, blockSize, fileNameRes )\n"\
"\n"\
"\n"\
"# Import de vectImpactTotal, startingPoint et nIntervals\n"\
"myStudy = ot.Study('myECLM.xml')\n"\
"myStudy.load()\n"\
"integrationAlgo = ot.IntegrationAlgorithm()\n"\
"myStudy.fillObject('integrationAlgo', integrationAlgo)\n"\
"totalImpactVector = ot.Indices()\n"\
"myStudy.fillObject('totalImpactVector', totalImpactVector)\n"\
"nIntervalsAsIndices = ot.Indices()\n"\
"myStudy.fillObject('nIntervalsAsIndices', nIntervalsAsIndices)\n"\
"startingPoint = ot.Point()\n"\
"myStudy.fillObject('startingPoint', startingPoint)\n"\
"\n"\
"myECLM = ECLM(totalImpactVector, integrationAlgo, nIntervalsAsIndices[0])\n"\
"\n"\
"# Sollicitations number\n"\
"N = sum(totalImpactVector)\n"\
"\n"\
"# Empirical distribution of the number of sets of failures among N sollicitations\n"\
"MultiNomDist = ot.Multinomial(N, [v/N for v in totalImpactVector])\n"\
"\n"\
"def job(inP):\n"\
"    point, startingPoint, index = inP\n"\
"    vectImpactTotal = ot.Indices([int(round(x)) for x in point])\n"\
"    myECLM.setTotalImpactVector(vectImpactTotal)\n"\
"    res = myECLM.estimateMaxLikelihoodFromMankamo(startingPoint, False)\n"\
"    resMankamo = res[0]\n"\
"    resGeneral = res[1]\n"\
"    resFinal = resMankamo + resGeneral\n"\
"    return resFinal, index\n"\
"\n"\
"Ndone = 0\n"\
"block = 0\n"\
"t00 = time()\n"\
"\n"\
"\n"\
"# the dimension of res is 9\n"\
"allResults = ot.Sample(Nbootstrap, 9)\n"\
"#  Si des calculs ont déjà été faits, on les importe:\n"\
"try:\n"\
"    print('[ParamFromMankano] Try to import previous results from {}'.format(fileNameRes))\n"\
"    allResultsDone = ot.Sample.ImportFromCSVFile(fileNameRes)\n"\
"    allResults[0:Ndone] = allResultsDone\n"\
"    Ndone = len(allResultsDone)\n"\
"    print('import ok')\n"\
"except:\n"\
"    print('No previous results')\n"\
"\n"\
"allResults.setDescription(['Pt', 'Px', 'Cco', 'Cx', 'pi', 'db', 'dx', 'dR', 'yxm'])\n"\
"\n"\
"# On passe les Nskip points déjà calculés (pas de pb si Nskip=0)\n"\
"print('Skip = ', Ndone)\n"\
"for i in range(Ndone):\n"\
"    noMatter = MultiNomDist.getRealization()\n"\
"while Ndone < Nbootstrap:\n"\
"    block += 1\n"\
"     # Nombre de calculs qui restent à faire\n"\
"    size = min(blockSize, Nbootstrap - Ndone)\n"\
"    print('Generate bootstrap data, block=', block, 'size=', size, 'Ndone=', Ndone, 'over', Nbootstrap)\n"\
"    t0 = time()\n"\
"    allInputs = [[MultiNomDist.getRealization(), startingPoint, Ndone + i] for i in range(size)]\n"\
"    t1 = time()\n"\
"    print('t=%g' % (t1 - t0), 's')\n"\
"\n"\
"    t0 = time()\n"\
"    with Pool() as pool:\n"\
"        # Calcul parallèle: pas d'ordre, retourné dès que réalisé\n"\
"        res = list(tqdm.tqdm(pool.imap_unordered(job, allInputs, chunksize=1), total=len(allInputs)))\n"\
"        for r in res:\n"\
"            allResults[r[1]] = r[0]\n"\
"    t1 = time()\n"\
"    print('t=%g' % (t1 - t0), 's', 't (start)=%g' %(t1 - t00), 's')\n"\
"    Ndone += size\n"\
"    # Sauvegarde apres chaque bloc\n"\
"    allResults.exportToCSVFile(fileNameRes)\n"\
"   ")

        command =  'python script_bootstrap_ParamFromMankamo.py {} {} {}'.format(Nbootstrap, blockSize, fileNameRes)
        os.system(command)
        os.remove(fileName)
        os.remove("myECLM.xml")


    def computeECLMProbabilitiesFromMankano(self, fileNameInput, fileNameRes, blockSize=256):
        r"""
        Computes the sample of all the ECLM probabilities from a sample of Mankamo parameters using the Mankamo assumption.

        Parameters
        ----------
        fileNameInput: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.
        fileNameRes: string,
            The csv file that stores the ECLM probabilities.
        blockSize : int,
            The block size after which the sample is saved. Default value is 256.

        Notes
        -----
        The ECLM probabilities are computed using the Mankamo assumption :eq:`mankamoHyp`. They are returned according to the order :math:`(\mathrm{PEG}(0|n), \dots, \mathrm{PEG}(n|n), \mathrm{PSG}(0|n), \dots, \mathrm{PSG}(n|n), \mathrm{PES}(0|n), \dots, \mathrm{PES}(n|n), \mathrm{PTS}(0|n), \dots, \mathrm{PTS}(n|n))` using equations :eq:`PEG_red`, :eq:`PSG_red`, :eq:`PES_red`, :eq:`PTS_red`, using the Mankamo assumption :eq:`mankamoHyp`.

        The function generates the script *script_bootstrap_ECLMProbabilities.py* that uses the parallelisation of the pool object of the multiprocessing module.  It also creates a file *myECLM.xml* that stores the total impact vector to be read by the script. Both files are removed at the end of the execution of the method.

        The computation is saved in the csv file named *fileNameRes* every blockSize calculus. The computation can be interrupted: it will be restarted from the last *filenameRes* saved.
        """

        myStudy = ot.Study('myECLM.xml')
        myStudy.add('integrationAlgo', self.integrationAlgo)
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.add('nIntervalsAsIndices', ot.Indices(1, self.nIntervals))
        myStudy.save()

        import os
        directory_path = os.getcwd()
        fileName = directory_path + "/script_bootstrap_ECLMProbabilities.py"
        if os.path.exists(fileName):
            os.remove(fileName)
        with open(fileName, "w") as f:
            f.write("#############################\n"\
"# Ce script :\n"\
"#    - récupère un échantillon de (Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim] dans le fichier fileNameInput (de type adresse/fichier.csv)\n"\
"#     - calcule toutes les grandeurs probabilistes de l'ECLM\n"\
"#     - sauve le sample des grandeurs  probabilistes dans  le fichier fileNameRes (de type adresse/fichier.csv)\n"\
"\n"\
"\n"\
"\n"\
"import openturns as ot\n"\
"from oteclm import ECLM\n"\
"\n"\
"from signal import signal, SIGPIPE, SIG_DFL\n"\
"signal(SIGPIPE, SIG_DFL)\n"\
"\n"\
"from time import time\n"\
"import sys\n"\
"\n"\
"from multiprocessing import Pool\n"\
"# barre de progression\n"\
"import tqdm\n"\
"\n"\
"# Ot Parallelisme desactivated\n"\
"ot.TBB.Disable()\n"\
"\n"\
"\n"\
"# Nombre de grappes\n"\
"n = int(sys.argv[1])\n"\
"# taille des blocs\n"\
"blockSize = int(sys.argv[2])\n"\
"# Nom du fichier csv contenant le sample des paramètres (P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})\n"\
"fileNameInput = str(sys.argv[3])\n"\
"# Nom du fichier de sauvegarde: de type adresse/fichier.csv\n"\
"fileNameRes = str(sys.argv[4])\n"\
"\n"\
"print('ECLM prob')\n"\
"print('n, fileNameInput, fileNameRes = ', n, fileNameInput, fileNameRes)\n"\
"\n"\
"# Import de  vectImpactTotal et startingPoint\n"\
"myStudy = ot.Study('myECLM.xml')\n"\
"myStudy.load()\n"\
"integrationAlgo = ot.IntegrationAlgorithm()\n"\
"myStudy.fillObject('integrationAlgo', integrationAlgo)\n"\
"totalImpactVector = ot.Indices()\n"\
"myStudy.fillObject('totalImpactVector', totalImpactVector)\n"\
"nIntervalsAsIndices = ot.Indices()\n"\
"myStudy.fillObject('nIntervalsAsIndices', nIntervalsAsIndices)\n"\
"\n"\
"myECLM = ECLM(totalImpactVector, integrationAlgo, nIntervalsAsIndices[0])\n"\
"\n"\
"\n"\
"def job(inP):\n"\
"    # inP = [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]\n"\
"    myECLM.setGeneralParameter(inP[4:9])\n"\
"    PEG_list = myECLM.computePEGall()\n"\
"    PSG_list = myECLM.computePSGall()\n"\
"    PES_list = myECLM.computePESall()\n"\
"    PTS_list = myECLM.computePTSall()\n"\
"    res_list = list()\n"\
"    res_list += PEG_list\n"\
"    res_list += PSG_list\n"\
"    res_list += PES_list\n"\
"    res_list += PTS_list\n"\
"    return res_list\n"\
"\n"\
"\n"\
"Ndone = 0\n"\
"block = 0\n"\
"t00 = time()\n"\
"\n"\
"\n"\
"# Import de l'échantillon de [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]\n"\
"# sauvé dans le fichier fileNameInput\n"\
"sampleParam = ot.Sample.ImportFromCSVFile(fileNameInput)\n"\
"Nsample = sampleParam.getSize()\n"\
"\n"\
"\n"\
"#  Si des calculs ont déjà été faits, on les importe:\n"\
"try:\n"\
"    print('[ECLMProbabilities] Try to import previous results from {}'.format(fileNameRes))\n"\
"    allResults = ot.Sample.ImportFromCSVFile(fileNameRes)\n"\
"    print('import ok')\n"\
"except:\n"\
"    # la dimension du sample est 4*(n+1)\n"\
"    dim = 4*(n+1)\n"\
"    allResults = ot.Sample(0, dim)\n"\
"\n"\
"\n"\
"# Description\n"\
"desc = ot.Description()\n"\
"desc =  ['PEG('+str(k) + '|' +str(n) +')' for k in range(n+1)]\n"\
"desc += ['PSG('+str(k) +')' for k in range(n+1)]\n"\
"desc += ['PES('+str(k) + '|' +str(n) +')' for k in range(n+1)]\n"\
"desc += ['PTS('+str(k) + '|' +str(n) +')' for k in range(n+1)]\n"\
"allResults.setDescription(desc) \n"\
"\n"\
"# On passe les Nskip points déjà calculés (pas de pb si Nskip=0)\n"\
"Nskip = allResults.getSize()\n"\
"remainder_sample = sampleParam.split(Nskip)\n"\
"N_remainder = remainder_sample.getSize()\n"\
"print('N_remainder = ', N_remainder)\n"\
"while Ndone < N_remainder:\n"\
"    block += 1\n"\
"    # Nombre de calculs qui restent à faire\n"\
"    size = min(blockSize, N_remainder - Ndone)\n"\
"    print('Generate bootstrap data, block=', block, 'size=', size, 'Ndone=', Ndone, 'over', N_remainder)\n"\
"    t0 = time()\n"\
"    allInputs = [remainder_sample[(block-1)*blockSize + i] for i in range(size)]\n"\
"    t1 = time()\n"\
"    print('t=%.3g' % (t1 - t0), 's')\n"\
"\n"\
"    t0 = time()\n"\
"    pool = Pool()\n"\
"    # Calcul parallèle: pas d'ordre, retourné dès que réalisé\n"\
"    allResults.add(list(tqdm.tqdm(pool.imap_unordered(job, allInputs, chunksize=1), total=len(allInputs))))\n"\
"    pool.close()\n"\
"    t1 = time()\n"\
"    print('t=%.3g' % (t1 - t0), 's', 't (start)=%.3g' %(t1 - t00), 's')\n"\
"    Ndone += size\n"\
"    # Sauvegarde apres chaque bloc\n"\
"    allResults.exportToCSVFile(fileNameRes)")

        command =  'python script_bootstrap_ECLMProbabilities.py {} {} {} {}'.format(self.n, blockSize, fileNameInput, fileNameRes)
        os.system(command)
        os.remove(fileName)
        os.remove("myECLM.xml")


    def analyseGraphsECLMParam(self, fileNameSample):
        r"""
        Produces graphs to analyse a sample of (Mankamo and general) parameters.

        Parameters
        ----------
        fileNameSample: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.

        Returns
        -------
        graphPairsMankamoParam : :class:`~openturns.Graph`
            The Pairs graph of the Mankamo parameter  :eq:`MankamoParam`.
        graphPairsGeneralParam : :class:`~openturns.Graph`
            The Pairs graph of the general parameter :eq:`generalParam`.
        graphMarg_list : list of :class:`~openturns.Graph`
            The list of the marginal pdf of the Mankamoand general parameters.
        descParam: :class:`~openturns.Description`
            Description of each paramater.

        Notes
        -----
        i        The  marginal distributions are first estimated for the Mankamo parameter :eq:`MankamoParam` then for the general parameter :eq:`generalParam`.

        Each distribution is approximated with a Histogram and a normal kernel smoothing.
        """

        sampleParamAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        sampleParamRed = sampleParamAll.getMarginal([0, 1, 2, 3])
        sampleParamInit = sampleParamAll.getMarginal([4, 5, 6, 7, 8])
        descParam = sampleParamAll.getDescription()

        # Graphe Pairs sur le paramétrage  [Pt, Px_optim, Cco_optim, Cx_optim]
        graphPairsMankamoParam = ot.VisualTest.DrawPairs(sampleParamRed)

        # Graphe Pairs sur le paramétrage  [pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]
        graphPairsGeneralParam = ot.VisualTest.DrawPairs(sampleParamInit)

        graphMarg_list = list()

        # Graphe des marginales (Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim)
        for k in range(sampleParamAll.getDimension()):
            sample = sampleParamAll.getMarginal(k)
            Histo = ot.HistogramFactory().build(sample)
            KS = ot.KernelSmoothing()
            KS.setBoundaryCorrection(True)
            KS.setBoundingOption(ot.KernelSmoothing.BOTH)
            KS.setLowerBound(0.0)
            KS.setUpperBound(1.1)
            KS_dist = KS.build(sample)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descParam[k])
            graphMarg_list.append(graph)

        return graphPairsMankamoParam, graphPairsGeneralParam, graphMarg_list, descParam


    def analyseGraphsECLMProbabilities(self, fileNameSample, kMax):
        r"""
        Produces graphs to analyse a sample of all the ECLM probabilities.

        Parameters
        ----------
        fileNameSample: string
            The csv file that stores the ECLM probabilities.

        kMax : int, :math:`0 \leq k_{max} \leq n`
            The maximal multiplicity of the common cause failure.

        Returns
        -------
        graphPairs_list : list of :class:`~openturns.Graph`
            The Pairs graph of the ECLM probabilities.

        graphPEG_PES_PTS_list : list of :class:`~openturns.Graph`
            The Pairs graph of the probabilities :math:`(\mathrm{PEG}(k|n),\mathrm{PES}(k|n), \mathrm{PTS}(k|n))` for :math:`0 \leq k \leq k_{max}`.

        graphMargPEG_list : list of :class:`~openturns.Graph`
            The list of the marginal pdf of the  :math:`\mathrm{PEG}(k|n)` probabilities  for :math:`0 \leq k \leq k_{max}`.

        graphMargPSG_list : list of :class:`~openturns.Graph`
            The list of the marginal pdf of the  :math:`\mathrm{PSG}(k|n)` probabilities  for :math:`0 \leq k \leq k_{max}`.

        graphMargPES_list : list of :class:`~openturns.Graph`
            The list of the marginal pdf of the  :math:`\mathrm{PES}(k|n)` probabilities  for :math:`0 \leq k \leq k_{max}`.

        graphMargPTS_list : list of :class:`~openturns.Graph`
            The list of the marginal pdf of the  :math:`\mathrm{PTS}(k|n)` probabilities  for :math:`0 \leq k \leq k_{max}`.

        desc_list: :class:`~openturns.Description`
            Description of each graph.

        Notes
        -----
        Each distribution is approximated with a Histogram and a normal kernel smoothing.
        """

        sampleProbaAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        desc = sampleProbaAll.getDescription()
        dim = sampleProbaAll.getDimension()
        # dim = 4(n+1)
        n = dim // 4 - 1
        samplePEG = sampleProbaAll.getMarginal([k for k in range(n + 1)])
        descPEG = samplePEG.getDescription()
        samplePSG = sampleProbaAll.getMarginal([k for k in range(n + 1, 2 * n + 2)])
        descPSG = samplePSG.getDescription()
        samplePES = sampleProbaAll.getMarginal([k for k in range(2 * n + 2, 3 * n + 3)])
        descPES = samplePES.getDescription()
        samplePTS = sampleProbaAll.getMarginal([k for k in range(3 * n + 3, 4 * n + 4)])
        descPTS = samplePTS.getDescription()

        descPairs = ot.Description()

        # Graphe Pairs sur les PEG(k|n)
        graphPairsPEG = ot.VisualTest.DrawPairs(samplePEG.getMarginal([k for k in range(kMax + 1)]))
        descPairs.add('Pairs_PEG')
        # Graphe Pairs sur les PSG(k|n)
        graphPairsPSG = ot.VisualTest.DrawPairs(samplePSG.getMarginal([k for k in range(kMax + 1)]))
        descPairs.add('Pairs_PSG')
        # Graphe Pairs sur les PES(k|n)
        graphPairsPES = ot.VisualTest.DrawPairs(samplePES.getMarginal([k for k in range(kMax + 1)]))
        descPairs.add('Pairs_PES')
        # Graphe Pairs sur les PTS(k|n)
        graphPairsPTS = ot.VisualTest.DrawPairs(samplePTS.getMarginal([k for k in range(kMax + 1)]))
        descPairs.add('Pairs_PTS')

        # Comparaison  PEG(k|n) <= PES(k|n) <= PTS(k|n)
        # Graphe des lois marginales PEG(k|n)
        # Graphe des lois marginales PES(k|n)
        # Graphe des lois marginales PTS(k)

        graphPEG_PES_PTS_list = list()
        graphMargPEG_list = list()
        graphMargPSG_list = list()
        graphMargPES_list = list()
        graphMargPTS_list = list()

        descPEG_PES_PTS = ot.Description()
        descMargPEG = ot.Description()
        descMargPSG = ot.Description()
        descMargPES = ot.Description()
        descMargPTS = ot.Description()


        for k in range(kMax+1):
            samplePEG_k = samplePEG.getMarginal(k)
            samplePSG_k = samplePSG.getMarginal(k)
            samplePES_k = samplePES.getMarginal(k)
            samplePTS_k = samplePTS.getMarginal(k)

            # Comparaison  PEG(k|n) <= PES(k|n) <= PTS(k|n)
            samplePEG_PES_PTS_k = ot.Sample(0, 1)
            samplePEG_PES_PTS_k.add(samplePEG_k)
            samplePEG_PES_PTS_k.setDescription(samplePEG_k.getDescription())
            samplePEG_PES_PTS_k.stack(samplePES_k)
            samplePEG_PES_PTS_k.stack(samplePTS_k)
            graphPairs_k = ot.VisualTest.DrawPairs(samplePEG_PES_PTS_k)
            graphPEG_PES_PTS_list.append(graphPairs_k)
            descPEG_PES_PTS.add('PEG_PES_PTS_'+str(k))

            # Graphe des lois marginales PEG(k|n)
            Histo = ot.HistogramFactory().build(samplePEG_k)
            KS = ot.KernelSmoothing()
            KS.setBoundaryCorrection(True)
            KS.setBoundingOption(ot.KernelSmoothing.BOTH)
            KS.setLowerBound(0.0)
            KS.setUpperBound(1.1)
            KS_dist = KS.build(samplePEG_k)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descPEG[k])
            graphMargPEG_list.append(graph)
            descMargPEG.add('PEG_'+str(k))

            # Graphe des probabilités PSG(k)
            Histo = ot.HistogramFactory().build(samplePSG_k)
            KS = ot.KernelSmoothing()
            KS.setBoundaryCorrection(True)
            KS.setBoundingOption(ot.KernelSmoothing.BOTH)
            KS.setLowerBound(0.0)
            KS.setUpperBound(1.1)
            KS_dist = KS.build(samplePSG_k)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descPSG[k])
            graphMargPSG_list.append(graph)
            descMargPSG.add('PSG_'+str(k))

            # Graphe des lois marginales PES(k|n)
            Histo = ot.HistogramFactory().build(samplePES_k)
            KS = ot.KernelSmoothing()
            KS.setBoundaryCorrection(True)
            KS.setBoundingOption(ot.KernelSmoothing.BOTH)
            KS.setLowerBound(0.0)
            KS.setUpperBound(1.1)
            KS_dist = KS.build(samplePES_k)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descPES[k])
            graphMargPES_list.append(graph)
            descMargPES.add('PES_'+str(k))

            # Graphe des probabilités PTS(k|n)
            Histo = ot.HistogramFactory().build(samplePTS_k)
            KS = ot.KernelSmoothing()
            KS.setBoundaryCorrection(True)
            KS.setBoundingOption(ot.KernelSmoothing.BOTH)
            KS.setLowerBound(0.0)
            KS.setUpperBound(1.1)
            KS_dist = KS.build(samplePTS_k)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descPTS[k])
            graphMargPTS_list.append(graph)
            descMargPTS.add('PTS_'+str(k))

        return [graphPairsPEG, graphPairsPSG, graphPairsPES, graphPairsPTS], graphPEG_PES_PTS_list, graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list, [descPairs, descPEG_PES_PTS, descMargPEG, descMargPSG, descMargPES, descMargPTS]


    def analyseDistECLMProbabilities(self, fileNameSample, kMax, confidenceLevel, factoryColl):
        r"""
        Fits  distribution on ECL probabilities sample.

        Parameters
        ----------
        fileNameSample: string
            The csv file that stores the sample of all the ECLM probabilities.
        kMax : int, :math:`0 \leq k_{max} \leq 1`
            The maximal multipicity of the common cause failure.
        confidenceLevel : float, :math:`0 \leq confidenceLevel \leq 1`
            The confidence level of each interval.
        factoryCollection : list of :class:`~openturns.DistributionFactory`
            List of factories that will be used to fit a distribution to the sample.
        desc_list: :class:`~openturns.Description`
            Description of each graph.

        Returns
        -------
        confidenceInterval_list : list of :class:`~openturns.Interval`
            The confidence intervals of all the  ECLM probability.
        graph_marg_list : list of  :class:`~openturns.Graph`
            The fitting graphs of all the ECLM probabilities.

        Notes
        -----
        The confidence intervals and the graphs illustrating the fitting are given according to the following order:  :math:`(\mathrm{PEG}(0|n), \dots, \mathrm{PEG}(n|n), \mathrm{PSG}(0|n), \dots, \mathrm{PSG}(n|n), \mathrm{PES}(0|n), \dots, \mathrm{PES}(n|n), \mathrm{PTS}(0|n), \dots, \mathrm{PTS}(n|n))`.
        Each fitting is tested using the Lilliefors test. The result is printed and the best model among the list of factories is given. Care: it is not guaranted that the best model be accepted by the Lilliefors test.
        """

        sampleProbaAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        desc = sampleProbaAll.getDescription()
        dim = sampleProbaAll.getDimension()
        # dim = 4(n+1)
        n = int(dim/4)-1
        samplePEG = sampleProbaAll.getMarginal([k for k in range(n + 1)])
        descPEG = samplePEG.getDescription()
        samplePSG = sampleProbaAll.getMarginal([k for k in range(n + 1, 2 * n + 2)])
        descPSG = samplePSG.getDescription()
        samplePES = sampleProbaAll.getMarginal([k for k in range(2 * n + 2, 3 * n + 3)])
        descPES = samplePES.getDescription()
        samplePTS = sampleProbaAll.getMarginal([k for k in range(3 * n + 3, 4 * n + 4)])
        descPTS = samplePTS.getDescription()

        KS = ot.KernelSmoothing()
        KS.setBoundaryCorrection(True)
        KS.setBoundingOption(ot.KernelSmoothing.BOTH)
        KS.setLowerBound(0.0)
        KS.setUpperBound(1.1)

        IC_PEG_list = list()
        IC_PSG_list = list()
        IC_PES_list = list()
        IC_PTS_list = list()

        quantSup = 0.5 + confidenceLevel / 2
        quantInf = 0.5 - confidenceLevel / 2

        graphMargPEG_list = list()
        graphMargPSG_list = list()
        graphMargPES_list = list()
        graphMargPTS_list = list()

        descMargPEG = ot.Description()
        descMargPSG = ot.Description()
        descMargPES = ot.Description()
        descMargPTS = ot.Description()

        colors = ['blue', 'red', 'black', 'green', 'violet', 'pink']

        for k in range(kMax+1):
            samplePEG_k = samplePEG.getMarginal(k)
            samplePSG_k = samplePSG.getMarginal(k)
            samplePES_k = samplePES.getMarginal(k)
            samplePTS_k = samplePTS.getMarginal(k)

            ##################################
            # Intervalles de confiance centrés
            KS_dist_PEG_k = KS.build(samplePEG_k)
            KS_dist_PSG_k = KS.build(samplePSG_k)
            KS_dist_PES_k = KS.build(samplePES_k)
            KS_dist_PTS_k = KS.build(samplePTS_k)

            IC_PEG_k = ot.Interval(KS_dist_PEG_k.computeQuantile(quantInf)[0], KS_dist_PEG_k.computeQuantile(quantSup)[0])
            IC_PSG_k = ot.Interval(KS_dist_PSG_k.computeQuantile(quantInf)[0], KS_dist_PSG_k.computeQuantile(quantSup)[0])
            IC_PES_k = ot.Interval(KS_dist_PES_k.computeQuantile(quantInf)[0], KS_dist_PES_k.computeQuantile(quantSup)[0])
            IC_PTS_k = ot.Interval(KS_dist_PTS_k.computeQuantile(quantInf)[0], KS_dist_PTS_k.computeQuantile(quantSup)[0])

            IC_PEG_list.append(IC_PEG_k)
            IC_PSG_list.append(IC_PSG_k)
            IC_PES_list.append(IC_PES_k)
            IC_PTS_list.append(IC_PTS_k)

            ##################################
            # Adéquation à un famille de lois
            # test de Lilliefors
            # graphe pdf: histo + KS + lois proposées
            print('Test de Lilliefors')
            print('==================')
            print('')
            
            best_model_PEG_k = 'none'
            best_model_PSG_k = 'none'
            best_model_PES_k = 'none'
            best_model_PTS_k = 'none'
            
            print('Ordre k=', k)
            try:
                print('PEG...')
                state = ot.RandomGenerator.GetState()
                ot.RandomGenerator.SetSeed(0)
                best_model_PEG_k, best_result_PEG_k = ot.FittingTest.BestModelLilliefors(samplePEG_k, factoryColl)
                ot.RandomGenerator.SetState(state)

                print('Best model PEG(', k, '|n) : ', best_model_PEG_k, 'p-value = ', best_result_PEG_k.getPValue())
                best_model_PEG_k = best_model_PEG_k.getName()
            except:
                pass
            try:
                print('PSG...')
                best_model_PSG_k, best_result_PSG_k = ot.FittingTest.BestModelLilliefors(samplePSG_k, factoryColl)
                print('Best model PSG(', k, '|n) : ', best_model_PSG_k, 'p-value = ', best_result_PSG_k.getPValue())
                best_model_PSG_k = best_model_PSG_k.getName()
            except:
                pass
            try:
                print('PES...')
                best_model_PES_k, best_result_PES_k = ot.FittingTest.BestModelLilliefors(samplePES_k, factoryColl)
                print('Best model PES(', k, '|n) : ', best_model_PES_k, 'p-value = ', best_result_PES_k.getPValue())
                best_model_PES_k = best_model_PES_k.getName()
            except:
                pass
            try:
                print('PTS...')
                best_model_PTS_k, best_result_PTS_k = ot.FittingTest.BestModelLilliefors(samplePTS_k, factoryColl)
                print('Best model PTS(', k, '|n) : ', best_model_PTS_k, 'p-value = ', best_result_PTS_k.getPValue())
                best_model_PTS_k = best_model_PTS_k.getName()
            except:
                pass
            print('')

            ##############################
            # Graphe des ajustements
            # PEG
            Histo = ot.HistogramFactory().build(samplePEG_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PEG_k.drawPDF())
            leg.add('KS')
            graph.setColors(colors[0:2])
            graph.setLegends(leg)
            nbFact = len(factoryColl)
            for i in range(nbFact):
                try:
                    dist = factoryColl[i].build(samplePEG_k)
                    draw = dist.drawPDF().getDrawable(0)
                    draw.setColor(colors[i + 2])
                    draw.setLegend(dist.getName())
                    graph.add(draw)
                except:
                    pass

            graph.setLegendPosition('topright')
            graph.setXTitle(descPEG[k])
            graph.setTitle('PEG('+str(k) + '|' + str(n) + ') - best model : ' +  best_model_PEG_k)
            graphMargPEG_list.append(graph)
            descMargPEG.add('PEG_'+str(k))

            # PSG
            Histo = ot.HistogramFactory().build(samplePSG_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PSG_k.drawPDF())
            leg.add('KS')
            graph.setColors(colors[0:2])
            graph.setLegends(leg)
            nbFact = len(factoryColl)
            for i in range(nbFact):
                try:
                    dist = factoryColl[i].build(samplePSG_k)
                    draw = dist.drawPDF().getDrawable(0)
                    draw.setColor(colors[i+2])
                    draw.setLegend(dist.getName())
                    graph.add(draw)
                except:
                    pass
                    
            graph.setLegendPosition('topright')
            graph.setXTitle(descPSG[k])
            graph.setTitle('PSG('+str(k) + '|' + str(n) + ') - best model : ' +   best_model_PSG_k)
            graphMargPSG_list.append(graph)
            descMargPSG.add('PSG_'+str(k))

            # PES
            Histo = ot.HistogramFactory().build(samplePES_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PES_k.drawPDF())
            leg.add('KS')
            graph.setColors(colors[0:2])
            graph.setLegends(leg)
            nbFact = len(factoryColl)
            for i in range(nbFact):
                try:
                    dist = factoryColl[i].build(samplePES_k)
                    draw = dist.drawPDF().getDrawable(0)
                    draw.setColor(colors[i + 2])
                    draw.setLegend(dist.getName())
                    graph.add(draw)
                except:
                    pass

            graph.setLegendPosition('topright')
            graph.setXTitle(descPES[k])
            graph.setTitle('PES('+str(k) + '|' + str(n) + ') - best model : ' +   best_model_PES_k)
            graphMargPES_list.append(graph)
            descMargPES.add('PES_'+str(k))

            # PTS
            Histo = ot.HistogramFactory().build(samplePTS_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PTS_k.drawPDF())
            leg.add('KS')
            graph.setColors(colors[0:2])
            graph.setLegends(leg)
            nbFact = len(factoryColl)
            for i in range(nbFact):
                try:
                    dist = factoryColl[i].build(samplePTS_k)
                    draw = dist.drawPDF().getDrawable(0)
                    draw.setColor(colors[i+2])
                    draw.setLegend(dist.getName())
                    graph.add(draw)
                except:
                    pass

            graph.setLegendPosition('topright')
            graph.setXTitle(descPTS[k])
            graph.setTitle('PTS('+str(k) + '|' + str(n) + ') - best model : ' +   best_model_PTS_k)
            graphMargPTS_list.append(graph)
            descMargPTS.add('PTS_'+str(k))

        return [IC_PEG_list, IC_PSG_list, IC_PES_list, IC_PTS_list], [graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list] , [descMargPEG, descMargPSG, descMargPES, descMargPTS]


    def computeKMaxPTS(self, p):
        r"""
        Computes the minimal  multipicity of the common cause failure with a probability greater than a given threshold.

        Parameters
        ----------
        p : float, :math:` 0 \leq p \leq 1`
            The probability threshold.

        Returns
        -------
        kMax : int
            The minimal  multipicity of the common cause failure with a probability greater than :math:`p`.

        Notes
        -----
        The :math:`k_{max}` multiplicity is the minimal  multipicity of the common cause failure such that the probability that at least :math:`k_{max}` failures occur is greater than :math:`p`. Then :math:`k_{max}` is defined by:

        .. math::
            :label: kMaxDef

            k_{max}(p) = \max \{ k |  \mathrm{PTS}(k|n) > p \}

        The probability :math:`\mathrm{PTS}(k|n)` is computed using :eq:`PTS_red`.
        """

        k = 0
        while self.computePTS(k) > p:
            k += 1
        return k - 1


    def computeAnalyseKMaxSample(self, p, fileNameInput, fileNameRes, blockSize = 256):
        r"""
        Generates a :math:`k_{max}` sample and produces graphs to analyse it.

        Parameters
        ----------
        p : float, :math:` 0 \leq p \leq 1`
            The probability threshold.

        fileNameInput: string
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.

        fileNameRes: string
            The csv file that stores the sample of  :math:`k_{max}` defined by :eq:`kMaxDef`.

        blockSize : int,
            The block size after which the sample is saved. Default value is 256.

        Returns
        -------
        kmax_graph :  :class:`~openturns.Graph`
            The empirical distribution of :math:`K_{max}`.

        Notes
        -----
        The function generates the script *script_bootstrap_KMax.py* that uses the parallelisation of the pool object of the multiprocessing module.  It also creates a file *myECLM.xml* that stores the total impact vector to be read by the script. Both files are removed at the end of the execution of the method.

        The computation is saved in the csv file named *fileNameRes* every blockSize calculus. The computation can be interrupted: it will be restarted from the last *filenameRes* saved.

        The empirical distribution is fitted on the sample. The $90\%$ confidence interval is given, computed from the empirical distribution.
        """

        myStudy = ot.Study('myECLM.xml')
        myStudy.add('integrationAlgo', self.integrationAlgo)
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.add('nIntervalsAsIndices', ot.Indices(1, self.nIntervals))
        myStudy.save()

        import os
        directory_path = os.getcwd()
        fileName = directory_path + "/script_bootstrap_KMax.py"
        if os.path.exists(fileName):
            os.remove(fileName)
        with open(fileName, "w") as f:
            f.write("#############################\n"\
"# Ce script :\n"\
"#    - récupère un échantillon de (Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim] dans le fichier fileNameInput (de type adresse/fichier.csv)\n"\
"#     - calcule toutes les grandeurs probabilistes de l'ECLM\n"\
"#     - sauve le sample des grandeurs  probabilistes dans  le fichier fileNameRes (de type adresse/fichier.csv)\n"\
"\n"\
"\n"\
"import openturns as ot\n"\
"from oteclm import ECLM\n"\
"\n"\
"from signal import signal, SIGPIPE, SIG_DFL\n"\
"signal(SIGPIPE, SIG_DFL)\n"\
"\n"\
"from time import time\n"\
"import sys\n"\
"\n"\
"from multiprocessing import Pool\n"\
"# barre de progression\n"\
"import tqdm\n"\
"\n"\
"# Ot Parallelisme desactivated\n"\
"ot.TBB.Disable()\n"\
"\n"\
"# seuil de PTS\n"\
"p = float(sys.argv[1])\n"\
"# taille des blocs\n"\
"blockSize = int(sys.argv[2])\n"\
"# Nom du fichier de contenant le sample des paramètres de Mankamo: de type adresse/fichier.csv\n"\
"fileNameInput = str(sys.argv[3])\n"\
"# Nom du fichier de sauvegarde: de type adresse/fichier.csv\n"\
"fileNameRes = str(sys.argv[4])\n"\
"\n"\
"print('Calcul des réalisations de KMax')\n"\
"print('p, fileNameInput, fileNameRes = ', p, fileNameInput, fileNameRes)\n"\
"\n"\
"# Import de  vectImpactTotal\n"\
"myStudy = ot.Study('myECLM.xml')\n"\
"myStudy.load()\n"\
"integrationAlgo = ot.IntegrationAlgorithm()\n"\
"myStudy.fillObject('integrationAlgo', integrationAlgo)\n"\
"totalImpactVector = ot.Indices()\n"\
"myStudy.fillObject('totalImpactVector', totalImpactVector)\n"\
"nIntervalsAsIndices = ot.Indices()\n"\
"myStudy.fillObject('nIntervalsAsIndices', nIntervalsAsIndices)\n"\
"\n"\
"myECLM = ECLM(totalImpactVector, integrationAlgo, nIntervalsAsIndices)\n"\
"\n"\
"def job(inP):\n"\
"    # pointParam_Gen = [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]\n"\
"    generalParameter = inP[4:9]\n"\
"    myECLM.setGeneralParameter(generalParameter)\n"\
"    kMax = myECLM.computeKMaxPTS(p)\n"\
"    return [kMax]\n"\
"\n"\
"\n"\
"Ndone = 0\n"\
"block = 0\n"\
"t00 = time()\n"\
"\n"\
"# Import de l'échantillon de [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]\n"\
"# sauvé dans le fichier fileNameInput\n"\
"sampleParam = ot.Sample.ImportFromCSVFile(fileNameInput)\n"\
"Nsample = sampleParam.getSize()\n"\
"\n"\
"\n"\
"#  Si des calculs ont déjà été faits, on les importe:\n"\
"try:\n"\
"    print('[Kmax values] Try to import previous results from {}'.format(fileNameRes))\n"\
"    allResults = ot.Sample.ImportFromCSVFile(fileNameRes)\n"\
"    print('import ok')\n"\
"except:\n"\
"    # la dimension du sample est 1\n"\
"    dim = 1\n"\
"    allResults = ot.Sample(0, dim)\n"\
"\n"\
"\n"\
"# Description\n"\
"desc = ot.Description(['kMax'])\n"\
"allResults.setDescription(desc)\n"\
"\n"\
"# On passe les Nskip points déjà calculés (pas de pb si Nskip=0)\n"\
"Nskip = allResults.getSize()\n"\
"remainder_sample = sampleParam.split(Nskip)\n"\
"N_remainder = remainder_sample.getSize()\n"\
"print('N_remainder = ', N_remainder)\n"\
"while Ndone < N_remainder:\n"\
"    block += 1\n"\
"    # Nombre de calculs qui restent à faire\n"\
"    size = min(blockSize, N_remainder - Ndone)\n"\
"    print('Generate bootstrap data, block=', block, 'size=', size, 'Ndone=', Ndone, 'over', N_remainder)\n"\
"    t0 = time()\n"\
"    allInputs = [remainder_sample[(block-1)*blockSize + i] for i in range(size)]\n"\
"    t1 = time()\n"\
"    print('t=%.3g' % (t1 - t0), 's')\n"\
"\n"\
"    t0 = time()\n"\
"    with Pool() as pool:\n"\
"        # Calcul parallèle: pas d'ordre, retourné dès que réalisé\n"\
"        allResults.add(list(tqdm.tqdm(pool.imap_unordered(job, allInputs, chunksize=1), total=len(allInputs))))\n"\
"    t1 = time()\n"\
"    print('t=%.3g' % (t1 - t0), 's', 't (start)=%.3g' %(t1 - t00), 's')\n"\
"    Ndone += size\n"\
"    # Sauvegarde apres chaque bloc\n"\
"    allResults.exportToCSVFile(fileNameRes)")

        command =  'python script_bootstrap_KMax.py {} {} {} {}'.format(p, blockSize, fileNameInput, fileNameRes)
        os.system(command)
        os.remove(fileName)
        os.remove("myECLM.xml")

        # Loi KS
        sampleKmax = ot.Sample.ImportFromCSVFile(fileNameRes)
        UD_dist_KMax = ot.UserDefinedFactory().build(sampleKmax)

        # Graphe: UD
        graph = UD_dist_KMax.drawPDF()
        leg = ot.Description(1,'Empirical')
        graph.setColors(['blue'])
        graph.setLegends(leg)
        graph.setLegendPosition('topright')
        graph.setXTitle(r'$k_{max}$')
        graph.setTitle(r'Loi de $K_{max} = \arg \max \{k | PTS(k|$'+ str(self.n) + r'$) \geq $'+ format(p,'.1e') + r'$\}$')

        # IC à 90%
        print('Intervalle de confiance de niveau 90%: [',  UD_dist_KMax.computeQuantile(0.05)[0], ', ', UD_dist_KMax.computeQuantile(0.95)[0], ']')

        return graph


    def setTotalImpactVector(self, totalImpactVector):
        r"""
        Accessor to the total impact vector.

        Parameters
        ----------
        totalImpactVector : :class:`~openturns.Indices`
            The total impact vector of the common cause failure (CCF) group.
        """

        self.totalImpactVector = totalImpactVector

        # Remise à jour du n
        self.n = totalImpactVector.getSize() - 1

        #Remise à jour Pt
        N = sum(totalImpactVector)
        Pt = 0.0
        for i in range(1,self.n + 1):
            Pt += i * totalImpactVector[i]
        Pt /= self.n * N
        self.Pt = Pt
        self.PEGAll = ot.Point(self.n + 1, -1.0)

    def setIntegrationAlgo(self, integrationAlgo):
        r"""
        Accessor to the integration algorithm.

        Parameters
        ----------
        integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
            The integration algorithm used to compute the integrals.
        """

        self.integrationAlgo = integrationAlgo
        self.PEGAll = ot.Point(self.n + 1, -1.0)
        self.PSGAll = ot.Point(self.n + 1, -1.0)


    def setMankamoParameter(self, mankamoParameter):
        r"""
        Accessor to the Mankamo parameter.

        Parameters
        ----------
        mankamoParameter : list of float
            The Mankamo parameter :eq:`MankamoParam`.

        Notes
        -----
        It automaticcally updates the general parameter.
        """

        self.MankamoParameter = mankamoParameter
        # General parameter = (pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim)
        generalParam = self.computeGeneralParamFromMankamo(mankamoParameter)
        self.setGeneralParameter(generalParam)


    def setGeneralParameter(self, generalParameter):
        r"""
        Accessor to the general parameter.

        Parameters
        ----------
        generalParameter : list of float
            The general parameter  :eq:`generalParam`.
        """

        self.generalParameter = generalParameter
        self.PEGAll = ot.Point(self.n + 1, -1.0)
        self.PSGAll = ot.Point(self.n + 1, -1.0)


    def getTotalImpactVector(self):
        r"""
        Accessor to the total impact vector.

        Returns
        -------
        totalImpactVector : :class:`~openturns.Indices`
            The total impact vector of the CCF group.
        """

        return self.totalImpactVector


    def getIntegrationAlgorithm(self):
        r"""
        Accessor to the integration algorithm.

        Returns
        -------
        integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
            The integration algorithm used to compute the integrals.
        """

        return self.integrationAlgo


    def getMankamoParameter(self):
        r"""
        Accessor to the Mankamo Parameter.

        Returns
        -------
        mankamoParameter : :class:`~openturns.Point`
            The Mankamo parameter :eq:`MankamoParam`.
        """

        return self.MankamoParameter


    def getGeneralParameter(self):
        r"""
        Accessor to the general parameter.

        Returns
        -------
        generalParameter : list of foat
            The general parameter defined in :eq:`generalParam`.
        """

        return self.generalParameter


    def getN(self):
        r"""
        Accessor to the CCF group size :math:`n`.

        Returns
        -------
        n : int
            The CCF group size :math:`n`.
        """

        return self.n


    def getPt(self):
        r"""
        Accessor to the the probability :math:`P_t`.

        Returns
        -------
        Pt : float, :math:`0 < P_t < 1`
            The estimator of PT(:math:`|n`).
        """

        return self.Pt
