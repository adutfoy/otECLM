import numpy as np
import openturns as ot
import math as math


class ECLM(object):
    """
    Treatment of failure probabilities and dependencies in highly redundant systems according to the Extended Common Load method (ECLM).
    """ 
    
    def __init__(self, totalImpactVector, integrationAlgo = ot.GaussLegendre([50])):
        """
        Creates a new ECLM class.
        
        Parameters
        ----------
        totalImpactVector : :class:`~openturns.Indices`
            The total impact vector of the common cause failure (CCF) group.
        integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
            The integration algorithm used to compute the integrals.

            By defaut, :class:`~openturns.GaussLegendre` parameterized by 50 points.
    
        Notes
        -----
        We consider a common cause failure (CCF) group of :math:`n` components supposed to be independent and identical. 

        We denote by :math:`S` the load and :math:`R_i` the resistance of the component :math:`i` of the CCF group. We assume that :math:`(R_1, \dots, R_n)` are independant and identically distributed according to :math:`R`.
 
        We assume that the load :math:`S` is modelled as a mixture of two normal distributions:

            - the base part  with mean and variance :math:`(\mu_b, \sigma_b^2)` 
            - the extreme load parts  with mean and variance :math:`(\mu_x, \sigma_x^2)` 
            - the respective weights :math:`(\pi, 1-\pi)`.

        Then the density of :math:`S` is written as:

        ..math::
        
            f_S(s) = \pi \varphi \left(\dfrac{s-\mu_b}{\sigma_b}\right) + (1-\pi) \varphi \left(\dfrac{s-\mu_x}{\sigma_x}\right)\quad \forall s \in \Rset

        We assume that the resistance :math:`R` is modelled as normal distribution with  mean and variance :math:`(\mu_R, \sigma_R^2)`. We denote by :math:`p_R` and :math:`F_R` its density and the cumulative density function.

        We define the ECLM probabilities, for :math:`0 \leq k \leq n`: 

            - :math:`\mbox{PSG}(k)`: probability that  specific set of :math:`k` components fail.
            - :math:`\mbox{PEG}(k|n)`: probability that a specific set of :math:`k` components fail in a CCF group of size :math:`n` while the other :math:`(n-k)` survive.
            - :math:`\mbox{PES}(k|n)`: probability that some set of :math:`k` components fail while the other :math:`(n-k)` survive.
            - :math:`\mbox{PTS}(k|n)`: probability that  at least some specific set of :math:`k` components fail in a CCF group of size :math:`n`.

        Then the  :math:` \mbox{PEG}(k|n)`  probabilities are defined as:

        ..math::
    
            \begin{array}{rcl}
              \mbox{PEG}(k|n) & = & \Prob{S>R_1, \dots, S>R_k, S<R_{k+1}, \dots, S<R_n} \\
                              & = &  \int_{s\in  \Rset} f_S(s) \left[F_R(s)\right]^k \left[1-F_R(s)\right]^{n-k} \, ds
            \end{array}
        
        and the  :math:` \mbox{PSG}(k|n)` probabilities are defined as:
        
        ..math::
    
            \begin{array}{rcl}
              \mbox{PSG}(k) & = & \Prob{S>R_1, \dots, S>R_k}\\
                            & = &  \int_{s\in  \Rset} f_S(s) \left[F_R(s)\right]^k\, ds
            \end{array}
    
        We get the :math:`\mbox{PES}(k|n)` probabilities and  :math:`\mbox{PTS}(k|n)` with the relations: 

        ..math::
    
            \begin{array}{rcl}
                \mbox{PES}(k|n) & = & C_n^k \, \mbox{PEG}(k|n)\\
                \mbox{PTS}(k|n) & = & \sum_{i=k}^n  \mbox{PES}(i|n)
            \end{array}

        We use the following set of parameters called General Parameter: :math:`(\pi, d_b, d_x, d_R, y_{xm})` defined by:

        ..math::
            :label:`generalParam`

            \begin{array}{rcl}
             d_{b} & = & \dfrac{\sigma_b}{\mu_R-\mu_b}\\
             d_{x} & = & \dfrac{\sigma_x}{\mu_R-\mu_b}\\
             d_{R} & = & \dfrac{\sigma_R}{\mu_R-\mu_b}\\
             y_{xm} & = & \dfrac{\mu_x-\mu_b}{\mu_R-\mu_b}
            \end{array}

        Then, the  :math:` \mbox{PEG}(k|n)` and :math:` \mbox{PSG}(k|n)` probabilities are written as:

        ..math::
            :label:`PEG_red`

            \begin{array}{rcl}
                 \mbox{PEG}(k|n) & = &   \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right] \left[\Phi\left(\dfrac{y-1}{d_R}\right)\right]^k \left[1-\Phi\left(\dfrac{y-1}{d_R}\right)\right]^{n-k} \, dy \\
                 \mbox{PSG}(k) & = &   \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right] \left[\Phi\left(\dfrac{y-1}{d_R}\right)\right]^k  \, dy
            \end{array}

         The computation of the :math:`PEG(k|n)`  and :math:` \mbox{PSG}(k|n)` probabilities is done with a quadrature method. By default, we use the Gauss-Legendre quadrature method parameterized by 50 points.

        The data is the  impact vector :math:`V_t^{n,N}` of the CCF group the component :math:`k for :math:{0 \leq k \leq n` is the number of failure events of multiplicity :math:`k` in the CCF group. In addition, :math:`N` is the number of tests and demands on the whole group. Then we have :math:`N = \sum_{k=0}^n V_t^{n,N}[k]`.
        """
        
        # set attribute
        self.totalImpactVector = totalImpactVector
        self.integrationAlgo = integrationAlgo
        self.n = self.totalImpactVector.getSize()-1
        # Mankamo Param: (P_t, P_x, C_{co}, C_x)
        self.MankamoParameter = None
        # GeneralParam: (pi, d_b, d_x, d_R, y_{xm})
        self.generalParameter = None

    def setTotalImpactVector(self, totalImpactVector):
        """
        Accessor to the total impact vector.

        Parameters
        ----------
        totalImpactVector : :class:`~openturns.Indices`
            The total impact vector of the common cause failure (CCF) group.
        """

        self.totalImpactVector = totalImpactVector

        
    def setIntegrationAlgo(self, integrationAlgo):
        """
        Accessor to the integration algorithm.

        Parameters
        ----------
        integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
            The intergration algorithm used to compute the integrals.
        """

        self.integrationAlgo = integrationAlgo

        
    def setMankamoParameter(self, mankamoParameter):
        """
        Accessor to the Mankamo Parameter :math:`(P_t, P_x, C_{co}, C_x)`.

        Parameters
        ----------
        mankamoParameter : list of float
            The Mankamo parameter  :math:`(P_t, P_x, C_{co}, C_x)`.
        """

        self.MankamoParameter = mankamoParameter

        
    def setGeneralParameter(self, generalParameter):
        """
        Accessor to the general Parameter :math:`(pi, d_b, d_x, d_R, y_{xm})`.

        Parameters
        ----------
        generalParameter : list of float
            The general parameter  :math:`(pi, d_b, d_x, d_R, y_{xm})`
        """

        self.generalParameter = generalParameter
                

    def setN(self, n):
        """
        Accessor to the size of the CCF group :math:`n`.

        Parameters
        ----------
        n : int
            The size of the CF group.
        """

        self.n = n

        
    def getTotalImpactVector(self):
        """
        Accessor to the total impact vector.

        Returns
        -------
        totalImpactVector : :class:`~openturns.Indices`
            The total impact vector of the common cause failure (CCF) group.
        """

        return self.totalImpactVector

        
    def getIntegrationAlgorithm(self):
        """
        Accessor to the integration algorithm.

        Returns
        -------
        integrationAlgo : :class:`~openturns.IntegrationAlgorithm`
            The intergration algorithm used to compute the integrals.
        """

        return self.integrationAlgo

        
    def getMankamoParameter(self):
        """
        Accessor to the Mankamo Parameter :math:`(P_t, P_x, C_{co}, C_x)`.

        Returns
        -------
        mankamoParameter : :class:`~openturns.Point`
            The Mankamo parameter.
        """

        return self.MankamoParameter

        
    def getGeneralParameter(self):
        """
        Accessor to the General Parameter :math:`(pi, d_b, d_x, d_R, y_{xm})`.

        Returns
        -------
        generalParameter : list of foat
            The General parameter defined in :eq:`generalParam`.
        """

        return self.generalParameter

    def getN(self):
        """
        Accessor to the size of the CCF group :math:`n`.

        Returns
        -------
        n : int
            The size of the CF group.
        """

        return self.n
        
    def estimateMaxLikelihoodFromMankamo(self, startingPoint, visuLikelihood=False, verbose=False):
        """
        Estimates the maximum likelihood General and Mankamo parameters under the Mankamo assumption.

        Parameters
        ----------
        startingPoint : :class:`~openturns.Point`
            Start point :math:`(P_t, P_x, C_{co}, C_x)` for the optimization problem.
        visuLikelihood : Bool
            Produce the graph of the log-likelihood function at the optimal point.
            By default, False.
        verbose : Bool
            Verbose level of the algorithm.

            By default, False.
            

        Returns
        -------
        paramList : :class:`~openturns.Point`
            The optimal point :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})` where :math:`y_{xm} = 1-d_R`.
        finalLogLikValue : float
            The value of the log-likelihood function at the optimal point.
        graphList : list  of :class:`~openturns.Graph`
            The collection of g_fixedlogPxCco, g_fixedlogPxCx, g_fixedCcoCx, g_fixedCx, g_fixedCco, g_fixedlogPx of the log-likelihood function at the optimal point when one or two components are fixed.

        Notes
        -----
         Mankamo introduces a new set of parameters:  :math:`(P_t, P_x, C_{co}, C_x, y_{xm})` and defined from the general parameters :math:`(\pi, d_b, d_x, d_R, y_{xm})` as follows:

        ..math::
            :label: Param2

            \begin{array}{rcl}
                 P_t & = & \mbox{PSG}(1) =   \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy \\
                 P_x &  = & \int_{-\infty}^{+\infty} \left[  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy = (1-\pi) \left[1-\Phi\left(\dfrac{1-y_{xm}}{\sqrt{d_x^2+d_R^2}}\right)\right]\\
                 c_{co} & = & \dfrac{d_b^2}{d_b^2+d_R^2}\\
                 c_x & = & \dfrac{d_x^2}{d_x^2+d_R^2}
            \end{array}
        
        Mankamo assumes that:

        ..math::
            :label: mankamoHyp

            y_{xm} = 1-d_R

        
        This assumption means that :math:`\mu_R = \mu_x+\sigma_R`. Then equations (:eq:`Param2`) simplify and we get:

        ..math::
            :label: Parma2to1Mankamo

            \begin{array}{rcl}
                (1-\pi) & = & \dfrac{P_x}{\Phi\left(\textcolor{red}{-} \sqrt{1-c_{x}}\right)}\\
                d_b & = & \dfrac{\sqrt{c_{co}}}{\textcolor{red}{-}\Phi^{-1}\left(\dfrac{P_t-P_x}{\pi} \right)}\\
                d_R  & = & \dfrac{\sqrt{1-c_{co}}}{-\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)} \\
                d_x  & = & d_R \sqrt{\dfrac{c_{x}}{1-c_{x}}}
            \end{array}

        We call  :math:`(P_t, P_x, C_{co}, C_x)` the Mankamo parameter.

        The likelihood of the model is written as according to the total impact vector :math:`V_t^{n,N}` and the set of parameters :math:`(P_x, C_{co}, C_x)`

        ..math::

             \log \cL(\vect{\theta}|V_t^{n,N}) = \sum_{k=0}^n V_t^{n,N}[k] \log \mbox{PEG}(k|n)

        Then the optimal parameter maximises the log-likelihood of the model:        

        ..math::
            :label:`optimMankamo`  

             (P_x, C_{co}, C_x)_{optim}  = \argmax_{(P_x, C_{co}, C_x)} \log \cL((P_x, C_{co}, C_x)|V_t^{n,N})

        The optimization is done under the following constraints:       

        ..math::

            \begin{array}{l}
                 0 \leq P_x  \leq P_t \\
                 0 \leq c_{co} \leq 1 \\
                 0 \leq c_{x} \leq 1  \\
                 0 \leq 1-\pi = \dfrac{P_x}{\Phi\left(- \sqrt{1-c_{x}}\right)}\leq 1 \\
                 0 \leq d_b = \dfrac{\sqrt{c_{co}}}{-\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)} \\
                  0 \leq  d_R   = \dfrac{\sqrt{1-c_{co}}}{-\Phi^{-1}\left( \dfrac{P_t-P_x}{\pi} \right)} 
            \end{array}

        Assuming that $P_t \leq 0.2$, we can write the constraints as:

        ..math::

            \begin{array}{l}
                0< P_t \leq \dfrac{1}{2}\\
                0 < P_x  \leq \min\{P_t,  (P_t-\dfrac{1}{2} ) \left(
                1-\dfrac{1}{2\Phi\left(-\sqrt{1-c_{x}}\right)}\right)^{-1},  \Phi\left(- \sqrt{1-c_{x}}\right)  \} \\
                0 < c_{co} < 1 \\
                0 < c_{x} < 1  
            \end{array}         

        The parameter :math:`P_t` is directly estimated form the total impact vector: 

        ..math::

             \hat{P}_t = \sum_{i=1}^n\dfrac{iV_t^{n,N}[i]}{nN}

        """

        # Nombre de sollicitations
        N = sum(self.totalImpactVector)

        # estimateur de Pt
        Pt = 0.0
        for i in range(1,self.n+1):
            Pt += i*self.totalImpactVector[i]
        Pt /= self.n*N
        logPt = math.log(Pt)

        def logVrais_Mankamo(X):
            logPx, Cco, Cx = X
            #print('logVrais_Mankamo : logPx, Cco, Cx = ', logPx, Cco, Cx)
            # variables (pi, db, dx, dR, y_xm=1-dR)
            pi_weight, db, dx, dR, y_xm = self.computeGeneralParamFromMankamo([Pt, math.exp(logPx), Cco, Cx])
            self.setGeneralParameter([pi_weight, db, dx, dR, y_xm])
            #print('logVrais_Mankamo :[pi_weight, db, dx, dR, y_xm] = ', pi_weight, db, dx, dR, y_xm)
            #print('logVrais_Mankamo : self.generalParameter = ', self.generalParameter)
            S = 0.0
            for k in range(self.n+1):
                valImpactvect = self.totalImpactVector[k]
                if valImpactvect != 0:
                    val = self.computePEG(k)
                    log_PEG_k = math.log(val)
                    S += self.totalImpactVector[k] * log_PEG_k
            return [S]

        def func_constraints(X):
            logPx, Cco, Cx = X
            terme1 = ot.DistFunc.pNormal(-math.sqrt(1-Cx))
            terme2 = (Pt-0.5)/(1-1/(2*terme1))
            terme_min = math.log(min(terme1, terme2))

            # X respects the constraints if 
            # logPx < terme_min <==> terme_min -logPx > 0
            # we impose that terme_min -logPx >= eps|terme_min| > 0 <==> terme_min - eps|terme_min|-logPx>=0
            # <==> (1+eps)*terme_min-logPx >= 0
            # return un Point
            return [(1+eps)*terme_min-logPx]

        maFct_cont = ot.PythonFunction(3, 1, func_constraints)
        maFctLogVrais_Mankamo = ot.PythonFunction(3,1,logVrais_Mankamo)


        # test des res de Mankamo [logPx, Cco, Cx]
        #print('Best LogLik de Mankamo = ', maFctLogVrais_Mankamo([math.log(5.7e-7), 0.51, 0.85]))

        ######################################
        # Maximisation de la vraisemblance

        eps = 1e-9
        optimPb = ot.OptimizationProblem(maFctLogVrais_Mankamo)
        optimPb.setMinimization(False)
        # contraintes sur (Px, Cco, Cx): maFct_cont >= 0
        optimPb.setInequalityConstraint(maFct_cont)
        # bounds sur (logPx, Cco, Cx)
        boundsParam = ot.Interval([-35, eps, eps], [logPt, 1.0-eps, 1.0-eps])
        #print('boundsParam = ', boundsParam)
        optimPb.setBounds(boundsParam)
        # algo Cobyla pour ne pas avoir les gradients
        myAlgo = ot.Cobyla(optimPb)
        myAlgo.setRhoBeg(1e-1)
        myAlgo.setMaximumEvaluationNumber(10000)
        myAlgo.setMaximumIterationNumber(10000)
        myAlgo.setMaximumConstraintError(1e-5)
        myAlgo.setMaximumAbsoluteError(1e-5)
        myAlgo.setMaximumRelativeError(1e-5)
        myAlgo.setMaximumResidualError(1e-5)

        # Point de départ:
        # startPoint = [logPx, Cco, Cx]
        startPoint = [math.log(startingPoint[0]), startingPoint[1], startingPoint[2]]
        myAlgo.setStartingPoint(startPoint) 
        if verbose:
            print('Mon point de départ verifie-t il les contraintes? ')
            if maFct_cont(startPoint)[0] > 0.0:
                print('oui!')
            else:
                print('non')

        if verbose:
            ot.Log.Show(ot.Log.INFO)
        myAlgo.setVerbose(verbose)

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
        mankamoParam = [Pt, Px_optim, Cco_optim, Cx_optim]
        self.setMankamoParameter(mankamoParam)
        # General parameter = (pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim)
        generalParam = self.computeGeneralParamFromMankamo(mankamoParam)
        self.setGeneralParameter(generalParam)

        ######################################
        # Graphes de la log vraisemblance avec point optimal

        g_fixedlogPxCco, g_fixedlogPxCx, g_fixedCcoCx, g_fixedCx, g_fixedCco, g_fixedlogPx = [None]*6

        if visuLikelihood:
            print('Production of graphs')
            maFct_cont_fixedlogPx = ot.ParametricFunction(maFct_cont, [0], [logPx_optim])
            maFct_cont_fixedCco = ot.ParametricFunction(maFct_cont, [1], [Cco_optim])
            maFct_cont_fixedCx = ot.ParametricFunction(maFct_cont, [2], [Cx_optim])
            maFctLogVrais_Mankamo_fixedlogPx =ot. ParametricFunction(maFctLogVrais_Mankamo, [0], [logPx_optim])
            maFctLogVrais_Mankamo_fixedCco = ot.ParametricFunction(maFctLogVrais_Mankamo, [1], [Cco_optim])
            maFctLogVrais_Mankamo_fixedCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [2], [Cx_optim])
            maFctLogVrais_Mankamo_fixedlogPxCco = ot.ParametricFunction(maFctLogVrais_Mankamo, [0, 1], [logPx_optim, Cco_optim])
            maFctLogVrais_Mankamo_fixedCcoCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [1,2], [Cco_optim, Cx_optim])
            maFctLogVrais_Mankamo_fixedlogPxCx = ot.ParametricFunction(maFctLogVrais_Mankamo, [0,2], [logPx_optim, Cx_optim])

            coef = 0.1
            logPx_inf = (1-coef)*logPx_optim
            logPx_sup = (1+coef)*logPx_optim
            Cco_inf = (1-coef)*Cco_optim
            Cco_sup = (1+coef)*Cco_optim
            Cx_inf = (1-coef)*Cx_optim
            Cx_sup = (1+coef)*Cx_optim
            NbPt = 128
            NbPt2 = 64


            ####################
            # graphe (logPx) pour Cco = Cco_optim et Cx = Cx_optim
            # graphe de la loglikelihood
            print('graph (Cco, Cx) = (Cco_optim, Cx_optim)')
            g_fixedCcoCx = maFctLogVrais_Mankamo_fixedCcoCx.draw(logPx_inf, logPx_sup, NbPt)

            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, maFctLogVrais_Mankamo_fixedCcoCx([logPx_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCcoCx.add(myCloud)
            g_fixedCcoCx.setXTitle(r'$\log P_x$')
            g_fixedCcoCx.setTitle(r'Log likelihood at $(C_{co}, C_{x}) = $'+ format(Cco_optim,'.2E') + ',' +  format(Cx_optim,'.2E'))

            ####################
            # graphe (Cco) pour log Px = log Px_optim et Cx = Cx_optim
            # graphe de la loglikelihood
            print('graph (logPx, Cx) = (logPx_optim, Cx_optim)')
            g_fixedlogPxCx = maFctLogVrais_Mankamo_fixedlogPxCx.draw(Cco_inf, Cco_sup, NbPt)

            # + point optimal
            pointOptim = ot.Sample(1, [Cco_optim, maFctLogVrais_Mankamo_fixedlogPxCx([Cco_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPxCx.add(myCloud)
            g_fixedlogPxCx.setXTitle(r'$C_{co}$')
            g_fixedlogPxCx.setTitle(r'Log likelihood at $(\log P_{x}, C_{x}) = $'+ format(logPx_optim,'.2E') + ',' +  format(Cx_optim,'.2E'))

            ####################
            # graphe (Cx) pour logPx = logPx_optim et Cco = Cco_optim
            # graphe de la loglikelihood
            print('graph (logPx, Cco) = (logPx_optim, Cco_optim)')
            g_fixedlogPxCco = maFctLogVrais_Mankamo_fixedlogPxCco.draw(Cx_inf, Cx_sup, NbPt)

            # + point optimal
            pointOptim = ot.Sample(1, [Cx_optim, maFctLogVrais_Mankamo_fixedlogPxCco([Cx_optim])[0]])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPxCco.add(myCloud)
            g_fixedlogPxCco.setXTitle(r'$C_x$')
            g_fixedlogPxCco.setTitle(r'Log likelihood at $(\log P_{x}, C_{co}) = $'+ format(logPx_optim,'.2E') + ',' +  format(Cco_optim,'.2E'))


            ####################
            # graphe (Px, Cco) pour Cx = Cx_optim
            # fonction contrainte
            print('graph Cx = Cx_optim')
            g_constraint = maFct_cont_fixedCx.draw([logPx_inf, Cco_inf], [logPx_sup, Cco_sup], [NbPt2]*2)
            dr = g_constraint.getDrawable(0)
            dr.setLegend('constraint')
            dr.setLineStyle('dashed')
            dr.setColor('black')
            g_constraint.setDrawables([dr])
            # niveau de la loglikelihood
            g_fixedCx = maFctLogVrais_Mankamo_fixedCx.draw([logPx_inf, Cco_inf], [logPx_sup, Cco_sup], [NbPt2]*2)
            g_fixedCx.add(g_constraint)

            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, Cco_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCx.add(myCloud)
            g_fixedCx.setXTitle(r'$\log P_x$')
            g_fixedCx.setYTitle(r'$C_{co}$')
            g_fixedCx.setTitle(r'Log likelihood at $C_{x} = $'+ format(Cx_optim,'.2E'))

            ####################
            # graphe (logPx, Cx) pour Cco = Cco_optim
            # fonction contrainte
            print('graph Cco = Cco_optim')
            g_constraint = maFct_cont_fixedCco.draw([logPx_inf, Cx_inf], [logPx_sup, Cx_sup], [NbPt2]*2)
            dr = g_constraint.getDrawable(0)
            dr.setLegend('constraint')
            dr.setLineStyle('dashed')
            dr.setColor('black')
            g_constraint.setDrawables([dr])
            # niveau de la loglikelihood
            g_fixedCco = maFctLogVrais_Mankamo_fixedCco.draw([logPx_inf, Cx_inf], [logPx_sup, Cx_sup], [NbPt2]*2)
            g_fixedCco.add(g_constraint)

            # + point optimal
            pointOptim = ot.Sample(1, [logPx_optim, Cx_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedCco.add(myCloud)
            g_fixedCco.setXTitle(r'$\log P_x$')
            g_fixedCco.setYTitle(r'$C_{x}$')
            g_fixedCco.setTitle(r'Log likelihood at $C_{co} = $'+ format(Cco_optim,'.2E'))

            ####################
            # graphe (Cco, Cx) pour logPx = logPx_optim
            # fonction contrainte
            print('graph logPx = logPx_optim')
            g_constraint = maFct_cont_fixedlogPx.draw([Cco_inf, Cx_inf], [Cco_sup, Cx_sup], [NbPt2]*2)
            dr = g_constraint.getDrawable(0)
            dr.setLegend('constraint')
            dr.setLineStyle('dashed')
            dr.setColor('black')
            g_constraint.setDrawables([dr])
            # niveau de la loglikelihood
            g_fixedlogPx = maFctLogVrais_Mankamo_fixedlogPx.draw([Cco_inf, Cx_inf], [Cco_sup, Cx_sup], [NbPt2]*2)
            g_fixedlogPx.add(g_constraint)

            # + point optimal
            pointOptim = ot.Sample(1, [Cco_optim, Cx_optim])
            myCloud = ot.Cloud(pointOptim, 'black', 'bullet')
            g_fixedlogPx.add(myCloud)
            g_fixedlogPx.setXTitle(r'$C_{co}$')
            g_fixedlogPx.setYTitle(r'$C_{x}$')
            g_fixedlogPx.setTitle(r'Log likelihood at $\log P_{x} = $'+ format(logPx_optim,'.2E'))

        return mankamoParam, generalParam, finalLogLikValue, [g_fixedlogPxCco, g_fixedlogPxCx, g_fixedCcoCx, g_fixedCx, g_fixedCco, g_fixedlogPx]


    
    def computeGeneralParamFromMankamo(self, mankamoParam):
        """
        Compute the general parameter :math:`(\pi, d_b, d_x, d_R, y_{xm})` from the Mankamo parameter :math:`(P_t, P_x, C_{co}, C_x)` under the Mankamo assumption (:eq:`mankamoHyp`).

        Parameters
        ----------
        mankamoParam :  list of float
            The point :math:`(P_t, P_x, C_{co}, C_x)`
        
        Returns
        -------
        generalParam : list of float
            The point :math:`(\pi, d_b, d_x, d_R, y_{xm})`

        Notes
        -----
        The general parameter  :math:`(\pi, d_b, d_x, d_R, y_{xm})` is computed from the parameter  :math:`(P_t, P_x, C_{co}, C_x)` under the Mankamo assumption where :math:`y_{xm} = 1-d_R`, using equations (:eq:`Parma2to1Mankamo`).
        """
        
        Pt, Px, Cco, Cx = mankamoParam
        pi_weight = 1- Px/ot.DistFunc.pNormal(-math.sqrt(1-Cx))
        db = -math.sqrt(Cco)/ot.DistFunc.qNormal((Pt-Px)/pi_weight)
        dR = -math.sqrt(1-Cco)/ot.DistFunc.qNormal((Pt-Px)/pi_weight)
        dx = dR*math.sqrt(Cx/(1-Cx))
        yxm = 1-dR

        return [pi_weight, db, dx, dR, yxm]

    
    def computePEG(self, k):
        """
        Compute the :math:`\mbox{PEG}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Order of the common failure.
        
        Returns
        -------
        peg_kn : float,  :math:`0 \leq  \mbox{PEG}(k|n) \leq 1`
            The :math:`\mbox{PEG}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mbox{PEG}(k|n)` is computed using (:eq:`PEG_red`).
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')

        pi_weight, db, dx, dR, y_xm = self.generalParameter
        
        # Numerical range of the  Normal() distribution
        val_min = -7.65
        val_max = 7.65

        def kernel_b(yPoint):
            y = yPoint[0]
            terme1 = pi_weight/db * ot.DistFunc.dNormal(y/db)
            temp = ot.DistFunc.pNormal((y-1)/dR)
            terme2 = math.pow(temp, k)
            # tail CDF
            terme3 = math.pow(1.0-temp, self.n-k)
            return [terme1 * terme2 * terme3]

        def kernel_x(yPoint):
            y = yPoint[0]
            terme1 = (1-pi_weight)/dx*ot.DistFunc.dNormal((y-y_xm)/dx)
            temp = ot.DistFunc.pNormal((y-1)/dR)
            terme2 = math.pow(temp, k)
            # tail CDF
            terme3 = math.pow(1.0-temp, self.n-k)
            #print('(x) y=', y, 'terme1, terme2, terme3 = ', terme1, terme2, terme3)
            return [terme1 * terme2 * terme3]

        maFctKernel_b = ot.PythonFunction(1,1,kernel_b)
        maFctKernel_x = ot.PythonFunction(1,1,kernel_x)

        # Numerical integration interval
        yMin_b = max(val_min*db, 1.0+dR*val_min)
        yMax_b = min(val_max*db, 1.0+dR*val_max)
        myInterval_b = ot.Interval(yMin_b, yMax_b)
        yMin_x = max(val_min*dx+y_xm, 1.0+dR*val_min)
        yMax_x = min(val_max*dx+y_xm, 1.0+dR*val_max)
        myInterval_x = ot.Interval(yMin_x, yMax_x)
        # base load part integration
        int_b = 0.0
        if yMin_b < yMax_b:
            int_b =  self.integrationAlgo.integrate(maFctKernel_b, myInterval_b)[0]
        # extreme load part integration
        int_x = 0.0
        if yMin_x < yMax_x:    
            int_x =  self.integrationAlgo.integrate(maFctKernel_x, myInterval_x)[0]

        PEG = int_b + int_x
        return PEG



    def  computePEGall(self):
        """
        Compute all the :math:`\mbox{PEG}(k|n)` probabilities for :math:`0 \leq k \leq n`.
        
        Returns
        -------
        peg_list : seq of float,  :math:`0 \leq  \mbox{PEG}(k|n)`
            The :math:`\mbox{PEG}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mbox{PEG}(k|n)` probabilities are computed using (:eq:`PEG_red`).
        """

        PEG_list = list()
        for k in range(self.n+1):
            PEG_list.append(self.computePEG(k))
        return PEG_list
    
   
    def computePSG1(self):
        """
        Compute the :math:`\mbox{PSG}(1|n)` probability.
        
        Returns
        -------
        psg_1n : float,  :math:`0 \leq  \mbox{PSG}(1|n) \leq 1`
            The :math:`\mbox{PSG}(1|n)`  probability.

        Notes
        -----
        The  :math:`\mbox{PSG}(1|n)` is computed using:

        ..math::
            :label:`PSG1_red`
        
            \begin{array}{lcl}
               PSG(1) & = & \int_{-\infty}^{+\infty}\left[ \dfrac{\pi}{d_b} \varphi
         \left(\dfrac{y}{d_b}\right) \right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy +  \int_{-\infty}^{+\infty} \left[  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy \\
                     & = & \pi \left[\textcolor{red}{1-}\Phi\left(\dfrac{1}{\sqrt{d_b^2+d_R^2}}\right)\right] +  (1-\pi) \left[\textcolor{red}{1-}\Phi\left(\dfrac{1-y_{xm}}{\sqrt{d_x^2+d_R^2}}\right)\right]
            \end{array}

        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')
        
        pi_weight, db, dx, dR, y_xm = self.generalParameter
        
        # PSG(1) = Pb + Px
        val_b = math.sqrt(db*db+dR*dR)
        val_x = math.sqrt(dx*dx+dR*dR)
        Pb = pi_weight * ot.DistFunc.pNormal(-1.0/val_b)
        Px = (1-pi_weight) * ot.DistFunc.pNormal(-(1-y_xm)/val_x)
        return Pb+Px


      

    def computePSG(self, k):
        """
        Compute the :math:`\mbox{PSG}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Order of the common failure.
        
        Returns
        -------
        psg_kn : float,  :math:`0 \leq  \mbox{PSG}(k|n) \leq 1`
            The :math:`\mbox{PSG}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mbox{PSG}(k|n)` is computed using fo :math:`k !=1`:

        ..math::
            :label:`PSG_red` 

            \mbox{PSG}(k) =  \int_{-\infty}^{+\infty} \left[ \dfrac{\pi}{d_b} \varphi \left(\dfrac{y}{d_b}\right) +  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right] \left[\Phi\left(\dfrac{y-1}{d_R}\right)\right]^k  \, dy

        and for :math:`k=1`:

        ..math::
            :label:`PSG1_red`
        
            \begin{array}{lcl}
               PSG(1) & = & \int_{-\infty}^{+\infty}\left[ \dfrac{\pi}{d_b} \varphi
         \left(\dfrac{y}{d_b}\right) \right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy +  \int_{-\infty}^{+\infty} \left[  \dfrac{(1-\pi)}{d_x}\varphi \left(\dfrac{y-y_{xm}}{d_x}\right)\right]\left[\Phi\left(\dfrac{y-1}{d_R}\right)\right] \, dy \\
                     & = & \pi \left[\textcolor{red}{1-}\Phi\left(\dfrac{1}{\sqrt{d_b^2+d_R^2}}\right)\right] +  (1-\pi) \left[\textcolor{red}{1-}\Phi\left(\dfrac{1-y_{xm}}{\sqrt{d_x^2+d_R^2}}\right)\right]
            \end{array}    
        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')
        
        pi_weight, db, dx, dR, y_xm = self.generalParameter
        
        # specal computation on that case
        if k==1:
            return self.computePSG1()

        # range numerique de la Normal()
        val_min = -7.65
        val_max = 7.65

        def kernel_b(yPoint):
            y = yPoint[0]
            terme1 = pi_weight/db * ot.DistFunc.dNormal(y/db) 
            terme2 = math.pow(ot.DistFunc.pNormal((y-1)/dR), k)
            #print('terme1, terme2 = ', terme1, terme2)
            return [terme1 * terme2]

        def kernel_x(yPoint):
            y = yPoint[0]
            terme1 = (1-pi_weight)/dx*ot.DistFunc.dNormal((y-y_xm)/dx)
            terme2 = math.pow(ot.DistFunc.pNormal((y-1)/dR), k)
            #print('terme1, terme2 = ', terme1, terme2)
            return [terme1 * terme2]

        maFctKernel_b = ot.PythonFunction(1,1,kernel_b)
        maFctKernel_x = ot.PythonFunction(1,1,kernel_x)

        # le range num de la loi Normale() est [val_min, val_max] = [-7.65, 7.65] 
        yMin_b = max(val_min*db, 1.0+dR*val_min)
        yMax_b = min(val_max*db, 1.0+dR*val_max)
        myInterval_b = ot.Interval(yMin_b, yMax_b)
        yMin_x = max(val_min*dx+y_xm, 1.0+dR*val_min)
        yMax_x = min(val_max*dx+y_xm, 1.0+dR*val_max)
        myInterval_x = ot.Interval(yMin_x, yMax_x)
        #print('myInterval_b, myInterval_x = ', myInterval_b, myInterval_x)
        # integrate retourne un Point
        # integrale partie b
        int_b = 0.0
        if yMin_b < yMax_b:
            int_b =  self.integrationAlgo.integrate(maFctKernel_b, myInterval_b)[0]
        # integrale partie x
        int_x = 0.0
        if yMin_x < yMax_x:    
            int_x =  self.integrationAlgo.integrate(maFctKernel_x, myInterval_x)[0]
        #print('int_b, int_x,  = ', int_b, int_x)

        PSG = int_b + int_x
        return PSG

    
      
    def  computePSGall(self):
        """
        Compute all the :math:`\mbox{PSG}(k|n)` probabilities for :math:`0 \leq k \leq n`.
        
        Returns
        -------
        psg_list : seq of float,  :math:`0 \leq  \mbox{PSG}(k|n)`
            The :math:`\mbox{PSG}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mbox{PSG}(k|n)` probabilities are computed using (:eq:`PSG_red`) for :math:`k != 1` and (:eq:`PSG1_red`) for :math:`k = 1`.
        """

        PSG_list = list()
        for k in range(self.n+1):
            PSG_list.append(self.computePSG(k))
        return PSG_list

        
    def computePES(self, k):
        """
        Compute the :math:`\mbox{PES}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Order of the common failure.
        
        Returns
        -------
        pes_kn : float,  :math:`0 \leq  \mbox{PES}(k|n) \leq 1`
            The :math:`\mbox{PES}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mbox{PES}(k|n)` is computed using  for :math:`0 \leq k \leq n`:

        ..math::
            :eq:`PES_red`

            \mbox{PES}(k|n) = C_n^k \, \mbox{PEG}(k|n)

        """

        if self.generalParameter is None:
            raise Exception('The general parameter has not been estimated!')
        
        pi_weight, db, dx, dR, y_xm = self.generalParameter
        
        PEG = self.computePEG(k)
        PES = math.comb(self.n,k)*PEG
        return PES


    def  computePESall(self):
        """
        Compute all the :math:`\mbox{PES}(k|n)` probabilities for :math:`0 \leq k \leq n`.
        
        Returns
        -------
        pes_list : seq of float,  :math:`0 \leq  \mbox{PES}(k|n)`
            The :math:`\mbox{PES}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mbox{PES}(k|n)` probabilities are computed using (:eq:`PES_red`).
        """

        PES_list = list()
        for k in range(self.n+1):
            PES_list.append(self.computePES(k))
        return PES_list

                      

    def computePTS(self, k):
        """
        Compute the :math:`\mbox{PTS}(k|n)` probability.

        Parameters
        ----------
        k : int, :math:`0 \leq k \leq n`
            Order of the common failure.
        
        Returns
        -------
        pts_kn : float,  :math:`0 \leq  \mbox{PTS}(k|n) \leq 1`
            The :math:`\mbox{PTS}(k|n)`  probability.

        Notes
        -----
        The  :math:`\mbox{PTS}(k|n)` is computed using  for :math:`0 \leq k \leq n`:

        ..math::
            :eq:`PTS_red`

             \mbox{PTS}(k|n) = \sum_{i=k}^n  \mbox{PES}(i|n)

        where  the :math:`\mbox{PES}(i|n)` probability is computed using :eq:`PES_red`.
        """

        
        PTS = 0.0
        for i in range(k,self.n+1):
            PTS += self.computePES(i)
        return PTS

                            

    def  computePTSall(self):
        """
        Compute all the :math:`\mbox{PTS}(k|n)` probabilities for :math:`0 \leq k \leq n`.
        
        Returns
        -------
        pts_list : seq of float,  :math:`0 \leq  \mbox{PTS}(k|n)`
            The :math:`\mbox{PTS}(k|n)`  probabilities for :math:`0 \leq k \leq n`.

        Notes
        -----
        All the  :math:`\mbox{PTS}(k|n)` probabilities are computed using (:eq:`PS_red`).
        """

        PTS_list = list()
        for k in range(self.n+1):
            PTS_list.append(self.computePTS(k))
        return PTS_list



    def estimateBootstrapParamSampleFromMankamo(self, Nb, startingPoint, blockSize, fileNameRes):
        """
        Generates a Bootstrap sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})` using the Mankamo assumption :math:`y_{xm} = 1-d_R`.
        
        Parameters
        ----------
        Nb : int
            The size of the sample generated.
        startingPoint : :class:`~openturns.Point`
            Start point :math:`(P_t, P_x, C_{co}, C_x)` for the optimization problem.
        blockSize : int,
            The block size after which the sample is saved. 
        fileNameRes: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})` using the Mankamo assumption :math:`y_{xm} = 1-d_R`.

        Notes
        -----
        The Mankamo parameter sample is obtained by bootstraping the empirical law of the total impact vector :math:`N_b` times. The total emprical law os the MultiNoamil law parametrerized by the empirical probabilities :math:`[p_0^{emp},
\dots, p_n^{emp}]` where :math:`p_k^{emp} = \dfrac{V_t^{n,N}[k]}{nN}` and :math:`N` is the number of tests and demands on the whole group. Then the optimisatio prblem (:eq:`optimMankamo`) is solved using the specifiedstarting point.

        The function calls the script script_bootstrap_ParamFromMankamo.py that uses the parallelisation of the pool object of the multiprocessing module. It creates a file :math:`myECLM.xml` that stores the total impact vector to be read by the script.

        The computation is saved in the csv file named fileNameRes every blocksize calculus. The computation can be interrupted: it will be restarted from the last filenameRes saved.
        """

        myStudy = ot.Study('myECLM.xml')
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.add('startingPoint', ot.Point(startingPoint))
        myStudy.save()

        import os
        command =  'python script_bootstrap_ParamFromMankamo.py {} {} {}'.format(Nb, blockSize, fileNameRes)         
        os.system(command)

    

    def computeECLMProbabilitiesFromMankano(self, blockSize, fileNameInput, fileNameRes):
        """
        Computes the sample of all the ECLM probabilities from a sample of Mankamo parameters using the Mankamo assumption (:eq:`mankamoHyp`).
        
        Parameters
        ----------
        blockSize : int,
            The block size after which the sample is saved. 
        fileNameInput: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`
        fileNameRes: string,
            The csv file that stores the ECLM probabilities.

        Notes
        -----
        The ECLM probabilities are computed according to the order :math:`(\mbox{PEG}(0|n), \dots, \mbox{PEG}(n|n), \mbox{PSG}(0|n), \dots, \mbox{PSG}(n|n), \mbox{PES}(0|n), \dots, \mbox{PES}(n|n), \mbox{PTS}(0|n), \dots, \mbox{PTS}(n|n))` using equantions (:eq:`PEG_red`), (:eq:`PSG_red`), (:eq:`PES_red`),(:eq:`PTS_red`), using the Mankamo assumption :math:`y_{xm} = 1-d_R`.

        The function calls the script script_bootstrap_ECLMProbabilities.py that uses the parallelisation of the pool object of the multiprocessing module. It creates a file :math:`myECLM.xml` that stores the total impact vector to be read by the script.

        The computation is saved in the csv file named fileNameRes every blocksize calculus. The computation can be interrupted: it will be restarted from the last filenameRes saved.
        """

        myStudy = ot.Study('myECLM.xml')
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.save()

        import os
        command =  'python script_bootstrap_ECLMProbabilities.py {} {} {} {}'.format(self.n, blockSize, fileNameInput, fileNameRes)
        os.system(command)

        
    def analyse_graphsECLMParam(self, fileNameSample):
        """
        Produces graphs to analyse a sample of :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.

        Parameters
        ----------
        fileNameSample: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`

        Returns
        -------
        graphPairsMankamoParam : :class:`~openturns.Graph`,
            The Pairs graph of the Mankamo parameter  :math:`(P_t, P_x, C_{co}, C_x)`

        graphPairsGeneralParam : :class:`~openturns.Graph`,
            The Pairs graph of the general parameter  :math:`(\pi, d_b, d_x, d_R, y_{xm})`
        
        graphMarg_list : list of :class:`~openturns.Graph`,
            The list of the marginal pdf of the Mankamo parameter and the general parameter.
        
        descParam: :class:`~openturns.Description`,
            Description of each param.

        Notes
        -----
        The list of graph  graphMarg_list is given following the order :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.
        Each distribution is approximated with a Histogrm and a gaussian kernel smoothing.
        """

        sampleParamAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        sampleParamRed = sampleParamAll.getMarginal([0,1,2,3])
        sampleParamInit = sampleParamAll.getMarginal([4,5,6,7,8])
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
            KS.setUpperBound(1.0)
            KS_dist = KS.build(sample)
            graph = Histo.drawPDF()
            graph.add(KS_dist.drawPDF())
            graph.setColors(['blue', 'red'])
            graph.setLegends(['Histo', 'KS'])
            graph.setLegendPosition('topright')
            graph.setXTitle(descParam[k])
            graphMarg_list.append(graph)

        return graphPairsMankamoParam, graphPairsGeneralParam, graphMarg_list, descParam

                            
    def analyse_graphsECLMProbabilities(self, fileNameSample, kMax):
        """
        Produces graphs to analyse a sample of all the ECL probabilities.

        Parameters
        ----------
        fileNameSample: string,
            The csv file that stores the ECLM probabilities.

        kMax : int, :math:`kMx \leq 1`
            The maximal order studied.

        Returns
        -------
        graphPairs_list : list of :class:`~openturns.Graph`,
            The Pairs graph of the ECLM probabilities.

        graphPEG_PES_PTS_list : list of :class:`~openturns.Graph`,
            The Pairs graph of the probabilities :math:`(\mbox{PEG}(k|n),\mbox{PES}(k|n), \mbox{PTS}(k|n))` for :math:`0 \leq k \leq kMax`.
        
        graphMargPEG_list : list of :class:`~openturns.Graph`,
            The list of the marginal pdf of the  :math:`\mbox{PEG}(k|n)` probabilities  for :math:`0 \leq k \leq kMax`.
        
        graphMargPSG_list : list of :class:`~openturns.Graph`,
            The list of the marginal pdf of the  :math:`\mbox{PSG}(k|n)` probabilities  for :math:`0 \leq k \leq kMax`.
        
        graphMargPES_list : list of :class:`~openturns.Graph`,
            The list of the marginal pdf of the  :math:`\mbox{PES}(k|n)` probabilities  for :math:`0 \leq k \leq kMax`.
        
        graphMargPTS_list : list of :class:`~openturns.Graph`,
            The list of the marginal pdf of the  :math:`\mbox{PTS}(k|n)` probabilities  for :math:`0 \leq k \leq kMax`.
        
        desc_list: :class:`~openturns.Description`,
            Description of each graph.

        Notes
        -----
        Each distribution is approximated with a Histogrm and a gaussian kernel smoothing.        
        """

        sampleProbaAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        desc = sampleProbaAll.getDescription()
        dim = sampleProbaAll.getDimension()
        # dim = 4(n+1)
        n = int(dim/4)-1
        samplePEG = sampleProbaAll.getMarginal([k for k in range(n+1)])
        descPEG = samplePEG.getDescription()
        samplePSG = sampleProbaAll.getMarginal([k for k in range(n+1, 2*n+2)])
        descPSG = samplePSG.getDescription()
        samplePES = sampleProbaAll.getMarginal([k for k in range(2*n+2, 3*n+3)])
        descPES = samplePES.getDescription()
        samplePTS = sampleProbaAll.getMarginal([k for k in range(3*n+3, 4*n+4)])
        descPTS = samplePTS.getDescription()

        descPairs = ot.Description()

        # Graphe Pairs sur les PEG(k|n)
        graphPairsPEG = ot.VisualTest.DrawPairs(samplePEG.getMarginal([k for k in range(kMax+1)]))
        descPairs.add('Pairs_PEG')
        # Graphe Pairs sur les PSG(k|n)
        graphPairsPSG = ot.VisualTest.DrawPairs(samplePSG.getMarginal([k for k in range(kMax+1)]))
        descPairs.add('Pairs_PSG')
        # Graphe Pairs sur les PES(k|n)
        graphPairsPES = ot.VisualTest.DrawPairs(samplePES.getMarginal([k for k in range(kMax+1)]))
        descPairs.add('Pairs_PES')
        # Graphe Pairs sur les PTS(k|n)
        graphPairsPTS = ot.VisualTest.DrawPairs(samplePTS.getMarginal([k for k in range(kMax+1)]))
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
            KS.setUpperBound(1.0)
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
            KS.setUpperBound(1.0)
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
            KS.setUpperBound(1.0)
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
            KS.setUpperBound(1.0)
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

                            

    def analyse_distECLMProbabilities(self, fileNameSample, kMax, confidenceLevel, factoryColl):
        """
        Fits a distribution on a sample of all the ECL probabilities.

        Parameter
        ---------
        fileNameSample: string,
            The csv file that stores the ECLM probabilities.
        kMax : int, :math:`kMax \leq 1`,
            The maximal order studied.
        confidenceLevel : float, :math:`0 \leq confidenceLevel \leq 1`,
            The confidence level of ech interval.
        factoryCollection : list of :class:`~openturns.DistributionFactory`,
            List of factories that will be used to fit a distribution to the sample.        
        desc_list: :class:`~openturns.Description`,
            Description of each graph.

        Returns
        -------
        confidenceInterval_list : list of :class:`openturns.Interval`,
            The confidence intervals of the ECLM probabilities.
        graph_marg_list : list of  :class:`openturns.Graph`,
            The fitting graph of each ECLM probability.

        Notes
        -----
        The confidence intervals and the graphs illustrating the fitting are given following the following order:  :math:`(\mbox{PEG}(0|n), \dots, \mbox{PEG}(n|n), \mbox{PSG}(0|n), \dots, \mbox{PSG}(n|n), \mbox{PES}(0|n), \dots, \mbox{PES}(n|n), \mbox{PTS}(0|n), \dots, \mbox{PTS}(n|n))`.

        Each fitting is tested using the Lilliefors test. The result is printed and the best model aong the list of factories is given. Care: it is not guaranted that the best model is accepted by the Lilliefors test.
        """

        sampleProbaAll = ot.Sample.ImportFromCSVFile(fileNameSample)
        desc = sampleProbaAll.getDescription()
        dim = sampleProbaAll.getDimension()
        # dim = 4(n+1)
        n = int(dim/4)-1
        samplePEG = sampleProbaAll.getMarginal([k for k in range(n+1)])
        descPEG = samplePEG.getDescription()
        samplePSG = sampleProbaAll.getMarginal([k for k in range(n+1, 2*n+2)])
        descPSG = samplePSG.getDescription()
        samplePES = sampleProbaAll.getMarginal([k for k in range(2*n+2, 3*n+3)])
        descPES = samplePES.getDescription()
        samplePTS = sampleProbaAll.getMarginal([k for k in range(3*n+3, 4*n+4)])
        descPTS = samplePTS.getDescription()

        KS = ot.KernelSmoothing()
        KS.setBoundaryCorrection(True)
        KS.setBoundingOption(ot.KernelSmoothing.BOTH)
        KS.setLowerBound(0.0)
        KS.setUpperBound(1.0)

        IC_PEG_list = list()
        IC_PSG_list = list()
        IC_PES_list = list()
        IC_PTS_list = list()

        quantSup = 0.5 + confidenceLevel/2
        quantInf = 0.5 - confidenceLevel/2

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

            best_model_PEG_k, best_result_PEG_k = ot.FittingTest.BestModelLilliefors(samplePEG_k, factoryColl)
            best_model_PSG_k, best_result_PSG_k = ot.FittingTest.BestModelLilliefors(samplePSG_k, factoryColl)
            best_model_PES_k, best_result_PES_k = ot.FittingTest.BestModelLilliefors(samplePES_k, factoryColl)
            best_model_PTS_k, best_result_PTS_k = ot.FittingTest.BestModelLilliefors(samplePTS_k, factoryColl)

            print('Ordre k=', k)
            print('Best model PEG(', k, '|n) : ', best_model_PEG_k, 'p-value = ', best_result_PEG_k.getPValue())
            print('Best model PSG(', k, '|n) : ', best_model_PSG_k, 'p-value = ', best_result_PSG_k.getPValue())
            print('Best model PES(', k, '|n) : ', best_model_PES_k, 'p-value = ', best_result_PES_k.getPValue())
            print('Best model PTS(', k, '|n) : ', best_model_PTS_k, 'p-value = ', best_result_PTS_k.getPValue())
            print('')

            ##############################
            # Graphe des ajustements
            # PEG
            Histo = ot.HistogramFactory().build(samplePEG_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PEG_k.drawPDF())
            leg.add('KS')
            nbFact = len(factoryColl)
            for i in range(nbFact):
                dist = factoryColl[i].build(samplePEG_k)
                graph.add(dist.drawPDF())
                leg.add(dist.getName())

            graph.setColors(colors[0:nbFact+2])
            graph.setLegends(leg)
            graph.setLegendPosition('topright')
            graph.setXTitle(descPEG[k])
            graph.setTitle('PEG('+str(k) + '|' + str(n) + ') - best model : ' +  str(best_model_PEG_k.getName()))
            graphMargPEG_list.append(graph)
            descMargPEG.add('PEG_'+str(k))

            # PSG
            Histo = ot.HistogramFactory().build(samplePSG_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PSG_k.drawPDF())
            leg.add('KS')
            nbFact = len(factoryColl)
            for i in range(nbFact):
                dist = factoryColl[i].build(samplePSG_k)
                graph.add(dist.drawPDF())
                leg.add(dist.getName())

            graph.setColors(colors[0:nbFact+2])
            graph.setLegends(leg)
            graph.setLegendPosition('topright')
            graph.setXTitle(descPSG[k])
            graph.setTitle('PSG('+str(k) + '|' + str(n) + ') - best model : ' +  str(best_model_PSG_k.getName()))
            graphMargPSG_list.append(graph)
            descMargPSG.add('PSG_'+str(k))

            # PES
            Histo = ot.HistogramFactory().build(samplePES_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PES_k.drawPDF())
            leg.add('KS')
            nbFact = len(factoryColl)
            for i in range(nbFact):
                dist = factoryColl[i].build(samplePES_k)
                graph.add(dist.drawPDF())
                leg.add(dist.getName())

            graph.setColors(colors[0:nbFact+2])
            graph.setLegends(leg)
            graph.setLegendPosition('topright')
            graph.setXTitle(descPES[k])
            graph.setTitle('PES('+str(k) + '|' + str(n) + ') - best model : ' +  str(best_model_PES_k.getName()))
            graphMargPES_list.append(graph)
            descMargPES.add('PES_'+str(k))

            # PTS
            Histo = ot.HistogramFactory().build(samplePTS_k)
            graph = Histo.drawPDF()
            leg = ot.Description(1,'Histo')
            graph.add(KS_dist_PTS_k.drawPDF())
            leg.add('KS')
            nbFact = len(factoryColl)
            for i in range(nbFact):
                dist = factoryColl[i].build(samplePTS_k)
                graph.add(dist.drawPDF())
                leg.add(dist.getName())

            graph.setColors(colors[0:nbFact+2])
            graph.setLegends(leg)
            graph.setLegendPosition('topright')
            graph.setXTitle(descPTS[k])
            graph.setTitle('PTS('+str(k) + '|' + str(n) + ') - best model : ' +  str(best_model_PTS_k.getName()))
            graphMargPTS_list.append(graph)
            descMargPTS.add('PTS_'+str(k))



        return [IC_PEG_list, IC_PSG_list, IC_PES_list, IC_PTS_list], [graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list] , [descMargPEG, descMargPSG, descMargPES, descMargPTS]



    def computeKMax_PTS(self, p):
        """
        Computes the minimal order with a probability greater than a given threshold.

        Parameter
        ---------
        p : float, :math:` 0 \leq p \leq 1`
            The probability threshold.

        Returns
        -------
        kMax : int,
            The minimal order with a probability greater than :math:`p`.

        Notes
        -----
        The :math:`k_{max}` order is the minimal order such that the probability that at least :math:`k_{max}` failures occurs is greater than :math:`p`. Then :math:`k_{max}` is defined by: 

        ..math::
            :label:`kMaxDef`
        
            k_{max}(p) = \max \{ k |  \mbox{PTS}(k|n) > p \}

        The probability :math:`\mbox{PTS}(k|n)` is computing using (:eq:`PTS_red`).

        """

        k=0
        while self.computePTS(k) > p:
            k+=1
        return k-1              


    
    def computeAnalyseKMaxSample(self, p, blockSize, fileNameInput, fileNameRes):
        """
        Generates a :math:`k_{max}` sample and produces graphs to analyse it.

        Parameter
        ---------
        p : float, :math:` 0 \leq p \leq 1`
            The probability threshold.

        blockSize : int,
            The block size after which the sample is saved. 

        fileNameInput: string,
            The csv file that stores the sample of  :math:`(P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})`.

        fileNameRes: string,
            The csv file that stores the sample of  :math:`k_{max}`.
     defined by (:eq:`kMaxDef`).

        Returns
        -------
        kmax_graph :  :class:`~openturns.Graph`,
            The empirical distribution of :math:`k_{max}`.

        Notes
        -----
        The function calls the script script_bootstrap_KMax.py that uses the parallelisation of the pool object of the multiprocessing module. It creates a file :math:`myECLM.xml` that stores the total impact vector to be read by the script.

        The computation is saved in the csv file named fileNameRes every blocksize calculus. The computation can be interrupted: it will be restarted from the last filenameRes saved.

        The empirical distribution is fitted on the sample. The $90\%$ confidence interval is given, computed from the empirical distribution.
        """

        myStudy = ot.Study('myECLM.xml')
        myStudy.add('totalImpactVector', ot.Indices(self.totalImpactVector))
        myStudy.save()

        import os
        command =  'python script_bootstrap_KMax.py {} {} {} {}'.format(p, blockSize, fileNameInput, fileNameRes)
        os.system(command)

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
        graph.setTitle(r'Loi de $K_{max} = $' + 'argmax {k | PTS(k|'+ str(self.n) + r'$) \geq $'+ format(p,'.1E') + r'}')

        # IC à 90%
        print('Intervalle de confiance de niveau 90%: [',  UD_dist_KMax.computeQuantile(0.05)[0], ', ', UD_dist_KMax.computeQuantile(0.95)[0], ']')

        return graph
    
                        
