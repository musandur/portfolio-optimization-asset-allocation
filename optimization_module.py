import numpy as np
import pandas as pd
import cvxpy as cvx
import statsmodels.api as sm

from abc import ABC, abstractmethod


# Optimization abstract class
class AbstractClassOptimalWeights(ABC):
    '''
    Abstract class as an Interface class for convex optimization methods.
    '''

    @abstractmethod
    def get_objective_func(self, weights, alpha_vector):

        raise NotImplementedError
    
    @abstractmethod
    def get_constraints(self, weights, factor_betas, risk):

        raise NotImplementedError
    
    def get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):

        B = factor_betas.loc[alpha_vector_index].values.T * weights
        F = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(B, F) + cvx.quad_form(weights, S)
    
    def find_optimal_holdings(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):

        weights = cvx.Variable(len(alpha_vector))
        
        risk = self.get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)

        objective_func = self.get_objective_func(weights, alpha_vector)
        constraints = self.get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        set_problem = cvx.Problem(objective_func, constraints)
        set_problem.solve()

        optimal_holdings = np.asarray(weights.value).flatten()

        return pd.DataFrame(data=optimal_holdings, index=alpha_vector.index)
    

# Optimization with regularization     
class OptimalHoldings(AbstractClassOptimalWeights):
    '''
    Derived class defining and implementing convex optimization with regularization.  
    
    '''

    def __init__(self, lambda_reg=0.01, risk_cap=0.05, factor_max=5.0, factor_min=-5.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        self.lambda_reg = lambda_reg
        
    def get_objective_func(self, weights, alpha_vector):

        assert (len(alpha_vector.columns) == 1)

        return cvx.Minimize(-alpha_vector.values.flatten() @ weights + self.lambda_reg * cvx.norm2(weights))
    
    def get_constraints(self, weights, factor_betas, risk):

        factor = factor_betas.T * weights

        constraints = [weights >= self.weights_min,
                       weights <= self.weights_max,
                       #weights >= 0,
                       sum(weights) == 0.0,
                       sum(cvx.abs(weights)) <= 1.0,
                       factor <= self.factor_max,
                       factor >= self.factor_min,
                       risk <= self.risk_cap**2]
                       
        return constraints


# Optimization function
def factor_betas_and_specific_return(fama_fac_and_return, factor_data):
    '''
    Function that estimates the factor betas and the specific returns. 
    
    INPUTS:
        fama_fac_and_return: Multi-Index DataFrame of the Fama-French 5-factors and the 10-day asset returns.
        factor_data: Fama-French 5-factors DataFrame.
    RETURN:
        betas: estimated factor betas dataFrame.
        specific_ret: specific returns dataFrame.
    '''
     
    betas = (fama_fac_and_return.groupby('Ticker', group_keys=False)
                    .apply(lambda x: sm.OLS(endog=x['return'], exog=sm.add_constant(x[fama_fac_and_return.columns[:-1]]))
                    .fit()
                    .params))
    
    betas = betas.loc[fama_fac_and_return.index.unique('Ticker')]

    if 'const' in betas.columns:
        betas = betas.drop(columns=['const'], axis=1)  

    date_idx = fama_fac_and_return['return'].index.unique('date')
    specific_ret = fama_fac_and_return['return'].unstack(0).subtract(factor_data.loc[date_idx, :].dot(betas.T))
     
    return betas, specific_ret


def get_optimal_weights(alpha_vector, fama_fac_and_return, factor_data):
    '''
    Function finding the optimal weights for an optimal portfolio
    
    INPUTS:
        alpha_vector: alpha values for  a selected day.
        fama_fac_and_return: Multi-Index DataFrame of the Fama-French 5-factors and the 10-day asset returns.
        factor_data: Fama-French 5-factors DataFrame.
    RETURN:
        opt_weights: Single column DataFrame of the optimal weights with underlying  as index.
    '''

    optimal_weigths = OptimalHoldings()

    factor_betas, specific_returns = factor_betas_and_specific_return(fama_fac_and_return, factor_data) 
    date_idx = fama_fac_and_return['return'].index.unique('date')
    factor_cov_matrix = np.sqrt(252)*np.cov(factor_data.loc[date_idx, :].T, ddof=1)
    idiosynchratic_var_vector = 252*specific_returns.var(ddof=1)

    opt_wights_df = optimal_weigths.find_optimal_holdings(alpha_vector, factor_betas, factor_cov_matrix, idiosynchratic_var_vector)
    opt_wights_df.rename(columns={0:'optimal_weights'}, inplace=True)

    return opt_wights_df

if __name__ == '__main__':
    print("Inplace Calling. No output expected!")