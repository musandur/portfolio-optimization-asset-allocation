
We rigorously design a Portfolio of stocks from end-to-end: data acquisition, alpha and risk factor modelling, and porfolio optimization.


- First, we build and evaluate an alpha factor strategy.
- Second, we model a risk factor using the Fama-French 5-factor model.
- Third we develop a portfolio optimization to construction the optimal portfolio.
- Finally step,  we proceed with asset allocation: by re-running the optimization routine and chosing wich assets to remove and wich assets to keep.

### The following modules are implemented for this project:

1. `alpha_strategy.py`:
This module implement two functions:

- `get_data.py`: extract the universe data.
- `model_alpha_factor.py`: implement the alpha factor model.

2. `optimization_module.py`
This module is responsible for building the optimization procedure via an abstract base class and a derived class. 

- An abstract class `class AbstractClassOptimalWeights(ABC):` serves as an interface for implementing different optimization routines by setting up the optimization problem and the risk model function, but abstract away  the constraints and the objective function.

-  A derived class `class OptimalHoldings(AbstractClassOptimalWeights):` for implementing the optimization with regulrization and the constrainsts.

- the function `get_optimal_weights()` calls the class and run the optimization procedure to get the optimal weights.

- The risk model inputs such as the factor betas and specific returns  have been estimated in the function `factor_betas_and_specific_return`.


### Main function

- The jupyter notebook file `portfolio_optimization.ipynb`is the main interface for data analysis, alpha factor  evaluation the alpha, data visualization, portfolio optimiazion results and visualisation, asset allocations, etc. 
- Nice plots out there!


### Unit Testing

In `unittest_optimization.py`, I  have developed a unit test procedure using the python `unittest`module to test the optimization engine that is being carried by the function `get_optimal_weights()`.






