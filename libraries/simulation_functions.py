import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_instrument_correlations(factor_correlations, factor_exposures, return_dataframe=True, round_digits=3):
    """
    Calculate instrument correlations from factor correlations and exposures.
    
    Parameters:
    -----------
    factor_correlations : pandas.DataFrame or numpy.ndarray
        Factor correlation matrix (n_factors x n_factors)
    factor_exposures : pandas.DataFrame or numpy.ndarray
        Factor exposures matrix (n_factors x n_instruments)
        Rows = factors, Columns = instruments
    return_dataframe : bool, default True
        If True, returns pandas DataFrame with labels
        If False, returns numpy array
    round_digits : int, default 3
        Number of decimal places to round to
        
    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        Correlation matrix between instruments
    """
    
    # Convert inputs to numpy arrays
    factor_corr = np.array(factor_correlations)
    
    if isinstance(factor_exposures, pd.DataFrame):
        exposures = factor_exposures.values
        instrument_names = factor_exposures.columns
        factor_names = factor_exposures.index
    else:
        exposures = np.array(factor_exposures)
        instrument_names = None
        factor_names = None
    
    # Validate dimensions
    n_factors_corr = factor_corr.shape[0]
    n_factors_exp = exposures.shape[0]
    
    if n_factors_corr != n_factors_exp:
        raise ValueError(f"Factor correlation matrix has {n_factors_corr} factors, "
                        f"but exposures matrix has {n_factors_exp} factors")
    
    if factor_corr.shape[0] != factor_corr.shape[1]:
        raise ValueError("Factor correlation matrix must be square")
    
    # Calculate instrument covariance matrix
    # Cov = B.T @ Factor_Cov @ B
    B = exposures  # n_factors x n_instruments
    cov_matrix = B.T @ factor_corr @ B  # n_instruments x n_instruments
    
    # Convert covariance to correlation
    std_dev = np.sqrt(np.diag(cov_matrix))
    
    # Handle zero standard deviation (avoid division by zero)
    std_dev = np.where(std_dev == 0, 1, std_dev)
    
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    
    # Round the results
    corr_matrix = np.round(corr_matrix, round_digits)
    
    # Return as DataFrame with labels if requested
    if return_dataframe and instrument_names is not None:
        return pd.DataFrame(corr_matrix, 
                          index=instrument_names, 
                          columns=instrument_names)
    else:
        return corr_matrix
    



def nearest_psd(matrix, epsilon=1e-8):
    """Force a covariance matrix to be positive semi-definite."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def simulate_portfolio(mu, sigma, corr_matrix, weights, n_scenarios=1000, n_periods=12):
    """
    Monte Carlo simulation of portfolio returns and volatilities.
    
    Parameters
    ----------
    mu : array
        Expected annual returns of instruments.
    sigma : array
        Annualized standard deviations of instruments.
    corr_matrix : 2D array
        Correlation matrix of instruments.
    weights : array
        Portfolio weights (must sum to 1).
    n_scenarios : int
        Number of simulation scenarios.
    n_periods : int
        Number of time periods per scenario (e.g. 12 for months).
    """
    # Convert annual stats to period stats
    mu_period = mu / n_periods
    sigma_period = sigma / np.sqrt(n_periods)
    
    # Covariance matrix per period
    cov_period = np.outer(sigma_period, sigma_period) * corr_matrix
    cov_period = nearest_psd(cov_period)  # ensure PSD
    
    sim_results = []
    
    for _ in range(n_scenarios):
        # Simulate path of returns for n_periods
        sim_path = np.random.multivariate_normal(mu_period, cov_period, size=n_periods)
        port_path = sim_path @ weights
        
        # Annualized return (compound)
        port_return = (np.prod(1 + port_path) - 1)
        
        # Annualized volatility (sample std * sqrt(periods))
        port_vol = port_path.std(ddof=1) * np.sqrt(n_periods)
        
        sim_results.append([port_return, port_vol])
    
    results = pd.DataFrame(sim_results, columns=["Simulated_Return", "Simulated_StdDev"])
    return results


def plot_simulated_portfolio(results):
    # Visualization
    plt.figure(figsize=(14,6))

    # Histogram of returns
    plt.subplot(1, 2, 1)
    sns.histplot(results["Simulated_Return"], bins=15, kde=True, color="skyblue")
    plt.axvline(results["Simulated_Return"].mean(), color="red", linestyle="--", label="Mean")
    plt.title("Distribution of Simulated Portfolio Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()

    # Scatter: return vs std dev
    plt.subplot(1, 2, 2)
    plt.scatter(results["Simulated_StdDev"], results["Simulated_Return"], alpha=0.7, color="darkblue")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Return")
    plt.title("Risk-Return Across Simulations")

    plt.tight_layout()
    plt.show()

    print("Average Portfolio Return:", results['Simulated_Return'].mean())
    print("Average Portfolio StdDev:", results['Simulated_StdDev'].mean())

    return results
