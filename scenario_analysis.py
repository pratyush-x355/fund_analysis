# file: /Users/pratyusatripathy/my_work/fund_analysis/scenario_analysis.py

import numpy as np
import pandas as pd

def run_scenario(fund_data, factor_shocks, asset_shocks=None, factor_corr=None):
    """
    Run scenario analysis given factor shocks and asset-specific shocks.
    
    Parameters
    ----------
    fund_data : pd.DataFrame
        Must include columns: ["Fund name", "Return_post", "Std Dev", "Weight",
                               "Value", "Momentum", "Quality", "Risk", "Technical"]
    factor_shocks : dict
        Example: {"Momentum": 0.05, "Risk": -0.03}
    asset_shocks : dict
        Example: {"Equity": -0.20, "Gold": 0.10} (optional)
    factor_corr : pd.DataFrame or np.ndarray, optional
        Factor correlation matrix (if portfolio vol needs recalculation).
    
    Returns
    -------
    results : dict
        Scenario-adjusted fund returns and portfolio stats.
    """
    fund_data = fund_data.copy()
    
    # Base expected returns
    base_returns = fund_data["Return_post"].values.astype(float)
    
    # Factor exposures
    exposures = fund_data[["Value", "Momentum", "Quality", "Risk", "Technical"]].values.astype(float)
    factors = ["Value", "Momentum", "Quality", "Risk", "Technical"]
    
    # Build shock vector
    shock_vec = np.zeros(len(factors))
    for i, f in enumerate(factors):
        shock_vec[i] = factor_shocks.get(f, 0.0)
    
    # Calculate impact of factor shocks
    factor_impacts = exposures @ shock_vec
    
    # Apply asset-specific shocks if provided
    asset_impact = np.zeros(len(base_returns))
    if asset_shocks:
        for asset, shock in asset_shocks.items():
            mask = fund_data["Fund name"].str.contains(asset, case=False, regex=True)
            asset_impact[mask] += shock
    
    # New fund returns
    scenario_returns = base_returns + factor_impacts + asset_impact
    fund_data["Scenario_Return"] = scenario_returns
    
    # Portfolio return
    weights = fund_data["Weight"].values.astype(float)
    portfolio_return = np.dot(weights, scenario_returns)
    
    # Portfolio volatility (optional: factor-based)
    if factor_corr is not None:
        # Instrument covariance via factor model
        factor_vols = np.ones(len(factors))  # assume unit vol for shocks
        factor_cov = np.outer(factor_vols, factor_vols) * factor_corr
        inst_cov = exposures @ factor_cov @ exposures.T
        port_vol = np.sqrt(weights @ inst_cov @ weights)
    else:
        port_vol = np.dot(weights, fund_data["Std Dev"].values)
    
    return {
        "fund_results": fund_data[["Fund name", "Return_post", "Scenario_Return"]],
        "portfolio_return": portfolio_return,
        "portfolio_volatility": port_vol
    }



