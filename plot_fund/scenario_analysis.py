# file: /Users/pratyusatripathy/my_work/fund_analysis/scenario_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            print( asset, shock)
            mask = fund_data["Strategy"].str.contains(asset, case=False, regex=True)
            asset_impact[mask] = base_returns[mask] * shock

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


def plot_scenario_analysis(data, scenarios, beta_adjustments=None, scale_factor=200):
    """
    Plot Portfolio Return vs Beta across different scenarios.
    
    Parameters:
    - data: DataFrame with columns [Strategy, Return_post, Beta, Net_exp, Lever_ratio]
    - scenarios: dict {scenario_name: {strategy: weight}}
    - beta_adjustments: dict {scenario_name: {strategy: multiplier}} (optional)
    - scale_factor: float, multiplier for bubble size scaling
    """
    data = data.copy()
    data["Gross_exp"] = data["Net_exp"] * data["Lever_ratio"]

    results = []
    for scenario, strat_weights in scenarios.items():
        temp = data.copy()

        # Strategy-level weights equally distributed across instruments
        temp["Weight"] = temp["Strategy"].apply(lambda s: strat_weights[s] / (temp["Strategy"]==s).sum())

        # Adjust beta if given
        if beta_adjustments and scenario in beta_adjustments:
            temp["Adj_Beta"] = temp.apply(
                lambda row: row["Beta"] * beta_adjustments[scenario].get(row["Strategy"], 1.0), axis=1
            )
        else:
            temp["Adj_Beta"] = temp["Beta"]

        # Portfolio metrics
        port_return = (temp["Return_post"] * temp["Weight"]).sum()
        port_beta   = (temp["Adj_Beta"] * temp["Weight"]).sum()
        gross_exp   = (temp["Gross_exp"] * temp["Weight"]).sum()*10  # scale for visibility

        results.append({"Scenario": scenario, "Return": port_return, "Beta": port_beta, "Gross_exp": gross_exp})

    results_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(results_df["Beta"], results_df["Return"],
                s=results_df["Gross_exp"] * scale_factor,
                alpha=0.6, color="blue", edgecolors="k")

    # Annotate
    for i, row in results_df.iterrows():
        plt.text(row["Beta"]+0.01, row["Return"],
                 f'{row["Scenario"]}\nGrossExp={row["Gross_exp"]:.2f}', fontsize=9)

    plt.title("Portfolio Return vs Beta across Scenarios (Bubble = Gross Exposure)")
    plt.xlabel("Portfolio Beta")
    plt.ylabel("Portfolio Return")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    
    return results_df
