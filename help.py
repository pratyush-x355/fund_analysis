from regime_modeling import HMMRegimeDetector
import pandas as pd
import numpy as np
from datetime import datetime


def demo_hmm_detector_monthly(monthly_data, price_column='Adj Close', date_column='Date'):
    """
    Simplified HMM regime detector for monthly data with fixed 4 regimes.
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        DataFrame containing monthly data with date and price columns
    price_column : str
        Name of the column containing price data (default: 'Adj Close')
    date_column : str
        Name of the column containing date data (default: 'Date')
    
    Returns:
    --------
    pd.DataFrame : DataFrame with original data plus regime assignments
    """
    
    print("=== HMM Regime Detector - Monthly Data ===")
    print(f"Data shape: {monthly_data.shape}")
    print(f"Date range: {monthly_data[date_column].min()} to {monthly_data[date_column].max()}")
    
    # Initialize detector with fixed 4 regimes
    detector = HMMRegimeDetector(random_state=42)
    
    # Fit the model with exactly 4 states
    print("\nFitting HMM model with 4 regimes...")
    """
    def fit(self, data, price_column='Adj Close', n_states=None, max_states=8, 
            criterion='aic', return_window=2, n_iterations=10, plot_selection=True):
    """
    detector.fit(
        data=monthly_data,
        price_column=price_column,
        n_states=4,  # Fixed at 4 regimes
        return_window=3,  # 5-month return window
        plot_selection=False  # No model selection plots needed
    )
    
    # Get data with regime assignments
    result_data = detector.get_data_with_regimes()
    result_data.reset_index(inplace=True)
    print(f"\nResults:")
    print(f"Total months: {len(result_data)}")
    print(f"Regimes detected: 4 (as specified)")
    
    # Display regime distribution
    regime_counts = result_data['hidden_state'].value_counts().sort_index()
    print(f"\nRegime distribution:")
    for regime, count in regime_counts.items():
        percentage = (count / len(result_data)) * 100
        print(f"Regime {regime}: {count} months ({percentage:.1f}%)")
    
    
    # Calculate regime statistics manually
    regime_stats = {}
    print(f"\nRegime Statistics Summary:")
    
    for regime in sorted(result_data['hidden_state'].unique()):
        regime_data = result_data[result_data['hidden_state'] == regime]['returns']
        avg_return = regime_data.mean()
        volatility = regime_data.std()
        
        regime_stats[regime] = {
            'average_return': avg_return,
            'volatility': volatility,
            'count': len(regime_data)
        }
        
        print(f"Regime {regime}: "
              f"Avg Return: {avg_return:.4f}, "
              f"Volatility: {volatility:.4f}, "
              f"Count: {len(regime_data)} months")
    
    # Get basic regime statistics from detector if available
    try:
        stats = detector.get_regime_statistics()
        print(f"\nDetailed Regime Statistics:")
        for regime in stats['regime_details']:
            print(f"Regime {regime['regime_id']}: "
                  f"Vol Rank {regime['volatility_rank']}, "
                  f"Time in regime: {regime['percentage_of_time']:.1f}%")
    except Exception as e:
        print(f"Note: Detailed detector statistics not available - {str(e)}")
    
    # Return just the essential columns and regime statistics
    essential_cols = [date_column, price_column, 'hidden_state']
    if 'returns' in result_data.columns:
        essential_cols.append('returns')
    
    final_result = result_data[essential_cols].copy()
    
    print(f"\nReturning DataFrame with regime assignments and regime statistics")
    return final_result, regime_stats


def calculate_monthly_performance_fees(data, annual_performance_fee_rate, annual_hurdle_rate):
    """
    Calculate monthly performance fees based on annual performance vs hurdle rate
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Original data containing monthly returns with columns:
        - gross_return: Monthly gross returns
        - Any other columns will be preserved
        Must have a date index or date column
    
    annual_performance_fee_rate : float
        Annual performance fee rate (e.g., 0.20 for 20%)
        
    annual_hurdle_rate : float
        Annual hurdle rate (e.g., 0.08 for 8%)
    
    Returns:
    --------
    pandas.DataFrame
        Updated data with new columns:
        - performance_fee: Monthly performance fee
        - annual_performance_fee: Annual performance fee (for reference)
        - annual_return: Annual gross return (for reference)
        - excess_return_annual: Annual excess return over hurdle (for reference)
    
    Logic:
    ------
    1. For each year, calculate annual return as sum of monthly gross returns
    2. If annual return > hurdle rate, calculate performance fee on excess return
    3. Distribute annual performance fee equally across 12 months
    4. If annual return <= hurdle rate, no performance fee
    """
    
    # Convert to DataFrame if dict
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Ensure we have a proper date index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            raise ValueError("Data must have a date index or 'date' column")
    
    # Validate required columns
    if 'gross_return' not in df.columns:
        raise ValueError("Data must contain 'gross_return' column")
    
    # Initialize new columns
    df['performance_fee'] = 0.0
    df['annual_performance_fee'] = 0.0
    df['annual_return'] = 0.0
    df['excess_return_annual'] = 0.0
    
    # Group by year and calculate performance fees
    print("Annual Performance Fee Calculation:")
    print("=" * 60)
    
    for year in df.index.year.unique():
        year_mask = df.index.year == year
        year_data = df[year_mask].copy()
        
        # Calculate annual return (sum of monthly returns)
        annual_gross_return = year_data['gross_return'].sum()
        
        # Calculate excess return over hurdle
        excess_return = annual_gross_return - annual_hurdle_rate
        
        # Store annual return for reference
        df.loc[year_mask, 'annual_return'] = annual_gross_return
        df.loc[year_mask, 'excess_return_annual'] = excess_return
        
        # Calculate performance fee if above hurdle
        if excess_return > 0:
            # Annual performance fee = excess return × fee rate
            annual_perf_fee = excess_return * annual_performance_fee_rate
            
            # Monthly performance fee = annual fee / 12
            n_months_in_year = len(year_data)
            monthly_perf_fee = annual_perf_fee / n_months_in_year
            
            # Apply to all months in the year
            df.loc[year_mask, 'performance_fee'] = monthly_perf_fee
            df.loc[year_mask, 'annual_performance_fee'] = annual_perf_fee
            
            print(f"Year {year}:")
            print(f"  Annual Gross Return:    {annual_gross_return:>8.2%}")
            print(f"  Hurdle Rate:            {annual_hurdle_rate:>8.2%}")
            print(f"  Excess Return:          {excess_return:>8.2%}")
            print(f"  Annual Performance Fee: {annual_perf_fee:>8.4%}")
            print(f"  Monthly Performance Fee:{monthly_perf_fee:>8.4%}")
            print(f"  Months in year:         {n_months_in_year:>8.0f}")
        else:
            print(f"Year {year}:")
            print(f"  Annual Gross Return:    {annual_gross_return:>8.2%}")
            print(f"  Hurdle Rate:            {annual_hurdle_rate:>8.2%}")
            print(f"  Below Hurdle - No Performance Fee")
        
        print("-" * 40)
    
    # Summary statistics
    total_perf_fees = df.groupby(df.index.year)['annual_performance_fee'].first()
    years_with_fees = (total_perf_fees > 0).sum()
    total_years = len(total_perf_fees)
    avg_annual_fee = total_perf_fees[total_perf_fees > 0].mean()
    
    print(f"\nSUMMARY:")
    print(f"Total Years:                    {total_years}")
    print(f"Years with Performance Fees:    {years_with_fees}")
    print(f"Years without Performance Fees: {total_years - years_with_fees}")
    if years_with_fees > 0:
        print(f"Average Annual Performance Fee: {avg_annual_fee:.4%}")
    
    return df