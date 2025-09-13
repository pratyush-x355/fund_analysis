from regime_modeling import HMMRegimeDetector
import pandas as pd
import numpy as np

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

