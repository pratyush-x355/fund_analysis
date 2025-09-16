from libraries.regime_modeling import HMMRegimeDetector
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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


def calculate_var_covar_pandas(data):
    
    # Calculate covariance matrix
    covar_matrix = data.cov()
    
    # Extract variances (diagonal elements)
    variances = np.diag(covar_matrix)
    
    return covar_matrix, variances

# Visualization function
def plot_covariance_matrix(covar_matrix, instrument_names=None):
    """
    Plot heatmap of covariance matrix
    """
    plt.figure(figsize=(5, 4))
    
    if instrument_names is None:
        instrument_names = [f'Instrument_{i+1}' for i in range(covar_matrix.shape[0])]
    
    sns.heatmap(covar_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                xticklabels=instrument_names,
                yticklabels=instrument_names,
                fmt='.4f')
    
    plt.title('Variance-Covariance Matrix')
    plt.tight_layout()
    plt.show()



def plot_risk_attribution(df, total_variance, title="Portfolio Risk Attribution Dashboard"):
    """
    Create a risk attribution dashboard with multiple visualizations and insights.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['Factor', 'Absolute_Risk', 'Risk_Percentage']
    total_variance : float
        Total portfolio variance (same units as Absolute_Risk)
    title : str
        Dashboard title
    """
    
    print("Portfolio Risk Attribution Analysis")
    print("=" * 50)
    print(f"Total Portfolio Variance: {total_variance:.6f}")
    print("\nRisk Attribution Table:")
    print(df.to_string(index=False))

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.95)

    # 1. Pie Chart - Positive Contributors
    ax1 = plt.subplot(2, 3, 1)
    positive_data = df[df['Risk_Percentage'] > 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    wedges, texts, autotexts = ax1.pie(positive_data['Risk_Percentage'], 
                                      labels=positive_data['Factor'],
                                      autopct='%1.1f%%',
                                      colors=colors[:len(positive_data)],
                                      startangle=90,
                                      explode=[0.1 if x > 50 else 0.05 for x in positive_data['Risk_Percentage']])
    ax1.set_title('Risk Contribution Distribution\n(Positive Contributors Only)', fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # 2. Bar Chart - All Factors
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(df['Factor'], df['Risk_Percentage'], 
                   color=['red' if x < 0 else 'green' if x > 10 else 'orange' for x in df['Risk_Percentage']],
                   alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_title('Risk Attribution by Factor', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Risk Contribution (%)', fontsize=12)
    ax2.set_xlabel('Risk Factors', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    for bar, value in zip(bars, df['Risk_Percentage']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                 f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                 fontweight='bold')

    # 3. Horizontal Bar Chart - Absolute Risk
    ax3 = plt.subplot(2, 3, 3)
    y_pos = np.arange(len(df))
    bars_h = ax3.barh(y_pos, df['Absolute_Risk']*1000000,
                      color=colors[:len(df)], alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(df['Factor'])
    ax3.set_xlabel('Absolute Risk (×10⁻⁶)', fontsize=12)
    ax3.set_title('Absolute Risk Contribution', fontsize=14, fontweight='bold')
    for i, (bar, value) in enumerate(zip(bars_h, df['Absolute_Risk'])):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{value:.6f}', ha='left', va='center', fontweight='bold')

    # 4. Cumulative Risk Build-up
    ax4 = plt.subplot(2, 3, 4)
    cumulative = np.cumsum([0] + list(df['Risk_Percentage']))
    for i, (factor, pct) in enumerate(zip(df['Factor'], df['Risk_Percentage'])):
        color = 'red' if pct < 0 else 'green' if pct > 10 else 'orange'
        ax4.bar(i, pct, bottom=cumulative[i] if pct > 0 else cumulative[i] + pct, 
                color=color, alpha=0.7, edgecolor='black')
        ax4.text(i, cumulative[i] + pct/2, f'{pct:.1f}%', 
                 ha='center', va='center', fontweight='bold')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['Factor'], rotation=45, ha='right')
    ax4.set_title('Cumulative Risk Build-up', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cumulative Risk (%)', fontsize=12)

    # 5. Risk Metrics Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('tight')
    ax5.axis('off')
    summary_data = [
        ['Total Variance', f'{total_variance:.6f}'],
        ['Largest Contributor', f"{df.loc[df['Risk_Percentage'].idxmax(), 'Factor']} ({df['Risk_Percentage'].max():.1f}%)"],
        ['Systematic Risk*', f"{df[df['Factor'] != 'Residual']['Risk_Percentage'].sum():.1f}%"],
        ['Idiosyncratic Risk', f"{df[df['Factor'] == 'Residual']['Risk_Percentage'].iloc[0]:.1f}%"],
        ['Risk Concentration', 'High' if df['Risk_Percentage'].max() > 50 else 'Moderate'],
        ['Factor Diversification', 'Low' if df['Risk_Percentage'].max() > 70 else 'Good']
    ]
    table = ax5.table(cellText=summary_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for i in range(len(summary_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4ECDC4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
    ax5.set_title('Portfolio Risk Summary', fontsize=14, fontweight='bold', pad=20)

    # 6. Ranking by Risk Contribution
    ax6 = plt.subplot(2, 3, 6)
    sorted_df = df.sort_values('Risk_Percentage', ascending=True)
    bars_rank = ax6.barh(range(len(sorted_df)), sorted_df['Risk_Percentage'], 
                         color=['red' if x < 0 else 'lightcoral' if x < 5 else 'gold' if x < 15 else 'darkgreen' 
                                for x in sorted_df['Risk_Percentage']], alpha=0.8)
    ax6.set_yticks(range(len(sorted_df)))
    ax6.set_yticklabels(sorted_df['Factor'])
    ax6.set_xlabel('Risk Contribution (%)', fontsize=12)
    ax6.set_title('Factors Ranked by Risk Contribution', fontsize=14, fontweight='bold')
    for i, (bar, value) in enumerate(zip(bars_rank, sorted_df['Risk_Percentage'])):
        ax6.text(bar.get_width() + (1 if value > 0 else -1), bar.get_y() + bar.get_height()/2,
                 f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Additional Analysis
    print("\n" + "="*60)
    print("RISK ANALYSIS INSIGHTS")
    print("="*60)
    print(f"\n🔍 KEY FINDINGS:")
    print(f"• Residual risk dominates the portfolio ({df[df['Factor']=='Residual']['Risk_Percentage'].iloc[0]:.1f}%)")
    print(f"• Systematic risk factors contribute {df[df['Factor'] != 'Residual']['Risk_Percentage'].sum():.1f}% of total risk")
    print(f"• {df.loc[df['Risk_Percentage'].idxmax(), 'Factor']} is the largest systematic contributor "
          f"({df['Risk_Percentage'].max():.1f}%)")
    print(f"• Portfolio shows {'HIGH' if df['Risk_Percentage'].max() > 70 else 'MODERATE'} concentration risk")

    print(f"\n📊 RISK DECOMPOSITION:")
    for _, row in df.iterrows():
        risk_level = "🔴 HIGH" if row['Risk_Percentage'] > 50 else "🟡 MEDIUM" if row['Risk_Percentage'] > 10 else "🟢 LOW"
        print(f"• {row['Factor']:<12}: {row['Risk_Percentage']:>6.2f}% {risk_level}")

    print(f"\n💡 RECOMMENDATIONS:")
    if df[df['Factor']=='Residual']['Risk_Percentage'].iloc[0] > 60:
        print("• Consider increasing factor exposure to reduce idiosyncratic risk")
    if any((df['Factor']=='Value') & (df['Risk_Percentage'] > 10)):
        print("• Value factor exposure appears significant - monitor value cycle")
    if df['Risk_Percentage'].max() > 70:
        print("• High risk concentration detected - consider diversification")
    print(f"\nNote: *Systematic Risk = Sum of all factor risks excluding residual")
