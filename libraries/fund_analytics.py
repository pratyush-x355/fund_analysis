import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalytics:
    """
    Comprehensive Portfolio Performance Analytics Class
    
    This class provides detailed analysis of portfolio performance including:
    - Gross and Net returns calculation
    - Risk metrics (Sharpe, Volatility, Drawdown, VaR, etc.)
    - Beta and tracking error analysis
    - Regime-wise analysis
    - Visualization capabilities
    """
    
    def __init__(self, data):
        """
        Initialize with monthly data
        
        Parameters:
        data (dict or DataFrame): Monthly data containing:
            - gross_return: Monthly gross returns
            - market_return: Monthly market returns
            - benchmark_return: Monthly benchmark returns
            - regime: Market regime for each month
            - management_fee: Monthly management fee
            - performance_fee: Monthly performance fee
            - cash_return: Monthly cash return (risk-free rate)
        """
        if isinstance(data, dict):
            self.df = pd.DataFrame(data)
        else:
            self.df = data.copy()
            
        self.validate_data()
        self.calculate_net_returns()
        self.results = {}
        
    def validate_data(self):
        """Validate input data"""
        required_columns = ['gross_return', 'market_return', 'benchmark_return', 
                          'regime', 'management_fee', 'performance_fee', 'cash_return']
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert to numeric
        numeric_cols = [col for col in required_columns if col != 'regime']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Check for missing values
        if self.df[numeric_cols].isnull().any().any():
            print("Warning: Missing values detected in numeric columns")
            
    def calculate_net_returns(self):
        """Calculate net returns after fees"""
        self.df['net_return'] = (self.df['gross_return'] - 
                                self.df['management_fee'] - 
                                self.df['performance_fee'])
        
    def calculate_basic_metrics(self):
        """Calculate basic return and risk metrics"""
        results = {}
        
        # Returns (annualized)
        results['gross_return_annual'] = (1 + self.df['gross_return'].mean())**12 - 1
        results['net_return_annual'] = (1 + self.df['net_return'].mean())**12 - 1
        results['benchmark_return_annual'] = (1 + self.df['benchmark_return'].mean())**12 - 1
        results['market_return_annual'] = (1 + self.df['market_return'].mean())**12 - 1
        results['cash_return_annual'] = (1 + self.df['cash_return'].mean())**12 - 1
        
        # Volatility (annualized)
        results['gross_volatility'] = self.df['gross_return'].std() * np.sqrt(12)
        results['net_volatility'] = self.df['net_return'].std() * np.sqrt(12)
        results['benchmark_volatility'] = self.df['benchmark_return'].std() * np.sqrt(12)
        
        # Sharpe Ratios
        excess_gross = self.df['gross_return'] - self.df['cash_return']
        excess_net = self.df['net_return'] - self.df['cash_return']
        
        results['sharpe_gross'] = (excess_gross.mean() / excess_gross.std()) * np.sqrt(12)
        results['sharpe_net'] = (excess_net.mean() / excess_net.std()) * np.sqrt(12)
        
        # Excess Returns
        results['excess_return_gross'] = results['gross_return_annual'] - results['cash_return_annual']
        results['excess_return_net'] = results['net_return_annual'] - results['cash_return_annual']
        results['excess_return_vs_benchmark_gross'] = results['gross_return_annual'] - results['benchmark_return_annual']
        results['excess_return_vs_benchmark_net'] = results['net_return_annual'] - results['benchmark_return_annual']
        
        # Beta
        market_excess = self.df['market_return'] - self.df['cash_return']
        gross_excess = self.df['gross_return'] - self.df['cash_return']
        net_excess = self.df['net_return'] - self.df['cash_return']
        
        results['beta_gross'] = np.cov(gross_excess, market_excess)[0,1] / np.var(market_excess)
        results['beta_net'] = np.cov(net_excess, market_excess)[0,1] / np.var(market_excess)
        
        # Tracking Error
        tracking_error_gross = self.df['gross_return'] - self.df['benchmark_return']
        tracking_error_net = self.df['net_return'] - self.df['benchmark_return']
        
        results['tracking_error_gross'] = tracking_error_gross.std() * np.sqrt(12)
        results['tracking_error_net'] = tracking_error_net.std() * np.sqrt(12)
        
        # Information Ratio
        results['information_ratio_gross'] = (tracking_error_gross.mean() / tracking_error_gross.std()) * np.sqrt(12)
        results['information_ratio_net'] = (tracking_error_net.mean() / tracking_error_net.std()) * np.sqrt(12)
        
        self.results.update(results)
        return results
    
    def calculate_drawdown(self):
        """Calculate drawdown metrics"""
        def calc_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown, cumulative
        
        # Gross drawdown
        dd_gross, cum_gross = calc_drawdown(self.df['gross_return'])
        self.df['drawdown_gross'] = dd_gross
        self.df['cumulative_gross'] = cum_gross
        
        # Net drawdown
        dd_net, cum_net = calc_drawdown(self.df['net_return'])
        self.df['drawdown_net'] = dd_net
        self.df['cumulative_net'] = cum_net
        
        # Benchmark drawdown
        dd_benchmark, cum_benchmark = calc_drawdown(self.df['benchmark_return'])
        self.df['drawdown_benchmark'] = dd_benchmark
        self.df['cumulative_benchmark'] = cum_benchmark
        
        # Drawdown statistics
        results = {
            'max_drawdown_gross': dd_gross.min(),
            'max_drawdown_net': dd_net.min(),
            'max_drawdown_benchmark': dd_benchmark.min(),
        }
        
        # Drawdown duration
        def calc_drawdown_duration(drawdown):
            is_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            
            for dd in is_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
                
            return drawdown_periods
        
        dd_duration_gross = calc_drawdown_duration(dd_gross)
        dd_duration_net = calc_drawdown_duration(dd_net)
        
        results['max_drawdown_duration_gross'] = max(dd_duration_gross) if dd_duration_gross else 0
        results['max_drawdown_duration_net'] = max(dd_duration_net) if dd_duration_net else 0
        results['avg_drawdown_duration_gross'] = np.mean(dd_duration_gross) if dd_duration_gross else 0
        results['avg_drawdown_duration_net'] = np.mean(dd_duration_net) if dd_duration_net else 0
        
        self.results.update(results)
        return results
    
    def calculate_var_es(self, confidence_level=0.05):
        """Calculate Value at Risk and Expected Shortfall"""
        results = {}
        
        # Monthly VaR and ES
        var_gross = np.percentile(self.df['gross_return'], confidence_level * 100)
        var_net = np.percentile(self.df['net_return'], confidence_level * 100)
        
        es_gross = self.df['gross_return'][self.df['gross_return'] <= var_gross].mean()
        es_net = self.df['net_return'][self.df['net_return'] <= var_net].mean()
        
        # Annualized VaR and ES (assuming normal distribution)
        results['var_gross_monthly'] = var_gross
        results['var_net_monthly'] = var_net
        results['var_gross_annual'] = var_gross * np.sqrt(12)
        results['var_net_annual'] = var_net * np.sqrt(12)
        
        results['expected_shortfall_gross_monthly'] = es_gross
        results['expected_shortfall_net_monthly'] = es_net
        results['expected_shortfall_gross_annual'] = es_gross * np.sqrt(12)
        results['expected_shortfall_net_annual'] = es_net * np.sqrt(12)
        
        self.results.update(results)
        return results
    
    def regime_analysis(self):
        """Analyze performance by regime"""
        regime_stats = {}
        
        for regime in self.df['regime'].unique():
            regime_data = self.df[self.df['regime'] == regime]
            
            stats_dict = {
                'gross_return': regime_data['gross_return'].mean() * 12,  # Annualized
                'net_return': regime_data['net_return'].mean() * 12,
                'benchmark_return': regime_data['benchmark_return'].mean() * 12,
                'excess_return_gross': (regime_data['gross_return'] - regime_data['benchmark_return']).mean() * 12,
                'excess_return_net': (regime_data['net_return'] - regime_data['benchmark_return']).mean() * 12,
                'volatility_gross': regime_data['gross_return'].std() * np.sqrt(12),
                'volatility_net': regime_data['net_return'].std() * np.sqrt(12),
                'sharpe_gross': ((regime_data['gross_return'] - regime_data['cash_return']).mean() / 
                               (regime_data['gross_return'] - regime_data['cash_return']).std()) * np.sqrt(12),
                'sharpe_net': ((regime_data['net_return'] - regime_data['cash_return']).mean() / 
                              (regime_data['net_return'] - regime_data['cash_return']).std()) * np.sqrt(12),
                'periods': len(regime_data)
            }
            
            regime_stats[regime] = stats_dict
        
        self.results['regime_analysis'] = regime_stats
        return regime_stats
    
    def create_plots(self):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16)
        
        # 1. Cumulative Returns
        axes[0,0].plot(self.df.index, self.df['cumulative_gross'], label='Gross Return', linewidth=2)
        axes[0,0].plot(self.df.index, self.df['cumulative_net'], label='Net Return', linewidth=2)
        axes[0,0].plot(self.df.index, self.df['cumulative_benchmark'], label='Benchmark', linewidth=2)
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[0,1].fill_between(self.df.index, self.df['drawdown_gross'], 0, alpha=0.3, label='Gross DD')
        axes[0,1].fill_between(self.df.index, self.df['drawdown_net'], 0, alpha=0.3, label='Net DD')
        axes[0,1].set_title('Drawdown')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Tracking Error
        tracking_error_gross = self.df['gross_return'] - self.df['benchmark_return']
        tracking_error_net = self.df['net_return'] - self.df['benchmark_return']
        
        axes[0,2].plot(self.df.index, tracking_error_gross.cumsum(), label='Gross TE', linewidth=2)
        axes[0,2].plot(self.df.index, tracking_error_net.cumsum(), label='Net TE', linewidth=2)
        axes[0,2].set_title('Cumulative Excess Return')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Rolling Information Ratio (12-month)
        window = min(12, len(self.df))
        if window > 1:
            rolling_ir_gross = (tracking_error_gross.rolling(window).mean() / 
                              tracking_error_gross.rolling(window).std()) * np.sqrt(12)
            rolling_ir_net = (tracking_error_net.rolling(window).mean() / 
                            tracking_error_net.rolling(window).std()) * np.sqrt(12)
            
            axes[1,0].plot(self.df.index, rolling_ir_gross, label='Gross IR', linewidth=2)
            axes[1,0].plot(self.df.index, rolling_ir_net, label='Net IR', linewidth=2)
            axes[1,0].set_title(f'{window}-Month Rolling Information Ratio')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 5. Return Distribution
        axes[1,1].hist(self.df['gross_return'], bins=20, alpha=0.7, label='Gross', density=True)
        axes[1,1].hist(self.df['net_return'], bins=20, alpha=0.7, label='Net', density=True)
        axes[1,1].hist(self.df['benchmark_return'], bins=20, alpha=0.7, label='Benchmark', density=True)
        axes[1,1].set_title('Return Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Regime Performance
        regime_data = self.results.get('regime_analysis', {})
        if regime_data:
            regimes = list(regime_data.keys())
            gross_returns = [regime_data[r]['gross_return'] for r in regimes]
            net_returns = [regime_data[r]['net_return'] for r in regimes]
            
            x = np.arange(len(regimes))
            width = 0.35
            
            axes[1,2].bar(x - width/2, gross_returns, width, label='Gross', alpha=0.7)
            axes[1,2].bar(x + width/2, net_returns, width, label='Net', alpha=0.7)
            axes[1,2].set_title('Annualized Returns by Regime')
            axes[1,2].set_xticks(x)
            axes[1,2].set_xticklabels(regimes)
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics()
        drawdown_metrics = self.calculate_drawdown()
        var_es_metrics = self.calculate_var_es()
        regime_stats = self.regime_analysis()
        
        print("="*80)
        print("PORTFOLIO PERFORMANCE ANALYTICS REPORT")
        print("="*80)
        
        # Basic Performance Metrics
        print("\n📊 BASIC PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Gross Annual Return:      {basic_metrics['gross_return_annual']:.2%}")
        print(f"Net Annual Return:        {basic_metrics['net_return_annual']:.2%}")
        print(f"Benchmark Return:         {basic_metrics['benchmark_return_annual']:.2%}")
        print(f"Cash Return:              {basic_metrics['cash_return_annual']:.2%}")
        
        print(f"\nGross Volatility:         {basic_metrics['gross_volatility']:.2%}")
        print(f"Net Volatility:           {basic_metrics['net_volatility']:.2%}")
        print(f"Benchmark Volatility:     {basic_metrics['benchmark_volatility']:.2%}")
        
        print(f"\nGross Sharpe Ratio:       {basic_metrics['sharpe_gross']:.3f}")
        print(f"Net Sharpe Ratio:         {basic_metrics['sharpe_net']:.3f}")
        
        # Risk Metrics
        print("\n⚠️  RISK METRICS")
        print("-" * 50)
        print(f"Max Drawdown (Gross):     {drawdown_metrics['max_drawdown_gross']:.2%}")
        print(f"Max Drawdown (Net):       {drawdown_metrics['max_drawdown_net']:.2%}")
        print(f"Max DD Duration (Gross):  {drawdown_metrics['max_drawdown_duration_gross']:.0f} months")
        print(f"Max DD Duration (Net):    {drawdown_metrics['max_drawdown_duration_net']:.0f} months")
        
        print(f"\nVaR (5%, Gross):          {var_es_metrics['var_gross_monthly']:.2%} monthly")
        print(f"VaR (5%, Net):            {var_es_metrics['var_net_monthly']:.2%} monthly")
        print(f"Expected Shortfall (Gross): {var_es_metrics['expected_shortfall_gross_monthly']:.2%} monthly")
        print(f"Expected Shortfall (Net):   {var_es_metrics['expected_shortfall_net_monthly']:.2%} monthly")
        
        # Beta and Tracking
        print("\n📈 BETA AND TRACKING METRICS")
        print("-" * 50)
        print(f"Beta (Gross):             {basic_metrics['beta_gross']:.3f}")
        print(f"Beta (Net):               {basic_metrics['beta_net']:.3f}")
        print(f"Tracking Error (Gross):   {basic_metrics['tracking_error_gross']:.2%}")
        print(f"Tracking Error (Net):     {basic_metrics['tracking_error_net']:.2%}")
        print(f"Information Ratio (Gross): {basic_metrics['information_ratio_gross']:.3f}")
        print(f"Information Ratio (Net):   {basic_metrics['information_ratio_net']:.3f}")
        
        # Excess Returns
        print("\n💰 EXCESS RETURNS")
        print("-" * 50)
        print(f"Excess Return vs Cash (Gross):      {basic_metrics['excess_return_gross']:.2%}")
        print(f"Excess Return vs Cash (Net):        {basic_metrics['excess_return_net']:.2%}")
        print(f"Excess Return vs Benchmark (Gross): {basic_metrics['excess_return_vs_benchmark_gross']:.2%}")
        print(f"Excess Return vs Benchmark (Net):   {basic_metrics['excess_return_vs_benchmark_net']:.2%}")
        
        # Regime Analysis
        print("\n🔄 REGIME ANALYSIS")
        print("-" * 50)
        regime_df = pd.DataFrame(regime_stats).T
        regime_df = regime_df.round(4)
        print(regime_df.to_string())
        
        return {
            'basic_metrics': basic_metrics,
            'drawdown_metrics': drawdown_metrics,
            'var_es_metrics': var_es_metrics,
            'regime_analysis': regime_stats,
            'data': self.df
        }
    
    def export_results(self, filename=None):
        """Export results to Excel"""
        if filename is None:
            filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary metrics
            summary_data = []
            for category, metrics in self.results.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        summary_data.append({'Category': category, 'Metric': metric, 'Value': value})
                else:
                    summary_data.append({'Category': 'General', 'Metric': category, 'Value': metrics})
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Raw data with calculated fields
            self.df.to_excel(writer, sheet_name='Data', index=True)
            
            # Regime analysis
            if 'regime_analysis' in self.results:
                regime_df = pd.DataFrame(self.results['regime_analysis']).T
                regime_df.to_excel(writer, sheet_name='Regime_Analysis', index=True)
        
        print(f"Results exported to {filename}")
        return filename

