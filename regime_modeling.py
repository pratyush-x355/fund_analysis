import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as ss
from hmmlearn import hmm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class HMMRegimeDetector:
    """
    A class for detecting market regimes using Hidden Markov Models.
    
    This class takes a time series dataset, applies HMM to detect different market regimes,
    and provides visualization and analysis capabilities.
    """
    
    def __init__(self, random_state=10):
        """
        Initialize the HMM Regime Detector.
        
        Parameters:
        -----------
        random_state : int, default=10
            Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.hidden_states = None
        self.vol_state_list = None
        self.hmm_df = None
        self.data = None
        self.original_data = None
        
    def _calculate_aic(self, model, X, lengths=None):
        """Calculate Akaike Information Criterion"""
        n_params = sum(model._get_n_fit_scalars_per_param().values())
        return -2 * model.score(X, lengths=lengths) + 2 * n_params
    
    def _calculate_bic(self, model, X, lengths=None):
        """Calculate Bayesian Information Criterion"""
        n_params = sum(model._get_n_fit_scalars_per_param().values())
        return -2 * model.score(X, lengths=lengths) + n_params * np.log(len(X))
    
    def _prepare_data(self, data, price_column, return_window=5):
        """
        Prepare the input data for HMM analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataframe with time series data
        price_column : str
            Name of the column containing price data
        return_window : int, default=5
            Rolling window for return calculation
            
        Returns:
        --------
        pandas.DataFrame : Prepared dataframe with returns
        """
        df = data.copy()
        
        # Ensure Date is the index
        if 'Date' in df.columns and df.index.name != 'Date':
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date' and not isinstance(df.index, pd.DatetimeIndex):
            # If no Date column exists, assume index is already datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
        
        # Calculate returns
        df['return'] = df[price_column].pct_change().dropna()
        df['returns'] = np.log(1 + df['return']) * 100
        df['rolling_return'] = df['returns'].rolling(return_window).sum()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def fit(self, data, price_column='Adj Close', n_states=None, max_states=8, 
            criterion='aic', return_window=5, n_iterations=10, plot_selection=True):
        """
        Fit the HMM model to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataframe with time series data
        price_column : str, default='Adj Close'
            Name of the column containing price data
        n_states : int, optional
            Fixed number of states. If None, will use model selection
        max_states : int, default=8
            Maximum number of states to consider for model selection
        criterion : str, default='aic'
            Criterion for model selection ('aic', 'bic', or 'll')
        return_window : int, default=5
            Rolling window for return calculation
        n_iterations : int, default=10
            Number of random initializations for each model
        plot_selection : bool, default=True
            Whether to plot the model selection criteria
            
        Returns:
        --------
        self : HMMRegimeDetector
            Returns self for method chaining
        """
        # Store original data
        self.original_data = data.copy()
        
        # Prepare data
        self.data = self._prepare_data(data, price_column, return_window)
        
        # Prepare observations
        observations = self.data[['rolling_return']].values
        
        if n_states is not None:
            # Use fixed number of states
            best_model = hmm.GaussianHMM(
                n_components=n_states, 
                covariance_type="full", 
                n_iter=200, 
                tol=1e-4, 
                random_state=self.random_state
            )
            best_model.fit(observations)
        else:
            # Model selection
            model_list = []
            aic_values = []
            bic_values = []
            ll_values = []
            
            states_range = range(2, max_states + 1)
            
            for n in states_range:
                best_ll = None
                best_model_for_n = None
                
                # Try multiple random initializations
                for i in range(n_iterations):
                    model = hmm.GaussianHMM(
                        n_components=n, 
                        covariance_type="full", 
                        n_iter=200, 
                        tol=1e-4, 
                        random_state=self.random_state + i
                    )
                    model.fit(observations)
                    score = model.score(observations)
                    
                    if not best_ll or best_ll < score:
                        best_ll = score
                        best_model_for_n = model
                
                # Store results
                aic_values.append(self._calculate_aic(best_model_for_n, observations))
                bic_values.append(self._calculate_bic(best_model_for_n, observations))
                ll_values.append(best_model_for_n.score(observations))
                model_list.append(best_model_for_n)
            
            # Plot model selection criteria
            if plot_selection:
                self._plot_model_selection(states_range, aic_values, bic_values, ll_values)
            
            # Select best model based on criterion
            if criterion == 'aic':
                best_model = model_list[aic_values.index(min(aic_values))]
            elif criterion == 'bic':
                best_model = model_list[bic_values.index(min(bic_values))]
            else:  # criterion == 'll'
                best_model = model_list[ll_values.index(max(ll_values))]
        
        # Store the model and predictions
        self.model = best_model
        self.hidden_states = best_model.predict(observations)
        
        # Create regime information dataframe
        temp_list = ss.rankdata(best_model.covars_.flatten()) - 1
        rank_list = (np.rint(temp_list)).astype(int)
        
        data_dict = {
            'Start Probabilities': best_model.startprob_.flatten(),
            'Means': best_model.means_.flatten(),
            'Variance': best_model.covars_.flatten(),
            'Volatility Rank': rank_list
        }
        
        self.hmm_df = pd.DataFrame(data_dict)
        
        # Create volatility state list
        self.vol_state_list = [
            int(self.hmm_df.iloc[state]['Volatility Rank']) 
            for state in self.hidden_states
        ]
        
        print("HMM Model Summary:")
        print("==================")
        print(f"Number of states: {best_model.n_components}")
        print(f"Log-likelihood: {best_model.score(observations):.2f}")
        print("\nRegime Characteristics:")
        print(self.hmm_df)
        
        return self
    
    def _plot_model_selection(self, states_range, aic_values, bic_values, ll_values):
        """Plot model selection criteria"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ln1 = ax.plot(states_range, aic_values, label="AIC", color="blue", marker="o")
        ln2 = ax.plot(states_range, bic_values, label="BIC", color="green", marker="o")
        
        ax2 = ax.twinx()
        ln3 = ax2.plot(states_range, ll_values, label="Log-Likelihood", color="orange", marker="o")
        
        ax.legend(handles=ax.lines + ax2.lines, loc='upper right')
        ax.set_title("Model Selection: AIC/BIC vs Log-Likelihood")
        ax.set_ylabel("Criterion Value (lower is better)")
        ax2.set_ylabel("Log-Likelihood (higher is better)")
        ax.set_xlabel("Number of HMM States")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regimes(self, start_date=None, end_date=None, series_column='returns', 
                     figsize=(20, 8), show_legend=True):
        """
        Plot the time series with regime identification.
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            Start date for plotting (if None, uses all data)
        end_date : str or datetime, optional
            End date for plotting (if None, uses all data)
        series_column : str, default='returns'
            Column to plot ('returns' or 'Adj Close')
        figsize : tuple, default=(20, 8)
            Figure size
        show_legend : bool, default=True
            Whether to show the legend
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        # Prepare data for plotting
        plot_data = self.data.reset_index().copy()
        
        if start_date or end_date:
            if start_date:
                plot_data = plot_data[plot_data['Date'] >= pd.to_datetime(start_date)]
            if end_date:
                plot_data = plot_data[plot_data['Date'] <= pd.to_datetime(end_date)]
            
            # Adjust hidden states accordingly
            start_idx = 0 if start_date is None else max(0, plot_data.index[0])
            end_idx = len(self.hidden_states) if end_date is None else min(len(self.hidden_states), plot_data.index[-1] + 1)
            plot_states = self.hidden_states[start_idx:end_idx]
        else:
            plot_states = self.hidden_states
        
        # Create color map
        states = pd.unique(plot_states)
        num_colors = len(states)
        cm = plt.get_cmap('viridis')
        color_list = [cm(1.0 * i / max(1, num_colors - 1)) for i in range(num_colors)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot vertical lines for each regime
        for i in range(len(plot_data)):
            if i < len(plot_states):
                regime_rank = int(self.hmm_df.iloc[plot_states[i]]['Volatility Rank'])
                ax.axvline(x=plot_data['Date'].iloc[i], linewidth=0.8, 
                          color=color_list[regime_rank], alpha=0.7)
        
        # Plot the main series
        if series_column in plot_data.columns:
            ax.plot(plot_data['Date'], plot_data[series_column], '-', 
                   color='black', alpha=0.8, linewidth=1.5, label=series_column)
        else:
            print(f"Warning: Column '{series_column}' not found. Available columns: {list(plot_data.columns)}")
        
        # Formatting
        ax.set_ylabel(series_column.title())
        ax.set_xlabel("Date")
        ax.set_title(f"Market Regimes - {series_column.title()}")
        ax.grid(True, alpha=0.3)
        
        # Add legend
        if show_legend:
            handles = []
            for i in range(len(states)):
                vol_rank = int(self.hmm_df.iloc[i]['Volatility Rank'])
                mean_val = self.hmm_df.iloc[i]['Means']
                var_val = self.hmm_df.iloc[i]['Variance']
                handles.append(mpatches.Patch(
                    color=color_list[vol_rank], 
                    label=f'Regime {i}: Vol Rank {vol_rank} (μ={mean_val:.2f}, σ²={var_val:.2f})'
                ))
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def get_data_with_regimes(self):
        """
        Get the original dataset with regime information attached.
        
        Returns:
        --------
        pandas.DataFrame : Dataset with regime columns added
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        # Create result dataframe
        result_df = self.data.copy()
        
        # Add regime information
        result_df['hidden_state'] = self.hidden_states
        result_df['volatility_rank'] = self.vol_state_list
        
        # Add regime characteristics
        regime_means = [self.hmm_df.iloc[state]['Means'] for state in self.hidden_states]
        regime_variances = [self.hmm_df.iloc[state]['Variance'] for state in self.hidden_states]
        
        result_df['regime_mean'] = regime_means
        result_df['regime_variance'] = regime_variances
        
        return result_df
    
    def get_regime_statistics(self):
        """
        Get comprehensive statistics about the detected regimes.
        
        Returns:
        --------
        dict : Dictionary containing regime statistics
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        data_with_regimes = self.get_data_with_regimes()
        
        stats = {}
        
        # Overall statistics
        stats['n_regimes'] = self.model.n_components
        stats['total_observations'] = len(self.hidden_states)
        
        # Regime-specific statistics
        stats['regime_details'] = []
        
        for i in range(self.model.n_components):
            regime_data = data_with_regimes[data_with_regimes['hidden_state'] == i]
            
            regime_stats = {
                'regime_id': i,
                'volatility_rank': int(self.hmm_df.iloc[i]['Volatility Rank']),
                'start_probability': self.hmm_df.iloc[i]['Start Probabilities'],
                'mean_return': self.hmm_df.iloc[i]['Means'],
                'variance': self.hmm_df.iloc[i]['Variance'],
                'n_observations': len(regime_data),
                'percentage_of_time': len(regime_data) / len(self.hidden_states) * 100,
                'avg_duration': self._calculate_avg_regime_duration(i),
                'actual_mean_return': regime_data['returns'].mean(),
                'actual_std_return': regime_data['returns'].std()
            }
            
            stats['regime_details'].append(regime_stats)
        
        # Transition matrix
        stats['transition_matrix'] = self.model.transmat_
        
        return stats
    
    def _calculate_avg_regime_duration(self, regime_id):
        """Calculate average duration of a specific regime"""
        durations = []
        current_duration = 0
        in_regime = False
        
        for state in self.hidden_states:
            if state == regime_id:
                if not in_regime:
                    in_regime = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_regime:
                    durations.append(current_duration)
                    in_regime = False
                    current_duration = 0
        
        # Handle case where last regime continues to the end
        if in_regime:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def predict_next_regime(self, current_regime=None):
        """
        Predict the most likely next regime based on transition probabilities.
        
        Parameters:
        -----------
        current_regime : int, optional
            Current regime state. If None, uses the last observed state.
            
        Returns:
        --------
        dict : Dictionary with next regime predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        if current_regime is None:
            current_regime = self.hidden_states[-1]
        
        # Get transition probabilities from current regime
        transition_probs = self.model.transmat_[current_regime]
        
        # Find most likely next regime
        next_regime = np.argmax(transition_probs)
        next_regime_prob = transition_probs[next_regime]
        
        return {
            'current_regime': current_regime,
            'most_likely_next_regime': next_regime,
            'probability': next_regime_prob,
            'all_transition_probabilities': dict(enumerate(transition_probs))
        }
