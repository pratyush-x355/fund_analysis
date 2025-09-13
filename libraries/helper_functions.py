import pandas as pd
import openpyxl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_excel_sheet_names(file_path):
    """
    Get all sheet names from an Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
        
    Returns:
    --------
    list : Sheet names in the Excel file
    """
    try:
        # Method 1: Using pandas (simpler)
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
    

def data_for_plotting(df, strategy_col='Strategy', beta_col='Beta', return_col='Return_post', 
                        default_size=300, colors=None):
    """
    Convert fund-level DataFrame to strategy-level averaged data for plotting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing fund data
    strategy_col : str
        Column name for strategy grouping
    beta_col : str  
        Column name for beta values
    return_col : str
        Column name for return values
    default_size : int
        Default bubble size for all points
    colors : list, optional
        List of colors to use. If None, uses default color scheme
        
    Returns:
    --------
    list : List of lists in format [beta, return, size, color, strategy_name]
    """
    
    if colors is None:
        colors = ['#4a90e2', '#7ed321', '#f5a623', '#bd10e0', '#b8e986', 
                 '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    
    # Calculate strategy-level averages
    strategy_averages = df.groupby(strategy_col).agg({
        beta_col: 'mean',
        return_col: 'mean'
    }).round(3)

    # Create the custom_data list
    strategy_data = []
    for i, (strategy, row) in enumerate(strategy_averages.iterrows()):
        strategy_data.append([
            row[beta_col], 
            row[return_col]*100,  # Convert to percentage
            default_size, 
            colors[i % len(colors)], 
            strategy
        ])
    
    # Calculate weighted overall averages
    weight_col = "Net_exp"
    total_weight = df[weight_col].sum()
    overall_beta = (df[beta_col] * df[weight_col]).sum() / total_weight
    overall_return = (df[return_col] * df[weight_col]).sum() / total_weight * 100
    
    # Add average summary point
    strategy_data.append([
        round(overall_beta, 3),
        round(overall_return, 3),
        default_size,
        '#000000',  # distinct color
        'Average'
    ])
    
    return strategy_data



def create_risk_return_plot(data_points=None, title="Risk-Return Analysis", 
                           xlabel="Beta to NIFTY", ylabel="Net Total Return",
                           figsize=(9, 6), xlim=None, ylim=None,
                           show_arrows=True, save_path=None, show_plot=True,
                           padding_factor=0.1):
    """
    Create a risk-return scatter plot with customizable data points and styling.
    
    Parameters:
    -----------
    data_points : list of lists, optional
        Each inner list should contain [beta, returns, size, color, label]
        If None, uses default fund data
    title : str
        Chart title
    xlabel : str  
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    xlim : tuple, optional
        X-axis limits (min, max). If None, auto-calculated from data
    ylim : tuple, optional
        Y-axis limits (min, max). If None, auto-calculated from data
    show_arrows : bool
        Whether to show arrows connecting numbered points
    save_path : str, optional
        Path to save the figure. If None, doesn't save
    show_plot : bool
        Whether to display the plot
    padding_factor : float
        Factor to add padding around data points (0.1 = 10% padding)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Default data if none provided
    if data_points is None:
        data_points = [
            [0.25, 7.2, 300, '#2c5f7a', '1'],  # Point 1
            [0.65, 10.5, 300, '#2c5f7a', '2'],  # Point 2
            [0.35, 10.0, 300, '#2c5f7a', '3'],  # Point 3
            [0.22, 12.0, 300, '#2c5f7a', '4'],  # Point 4
            [0.18, 14.5, 200, '#cccccc', 'Reference: typical US fund'],  # Reference
            [0.95, 14.5, 300, '#2d5f2d', 'PMS Long-Only Equity']  # PMS point
        ]
    
    # Extract beta and return values for auto-scaling
    betas = [point[0] for point in data_points]
    returns = [point[1] for point in data_points]
    
    # Auto-calculate limits if not provided
    if xlim is None:
        beta_min, beta_max = min(betas), max(betas)
        beta_range = beta_max - beta_min
        padding_x = beta_range * padding_factor
        xlim = (beta_min - padding_x, beta_max + padding_x)
    
    if ylim is None:
        return_min, return_max = min(returns), max(returns)
        return_range = return_max - return_min
        padding_y = return_range * padding_factor
        # Ensure y-axis starts at 0 or below for returns
        ylim = (max(0, return_min - padding_y), return_max + padding_y)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each point
    numbered_points = []  # Store numbered points for arrows
    
    for i, (beta, returns_val, size, color, label) in enumerate(data_points):
        # Check if this is a numbered point (contains only digits)
        is_numbered = label.isdigit()
        
        if is_numbered:
            # Numbered points - store for arrows and add text inside
            ax.scatter(beta, returns_val, s=size, c=color, alpha=0.8, zorder=3)
            ax.text(beta, returns_val, label, ha='center', va='center',
                   color='white', fontsize=12, fontweight='bold', zorder=4)
            numbered_points.append((beta, returns_val, int(label)))
        else:
            # Reference and other labeled points
            ax.scatter(beta, returns_val, s=size, c=color, alpha=0.8, zorder=3)
            
            # Calculate dynamic text offset based on axis ranges
            x_offset = (xlim[1] - xlim[0]) * 0.03  # 3% of x-range
            y_offset = (ylim[1] - ylim[0]) * 0.04  # 4% of y-range
            
            # Determine annotation style based on label content
            if 'Reference' in label or 'reference' in label.lower():
                ax.annotate(label, (beta, returns_val), 
                           xytext=(beta - x_offset, returns_val + y_offset),
                           fontsize=10, color='gray', ha='center')
            else:
                ax.annotate(label, (beta, returns_val), 
                           xytext=(beta - x_offset, returns_val + y_offset),
                           fontsize=10, color='green', ha='center', fontweight='bold')
    
    # Draw arrows between consecutive numbered points
    if show_arrows and len(numbered_points) > 1:
        # Sort numbered points by their number
        numbered_points.sort(key=lambda x: x[2])
        
        for i in range(len(numbered_points) - 1):
            x1, y1, _ = numbered_points[i]
            x2, y2, _ = numbered_points[i + 1]
            
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='#2c5f7a', 
                                     lw=2, alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color='gray')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='gray')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Auto-generate y-axis ticks and format as percentages
    y_range = ylim[1] - ylim[0]
    if y_range <= 5:
        y_step = 1
    elif y_range <= 20:
        y_step = 2
    elif y_range <= 50:
        y_step = 5
    else:
        y_step = 10
    
    y_start = int(ylim[0] // y_step) * y_step
    y_end = int(ylim[1] // y_step + 1) * y_step
    y_ticks = np.arange(y_start, y_end + y_step, y_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}%' for y in y_ticks])
    
    # Auto-generate x-axis ticks
    x_range = xlim[1] - xlim[0]
    if x_range <= 1:
        x_step = 0.2
    elif x_range <= 2:
        x_step = 0.4
    else:
        x_step = 0.5
    
    x_start = (int(xlim[0] // x_step) - 1) * x_step
    x_end = (int(xlim[1] // x_step) + 2) * x_step
    x_ticks = np.arange(x_start, x_end, x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x:.2f}' for x in x_ticks])
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax


def create_custom_risk_return_plot(funds_data, benchmark_data=None, 
                                  title="Custom Risk-Return Analysis"):
    """
    Simplified function for creating risk-return plots with fund data.
    
    Parameters:
    -----------
    funds_data : list of dicts
        Each dict should have keys: 'name', 'beta', 'return', 'color' (optional)
    benchmark_data : dict, optional
        Benchmark data with keys: 'name', 'beta', 'return'
    title : str
        Chart title
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    data_points = []
    
    # Add fund data points
    for i, fund in enumerate(funds_data):
        color = fund.get('color', '#2c5f7a')  # Default blue color
        size = fund.get('size', 300)
        
        # Use fund name or number as label
        label = fund.get('name', str(i+1))
        
        data_points.append([
            fund['beta'], 
            fund['return'], 
            size, 
            color, 
            label
        ])
    
    # Add benchmark if provided
    if benchmark_data:
        data_points.append([
            benchmark_data['beta'],
            benchmark_data['return'],
            200,
            '#cccccc',
            benchmark_data.get('name', 'Benchmark')
        ])
    
    return create_risk_return_plot(
        data_points=data_points,
        title=title,
        show_arrows=False  # Usually don't want arrows for custom data
    )
