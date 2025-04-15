#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Model Performance Monitoring System

This script provides comprehensive monitoring of all NBA prediction models,
tracking their performance over time and generating detailed reports and visualizations.

Features:
- Tracks each model's performance metrics over time
- Compares model performance across different time periods
- Generates performance dashboards with key metrics
- Alerts on model drift or performance degradation
- Automatically logs detailed model performance to the database
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.connection import init_db, get_connection_pool
from src.database.models import ModelPerformance, ModelWeights, SystemLog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_performance_history(time_window='all'):
    """Fetch performance history for all models from the database

    Args:
        time_window: Time period to fetch ('7d', '30d', '90d', 'all')

    Returns:
        DataFrame containing model performance history
    """
    try:
        # Initialize database connection
        if not init_db():
            logger.error("Failed to initialize database connection")
            return pd.DataFrame()
            
        # Convert time_window to datetime for filtering
        start_date = None
        if time_window != 'all':
            days = int(time_window.replace('d', ''))
            start_date = datetime.now() - timedelta(days=days)

        # Fetch all model performance records
        with get_connection_pool().getconn() as conn:
            query = """
            SELECT 
                model_name, version, prediction_target, time_window,
                metrics, recorded_at, num_predictions
            FROM model_performance
            """
            
            # Add date filtering if specified
            params = []
            if start_date:
                query += " WHERE recorded_at >= %s"
                params.append(start_date)
                
            query += " ORDER BY recorded_at DESC"
            
            # Execute query
            with conn.cursor() as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                    
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
            
            # Create DataFrame from results
            df = pd.DataFrame(results, columns=columns)
            
            # Parse metrics JSON
            if not df.empty and 'metrics' in df.columns:
                # Create individual columns for each metric
                for idx, row in df.iterrows():
                    metrics = json.loads(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
                    for key, value in metrics.items():
                        df.loc[idx, f'metric_{key}'] = value
                        
            return df
            
    except Exception as e:
        logger.error(f"Error fetching performance history: {str(e)}")
        return pd.DataFrame()


def calculate_performance_trends(performance_df):
    """Calculate performance trends over time for each model

    Args:
        performance_df: DataFrame with model performance history

    Returns:
        DataFrame with performance trends
    """
    if performance_df.empty:
        return pd.DataFrame()
        
    try:
        # Group by model and calculate metrics over time
        trends = []
        
        # Get list of all models
        models = performance_df['model_name'].unique()
        
        for model in models:
            model_data = performance_df[performance_df['model_name'] == model].copy()
            
            # Sort by recorded_at
            model_data.sort_values('recorded_at', inplace=True)
            
            # Get metric columns
            metric_cols = [col for col in model_data.columns if col.startswith('metric_')]
            
            if not metric_cols or len(model_data) < 2:
                continue
                
            # Calculate trends
            for metric in metric_cols:
                metric_name = metric.replace('metric_', '')
                
                # Calculate rolling metrics if enough data points
                if len(model_data) >= 3:
                    # Calculate moving average
                    model_data[f'{metric}_ma'] = model_data[metric].rolling(window=3, min_periods=1).mean()
                    
                    # Calculate trend (change from first to last)
                    first_value = model_data[metric].iloc[0]
                    last_value = model_data[metric].iloc[-1]
                    
                    if first_value != 0:  # Avoid division by zero
                        trend_pct = ((last_value - first_value) / first_value) * 100
                    else:
                        trend_pct = 0
                        
                    trends.append({
                        'model_name': model,
                        'metric': metric_name,
                        'first_value': first_value,
                        'last_value': last_value,
                        'change': last_value - first_value,
                        'change_pct': trend_pct,
                        'is_improving': trend_pct > 0 if metric_name in ['accuracy', 'auc', 'f1', 'r2'] 
                                       else trend_pct < 0
                    })
        
        return pd.DataFrame(trends)
        
    except Exception as e:
        logger.error(f"Error calculating performance trends: {str(e)}")
        return pd.DataFrame()


def visualize_performance(performance_df, trends_df, output_path=None):
    """Generate visualizations of model performance

    Args:
        performance_df: DataFrame with model performance history
        trends_df: DataFrame with performance trends
        output_path: Path to save visualizations (optional)

    Returns:
        None, saves visualizations to files if path provided
    """
    if performance_df.empty:
        logger.error("No performance data to visualize")
        return
        
    try:
        # Create output directory if specified
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Set the style
        sns.set(style="whitegrid")
        
        # 1. Plot accuracy/error metrics over time for each model
        models = performance_df['model_name'].unique()
        metric_cols = [col for col in performance_df.columns if col.startswith('metric_')]
        
        if metric_cols:
            for metric in metric_cols:
                metric_name = metric.replace('metric_', '')
                
                # Skip metrics that don't have numeric values
                if not pd.api.types.is_numeric_dtype(performance_df[metric]):
                    continue
                    
                plt.figure(figsize=(12, 6))
                
                for model in models:
                    model_data = performance_df[performance_df['model_name'] == model].copy()
                    model_data.sort_values('recorded_at', inplace=True)
                    
                    if not model_data.empty and metric in model_data.columns:
                        plt.plot(model_data['recorded_at'], model_data[metric], marker='o', label=model)
                
                plt.title(f'{metric_name.upper()} Over Time')
                plt.xlabel('Date')
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                if output_path:
                    plt.savefig(output_dir / f"{metric_name}_over_time_{timestamp}.png")
                    plt.close()
                else:
                    plt.show()
        
        # 2. Model comparison bar chart (for latest performance)
        latest_performance = performance_df.sort_values('recorded_at').drop_duplicates('model_name', keep='last')
        
        if not latest_performance.empty and metric_cols:
            key_metrics = ['metric_accuracy', 'metric_f1', 'metric_auc', 'metric_rmse', 'metric_mae']
            available_metrics = [m for m in key_metrics if m in latest_performance.columns]
            
            if available_metrics:
                for metric in available_metrics:
                    metric_name = metric.replace('metric_', '')
                    
                    plt.figure(figsize=(10, 6))
                    ax = sns.barplot(x='model_name', y=metric, data=latest_performance)
                    
                    # Add value labels
                    for i, v in enumerate(latest_performance[metric]):
                        if pd.notnull(v):  # Only add label if value is not null
                            ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
                    
                    plt.title(f'Latest {metric_name.upper()} by Model')
                    plt.xlabel('Model')
                    plt.ylabel(metric_name)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    if output_path:
                        plt.savefig(output_dir / f"latest_{metric_name}_comparison_{timestamp}.png")
                        plt.close()
                    else:
                        plt.show()
        
        # 3. Trend visualization
        if not trends_df.empty:
            # Plot change percentage by model and metric
            plt.figure(figsize=(12, 8))
            
            # Create a pivot table for better visualization
            pivot_df = trends_df.pivot_table(index='model_name', columns='metric', values='change_pct')
            
            # Plot heatmap
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
            plt.title('Performance Metric Change % by Model')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_dir / f"performance_trends_{timestamp}.png")
                plt.close()
            else:
                plt.show()
                
    except Exception as e:
        logger.error(f"Error visualizing performance: {str(e)}")


def generate_performance_report(performance_df, trends_df, output_path=None):
    """Generate a comprehensive performance report

    Args:
        performance_df: DataFrame with model performance history
        trends_df: DataFrame with performance trends
        output_path: Path to save report (optional)

    Returns:
        Report content as string
    """
    if performance_df.empty:
        return "No performance data available to generate report."
        
    try:
        report = []
        report.append("# NBA Prediction Model Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary Statistics
        report.append("## Summary Statistics")
        
        # Get latest performance for each model
        latest_performance = performance_df.sort_values('recorded_at').drop_duplicates('model_name', keep='last')
        
        report.append("### Latest Model Performance")
        report.append("|Model|Version|Prediction Target|Date|Metrics|")
        report.append("|-----|-----|-----|-----|-----|")
        
        for _, row in latest_performance.iterrows():
            # Extract key metrics to display
            metrics_display = {}
            metric_cols = [col for col in row.index if col.startswith('metric_')]
            
            for metric in metric_cols:
                metric_name = metric.replace('metric_', '')
                if pd.notnull(row[metric]) and isinstance(row[metric], (int, float)):
                    metrics_display[metric_name] = f"{row[metric]:.3f}"
            
            metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics_display.items()])
            
            report.append(f"|{row['model_name']}|{row['version']}|{row['prediction_target']}|" +
                         f"{row['recorded_at'].strftime('%Y-%m-%d')}|{metrics_str}|")
        
        # Performance Trends
        if not trends_df.empty:
            report.append("\n## Performance Trends")
            report.append("|Model|Metric|Change|Change %|Status|")
            report.append("|-----|-----|-----|-----|-----|")
            
            for _, row in trends_df.iterrows():
                # Determine status icon
                if row['is_improving']:
                    status = "⬆️ Improving"
                else:
                    status = "⬇️ Declining"
                    
                report.append(f"|{row['model_name']}|{row['metric']}|{row['change']:.3f}|" +
                             f"{row['change_pct']:.1f}%|{status}|")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        # Add automatic recommendations based on trends
        if not trends_df.empty:
            # Models with declining performance
            declining_models = trends_df[~trends_df['is_improving']]
            
            if not declining_models.empty:
                unique_declining = declining_models['model_name'].unique()
                
                if len(unique_declining) > 0:
                    report.append("\n### Models Requiring Attention")
                    
                    for model in unique_declining:
                        model_issues = declining_models[declining_models['model_name'] == model]
                        metrics = ", ".join(model_issues['metric'].values)
                        
                        report.append(f"* **{model}**: Performance declining in {metrics}")
                    
                    report.append("\n**Recommended Actions:**")
                    report.append("1. Retrain models with more recent data")
                    report.append("2. Review feature engineering process")
                    report.append("3. Consider hyperparameter optimization")
            
            # Best performing models
            top_models = []
            key_metric = None
            
            # Find a key metric that exists in the data
            for metric in ['accuracy', 'auc', 'f1']:
                if metric in trends_df['metric'].values:
                    key_metric = metric
                    break
            
            if key_metric:
                metric_data = latest_performance[[col for col in latest_performance.columns if key_metric in col]]
                if not metric_data.empty:
                    best_model = latest_performance.iloc[metric_data.values.argmax()]['model_name']
                    top_models.append(best_model)
                    
                    report.append(f"\n### Best Performing Model: {best_model}")
                    report.append("Consider using this model as the primary production model.")
        
        # Join all report sections
        full_report = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(output_dir / f"performance_report_{timestamp}.md", 'w') as f:
                f.write(full_report)
            
            logger.info(f"Saved performance report to {output_dir}/performance_report_{timestamp}.md")
        
        return full_report
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return f"Error generating report: {str(e)}"


def log_monitoring_activity(models_analyzed, trends_found=None):
    """Log monitoring activity to the database

    Args:
        models_analyzed: List of models that were analyzed
        trends_found: Optional dict of significant trends found

    Returns:
        bool: True if logging was successful
    """
    try:
        # Initialize database connection
        if not init_db():
            logger.error("Failed to initialize database connection")
            return False
            
        # Create log entry
        log_data = {
            "timestamp": datetime.now(),
            "activity": "model_monitoring",
            "models_analyzed": models_analyzed,
            "trends_found": trends_found or {}
        }
        
        # Convert to JSON for storage
        log_json = json.dumps(log_data)
        
        # Insert into system_logs table
        with get_connection_pool().getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO system_logs (log_type, timestamp, data)
                VALUES (%s, %s, %s)
                """, ('model_monitoring', datetime.now(), log_json))
                
            conn.commit()
            
        return True
        
    except Exception as e:
        logger.error(f"Error logging monitoring activity: {str(e)}")
        return False


def main():
    """Main function to run performance monitoring"""
    parser = argparse.ArgumentParser(description='Monitor NBA prediction model performance')
    parser.add_argument('--time-window', type=str, default='all', 
                        help='Time window to analyze (7d, 30d, 90d, all)')
    parser.add_argument('--output', type=str, help='Output directory for reports and visualizations')
    parser.add_argument('--report-only', action='store_true', help='Generate report only, no visualizations')
    args = parser.parse_args()

    # Initialize the database connection
    if not init_db():
        logger.error("Failed to initialize database connection")
        return

    # Fetch performance history
    logger.info(f"Fetching performance data for time window: {args.time_window}")
    performance_df = fetch_performance_history(args.time_window)
    
    if performance_df.empty:
        logger.error("No performance data found in database")
        return

    logger.info(f"Found performance data for {len(performance_df['model_name'].unique())} models")

    # Calculate performance trends
    trends_df = calculate_performance_trends(performance_df)

    # Generate visualizations
    if not args.report_only:
        logger.info("Generating performance visualizations")
        visualize_performance(performance_df, trends_df, args.output)

    # Generate performance report
    logger.info("Generating performance report")
    report = generate_performance_report(performance_df, trends_df, args.output)
    
    # Print report to console
    print("\n" + report)

    # Log monitoring activity
    models_analyzed = performance_df['model_name'].unique().tolist()
    log_monitoring_activity(models_analyzed)

    logger.info("Performance monitoring complete")


if __name__ == "__main__":
    main()
