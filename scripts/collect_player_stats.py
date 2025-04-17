#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Statistics Collection Script

This script collects NBA player statistics from BallDontLie API and
prepares them for use in training player prop prediction models.

The script saves player season averages and recent game statistics
in CSV format for use in the training pipeline.
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import ast

# Add project root to path to facilitate imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.balldontlie_client import BallDontLieClient

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"player_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def collect_active_players(client: BallDontLieClient, season: str) -> pd.DataFrame:
    """
    Collect a list of active NBA players
    
    Args:
        client: BallDontLie API client
        season: Season to collect players for (e.g., "2025")
        
    Returns:
        DataFrame with player information
    """
    logger.info(f"Collecting active players for season {season}")
    
    players_list = []
    page = 1
    per_page = 100
    total_pages = 1
    
    while page <= total_pages:
        try:
            response = client.get_players(page=page, per_page=per_page)
            
            if not response or 'data' not in response:
                logger.error(f"Invalid response from API on page {page}")
                break
                
            players = response['data']
            if not players:
                break
                
            # Extract meta information if available
            if 'meta' in response and 'total_pages' in response['meta']:
                total_pages = response['meta']['total_pages']
            
            # Add players to the list
            players_list.extend(players)
            
            logger.info(f"Collected {len(players)} players from page {page}/{total_pages}")
            page += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.6)
            
        except Exception as e:
            logger.error(f"Error fetching players on page {page}: {str(e)}")
            break
    
    if not players_list:
        logger.warning("No players collected")
        return pd.DataFrame()
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame(players_list)
        logger.info(f"Collected information for {len(df)} players")
        return df
    except Exception as e:
        logger.error(f"Error converting players to DataFrame: {str(e)}")
        return pd.DataFrame()


def collect_player_game_stats(client: BallDontLieClient, player_ids: List[int], days_back: int = 60) -> pd.DataFrame:
    """
    Collect player game statistics from recent games
    
    Args:
        client: BallDontLie API client
        player_ids: List of player IDs to collect stats for
        days_back: Number of days back to collect stats for
        
    Returns:
        DataFrame with player game statistics
    """
    logger.info(f"Collecting game statistics for {len(player_ids)} players from the last {days_back} days")
    
    # Calculate date range for recent games
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    all_stats = []
    
    # Process players in batches to optimize API calls
    for i, player_id in enumerate(player_ids):
        try:
            # Get player's game stats
            response = client.get_stats(
                player_id=player_id,
                start_date=start_str,
                end_date=end_str,
                per_page=100  # Get as many games as possible
            )
            
            if not response or 'data' not in response:
                logger.warning(f"No data returned for player {player_id}")
                continue
                
            player_stats = response['data']
            if not player_stats:
                logger.info(f"No game stats found for player {player_id} in the specified date range")
                continue
                
            all_stats.extend(player_stats)
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(player_ids):
                logger.info(f"Collected game stats for {i+1}/{len(player_ids)} players so far")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.6)
            
        except Exception as e:
            logger.error(f"Error fetching game stats for player {player_id}: {str(e)}")
    
    if not all_stats:
        logger.warning("No game statistics collected")
        return pd.DataFrame()
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame(all_stats)
        
        # Process and clean the data
        if 'player' in df.columns and 'team' in df.columns:
            # Extract player and team info from nested dictionaries
            for col in ['player', 'team', 'game']:
                if col in df.columns and df[col].dtype == 'object':
                    # Flatten the nested dictionary columns
                    nested_df = pd.json_normalize(df[col].dropna())
                    
                    # Rename columns to avoid conflicts
                    nested_df.columns = [f"{col}.{c}" for c in nested_df.columns]
                    
                    # Join with the main dataframe
                    df = df.drop(col, axis=1).reset_index(drop=True)
                    nested_df = nested_df.reset_index(drop=True)
                    
                    # Ensure the lengths match before joining
                    if len(df) == len(nested_df):
                        df = pd.concat([df, nested_df], axis=1)
        
        logger.info(f"Processed game statistics for {len(df)} player-game combinations")
        return df
    
    except Exception as e:
        logger.error(f"Error processing game statistics: {str(e)}")
        return pd.DataFrame()

def aggregate_player_statistics(game_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate game statistics into season averages for each player
    
    Args:
        game_stats_df: DataFrame with player game statistics
        
    Returns:
        DataFrame with aggregated season averages
    """
    logger.info("Aggregating game statistics into season averages")
    
    if game_stats_df.empty:
        logger.warning("No game statistics to aggregate")
        return pd.DataFrame()
    
    try:
        # Extract needed columns
        if 'player.id' not in game_stats_df.columns:
            logger.error("Required column 'player.id' not found in game statistics")
            return pd.DataFrame()
            
        # Group by player
        player_groups = game_stats_df.groupby('player.id')
        
        # Prepare data for aggregation
        aggregated_data = []
        
        for player_id, group in player_groups:
            # Get player info from the first game
            player_data = {
                'player_id': player_id,
                'player_name': group['player.first_name'].iloc[0] + ' ' + group['player.last_name'].iloc[0],
                'position': group['player.position'].iloc[0],
                'team_id': group['team.id'].iloc[0],
                'team_name': group['team.full_name'].iloc[0] if 'team.full_name' in group.columns else group['team.name'].iloc[0],
                'games_played': len(group)
            }
            
            # Calculate averages for statistical categories
            stat_columns = [
                'min', 'pts', 'reb', 'ast', 'stl', 'blk', 'turnover', 
                'pf', 'fg_pct', 'fg3_pct', 'ft_pct'
            ]
            
            # Ensure all required columns exist
            existing_columns = [col for col in stat_columns if col in group.columns]
            
            # Calculate averages
            for col in existing_columns:
                # Convert minutes format (e.g., "25:35") to float if needed
                if col == 'min' and isinstance(group[col].iloc[0], str):
                    # Extract minutes as float
                    player_data['minutes'] = group[col].apply(lambda x: 
                        float(x.split(':')[0]) + float(x.split(':')[1])/60 if ':' in str(x) else float(x)
                    ).mean()
                else:
                    # Rename some columns to more descriptive names
                    target_name = col
                    if col == 'pts': target_name = 'points'
                    elif col == 'reb': target_name = 'rebounds'
                    elif col == 'ast': target_name = 'assists'
                    elif col == 'stl': target_name = 'steals'
                    elif col == 'blk': target_name = 'blocks'
                    elif col == 'pf': target_name = 'fouls'
                    elif col == 'turnover': target_name = 'turnovers'
                    
                    # Calculate the mean of numeric columns
                    player_data[target_name] = group[col].mean()
            
            aggregated_data.append(player_data)
        
        # Create DataFrame from aggregated data
        if not aggregated_data:
            logger.warning("No data after aggregation process")
            return pd.DataFrame()
            
        agg_df = pd.DataFrame(aggregated_data)
        
        # Handle missing columns (fill with 0s)
        for col in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes']:
            if col not in agg_df.columns:
                agg_df[col] = 0.0
        
        # Round numeric values for readability
        numeric_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes', 
                        'fg_pct', 'fg3_pct', 'ft_pct']
        
        for col in numeric_cols:
            if col in agg_df.columns:
                agg_df[col] = agg_df[col].round(2)
        
        logger.info(f"Successfully aggregated statistics for {len(agg_df)} players")
        return agg_df
    
    except Exception as e:
        logger.error(f"Error aggregating player statistics: {str(e)}")
        return pd.DataFrame()


def collect_and_save_player_stats(season: str = None, days_back: int = 60, 
                                 output_dir: str = "data/player_stats") -> bool:
    """
    Collect all player statistics and save to CSV files
    
    Args:
        season: Season to collect stats for (e.g., "2024-2025")
        days_back: Number of days back to collect recent game stats
        output_dir: Directory to save output files
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine season if not provided
    if season is None:
        current_year = datetime.now().year
        # NBA seasons typically span two years (e.g., 2024-2025)
        # If we're in the latter half of the year, use current_year and next year
        # Otherwise, use previous year and current year
        if datetime.now().month >= 10:  # NBA season typically starts in October
            season = f"{current_year}-{current_year + 1}"
        else:
            season = f"{current_year - 1}-{current_year}"
    
    logger.info(f"Collecting player statistics for season {season}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize API client
    client = BallDontLieClient()
    
    try:
        # Collect active players
        players_df = collect_active_players(client, season.split('-')[0])
        if players_df.empty:
            logger.error("Failed to collect player information")
            return False
            
        # Save players data
        players_file = os.path.join(output_dir, "players.csv")
        players_df.to_csv(players_file, index=False)
        logger.info(f"Saved player information to {players_file}")
        
        # Extract player IDs
        player_ids = players_df['id'].tolist()
        
        # Collect player game statistics
        game_stats = collect_player_game_stats(client, player_ids, days_back)
        
        if game_stats.empty:
            logger.error("Failed to collect player game statistics")
            return False
        
        # Save raw game statistics
        game_stats_path = os.path.join(output_dir, "game_stats.csv")
        game_stats.to_csv(game_stats_path, index=False)
        logger.info(f"Saved raw game statistics to {game_stats_path}")
        
        # Aggregate game statistics into season averages
        season_averages = aggregate_player_statistics(game_stats)
        
        if season_averages.empty:
            logger.error("Failed to aggregate player statistics")
            return False
            
        # Save season averages
        season_averages_path = os.path.join(output_dir, "season_averages.csv")
        season_averages.to_csv(season_averages_path, index=False)
        logger.info(f"Saved season averages to {season_averages_path}")
        
        # Create simplified versions for each prop type
        try:
            # Points prediction features
            points_df = season_averages[[
                'player_id', 'games_played', 'points', 'minutes', 'fg_pct', 'ft_pct',
                'position', 'team_id', 'team_name'
            ]].copy()
            points_file = os.path.join(output_dir, "points_features.csv")
            points_df.to_csv(points_file, index=False)
            logger.info(f"Saved points prediction features to {points_file}")
            
            # Rebounds prediction features
            rebounds_df = season_averages[[
                'player_id', 'games_played', 'rebounds', 'minutes', 'position', 'team_id', 'team_name'
            ]].copy()
            rebounds_file = os.path.join(output_dir, "rebounds_features.csv")
            rebounds_df.to_csv(rebounds_file, index=False)
            logger.info(f"Saved rebounds prediction features to {rebounds_file}")
            
            # Assists prediction features
            assists_df = season_averages[[
                'player_id', 'games_played', 'assists', 'minutes', 'position', 'team_id', 'team_name'
            ]].copy()
            assists_file = os.path.join(output_dir, "assists_features.csv")
            assists_df.to_csv(assists_file, index=False)
            logger.info(f"Saved assists prediction features to {assists_file}")
            
        except Exception as e:
            logger.error(f"Error creating specialized feature files: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error collecting player statistics: {str(e)}")
        return False


def main():
    """
    Main function to collect player statistics
    """
    try:
        logger.info("Starting player statistics collection")
        
        # Get the current season
        current_season = "2024-2025"  # Hardcoded for now
        logger.info(f"Collecting player statistics for season {current_season}")
        
        # Initialize API client
        client = BallDontLieClient()
        
        # Create output directory if it doesn't exist
        data_dir = "data/player_stats"
        os.makedirs(data_dir, exist_ok=True)
        
        # Collect active players
        logger.info(f"Collecting active players for season {current_season.split('-')[0]}")
        
        active_players = collect_active_players(client, current_season.split('-')[0])
        
        if active_players.empty:
            logger.error("Failed to collect active players")
            return
            
        players_path = os.path.join(data_dir, "players.csv")
        active_players.to_csv(players_path, index=False)
        logger.info(f"Saved player information to {players_path}")
        
        # Extract player IDs
        player_ids = active_players['id'].tolist()
        
        # Collect player game statistics
        game_stats = collect_player_game_stats(client, player_ids)
        
        if game_stats.empty:
            logger.error("Failed to collect player game statistics")
            return
        
        # Save raw game statistics
        game_stats_path = os.path.join(data_dir, "game_stats.csv")
        game_stats.to_csv(game_stats_path, index=False)
        logger.info(f"Saved raw game statistics to {game_stats_path}")
        
        # Aggregate game statistics into season averages
        season_averages = aggregate_player_statistics(game_stats)
        
        if season_averages.empty:
            logger.error("Failed to aggregate player statistics")
            return
            
        # Save season averages
        season_averages_path = os.path.join(data_dir, "season_averages.csv")
        season_averages.to_csv(season_averages_path, index=False)
        logger.info(f"Saved season averages to {season_averages_path}")
        
        logger.info("Successfully collected and saved player statistics")
        
    except Exception as e:
        logger.error(f"Failed to collect and save player statistics: {str(e)}")


if __name__ == "__main__":
    main()
