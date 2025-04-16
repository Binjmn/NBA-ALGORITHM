#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Feature Engineering for NBA Predictions

This module creates advanced features from raw NBA data for use in prediction models.
It calculates team statistics, player metrics, and time-sensitive performance indicators
using rolling windows and sophisticated aggregations.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for generating advanced features from NBA game data"""
    
    def __init__(self, data_dir: str = None, lookback_days: int = 30):
        """
        Initialize the feature engineer
        
        Args:
            data_dir: Directory containing historical data (will default to data/historical)
            lookback_days: Number of days to look back for historical data
        """
        # Set up paths
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else base_dir / 'data' / 'historical'
        self.output_dir = base_dir / 'data' / 'features'
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.teams = {}
        self.games = []
        self.game_stats = {}
        self.players = {}
        
        # Feature settings
        self.rolling_window_sizes = [5, 10, 20]  # Last N games for rolling stats
        self.lookback_days = lookback_days  # Days to look back for historical data
        
        logger.info(f"Feature Engineer initialized with data directory: {self.data_dir} and lookback_days: {self.lookback_days}")
    
    def load_data(self) -> bool:
        """
        Load all required data for feature engineering
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        logger.info("Loading data for feature engineering")
        
        # Load teams
        try:
            teams_file = self.data_dir / 'teams' / 'teams.json'
            if teams_file.exists():
                with open(teams_file, 'r') as f:
                    teams_data = json.load(f)
                    
                    # Handle different formats
                    if isinstance(teams_data, list):
                        self.teams = {str(team['id']): team for team in teams_data}
                    elif isinstance(teams_data, dict) and 'data' in teams_data:
                        self.teams = {str(team['id']): team for team in teams_data['data']}
                    
                    logger.info(f"Loaded {len(self.teams)} teams")
            else:
                logger.warning(f"Teams file not found at {teams_file}")
                return False
        except Exception as e:
            logger.error(f"Error loading teams: {str(e)}")
            return False
        
        # Load games
        try:
            games_files = list((self.data_dir / 'games').glob('*.json'))
            if not games_files:
                logger.warning("No game files found")
                return False
            
            for game_file in games_files:
                try:
                    with open(game_file, 'r') as f:
                        game_data = json.load(f)
                        
                        # Handle different formats
                        if isinstance(game_data, list):
                            self.games.extend(game_data)
                        elif isinstance(game_data, dict):
                            if 'data' in game_data and isinstance(game_data['data'], list):
                                self.games.extend(game_data['data'])
                            else:
                                self.games.append(game_data)
                except Exception as e:
                    logger.error(f"Error loading game file {game_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.games)} games")
        except Exception as e:
            logger.error(f"Error loading games: {str(e)}")
            return False
        
        # Load game stats
        try:
            stats_files = list((self.data_dir / 'stats').glob('game_stats_*.json'))
            if not stats_files:
                logger.warning("No game stats files found")
            
            for stats_file in stats_files:
                try:
                    # Extract game ID from filename
                    game_id = stats_file.stem.split('_')[-1].split('.')[0]
                    
                    with open(stats_file, 'r') as f:
                        stats_data = json.load(f)
                        self.game_stats[game_id] = stats_data
                except Exception as e:
                    logger.error(f"Error loading stats file {stats_file}: {str(e)}")
            
            logger.info(f"Loaded stats for {len(self.game_stats)} games")
        except Exception as e:
            logger.error(f"Error loading game stats: {str(e)}")
        
        return True
    
    def create_game_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from game data for easier processing
        
        Returns:
            pd.DataFrame: DataFrame with game data
        """
        logger.info("Creating game DataFrame")
        
        if not self.games:
            logger.error("No games loaded")
            return pd.DataFrame()
        
        # Extract required fields and normalize the data structure
        game_records = []
        
        for game in self.games:
            try:
                # Get basic game info
                game_id = str(game.get('id'))
                date = game.get('date', '')
                season = game.get('season', '')
                
                # Get team info - handle both direct properties and nested objects
                if 'home_team' in game and isinstance(game['home_team'], dict):
                    home_team_id = str(game['home_team'].get('id'))
                    home_team_abbreviation = game['home_team'].get('abbreviation')
                else:
                    home_team_id = str(game.get('home_team_id'))
                    home_team_abbreviation = self.teams.get(home_team_id, {}).get('abbreviation')
                
                if 'visitor_team' in game and isinstance(game['visitor_team'], dict):
                    away_team_id = str(game['visitor_team'].get('id'))
                    away_team_abbreviation = game['visitor_team'].get('abbreviation')
                else:
                    away_team_id = str(game.get('visitor_team_id'))
                    away_team_abbreviation = self.teams.get(away_team_id, {}).get('abbreviation')
                
                # Get scores
                home_score = game.get('home_team_score')
                away_score = game.get('visitor_team_score')
                
                # Calculate game result
                if home_score is not None and away_score is not None:
                    home_won = 1 if home_score > away_score else 0
                    away_won = 1 - home_won
                    point_diff = home_score - away_score
                else:
                    home_won = None
                    away_won = None
                    point_diff = None
                
                # Create record
                record = {
                    'game_id': game_id,
                    'date': date,
                    'season': season,
                    'home_team_id': home_team_id,
                    'home_team_abbreviation': home_team_abbreviation,
                    'away_team_id': away_team_id,
                    'away_team_abbreviation': away_team_abbreviation,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_won': home_won,
                    'away_won': away_won,
                    'point_diff': point_diff,
                    'status': game.get('status')
                }
                
                game_records.append(record)
            except Exception as e:
                logger.error(f"Error processing game {game.get('id', 'unknown')}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(game_records)
        
        # Sort by date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        logger.info(f"Created game DataFrame with {len(df)} rows")
        return df
    
    def calculate_team_stats(self, games_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate team-level statistics
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping team_id to team stats DataFrame
        """
        logger.info("Calculating team statistics")
        
        if games_df.empty:
            logger.error("No game data available for calculating team stats")
            return {}
        
        team_stats = {}
        
        # Get unique teams
        home_teams = set(games_df['home_team_id'].unique())
        away_teams = set(games_df['away_team_id'].unique())
        all_teams = home_teams.union(away_teams)
        
        # Calculate stats for each team
        for team_id in all_teams:
            # Get games where this team played (home or away)
            home_games = games_df[games_df['home_team_id'] == team_id].copy()
            away_games = games_df[games_df['away_team_id'] == team_id].copy()
            
            # Skip if no games found
            if home_games.empty and away_games.empty:
                continue
            
            # Rename columns for consistency when team is away
            away_games = away_games.rename(columns={
                'away_team_id': 'team_id', 
                'home_team_id': 'opponent_id',
                'away_score': 'team_score',
                'home_score': 'opponent_score',
                'away_won': 'team_won',
                'home_won': 'opponent_won'
            })
            
            # Rename columns for consistency when team is home
            home_games = home_games.rename(columns={
                'home_team_id': 'team_id', 
                'away_team_id': 'opponent_id',
                'home_score': 'team_score',
                'away_score': 'opponent_score',
                'home_won': 'team_won',
                'away_won': 'opponent_won'
            })
            
            # Combine home and away games
            team_games = pd.concat([home_games, away_games], ignore_index=True)
            team_games = team_games.sort_values('date')
            
            # Add cumulative stats
            team_games['games_played'] = range(1, len(team_games) + 1)
            team_games['cum_wins'] = team_games['team_won'].cumsum()
            team_games['cum_win_rate'] = team_games['cum_wins'] / team_games['games_played']
            team_games['cum_points_scored'] = team_games['team_score'].cumsum()
            team_games['cum_points_allowed'] = team_games['opponent_score'].cumsum()
            team_games['cum_point_diff'] = team_games['cum_points_scored'] - team_games['cum_points_allowed']
            team_games['cum_avg_points_scored'] = team_games['cum_points_scored'] / team_games['games_played']
            team_games['cum_avg_points_allowed'] = team_games['cum_points_allowed'] / team_games['games_played']
            
            # Add rolling window stats for different window sizes
            for window in self.rolling_window_sizes:
                # Skip if we don't have enough games
                if len(team_games) < window:
                    continue
                
                # Calculate rolling stats
                team_games[f'win_rate_{window}g'] = team_games['team_won'].rolling(window=window).mean()
                team_games[f'avg_points_{window}g'] = team_games['team_score'].rolling(window=window).mean()
                team_games[f'avg_points_allowed_{window}g'] = team_games['opponent_score'].rolling(window=window).mean()
                team_games[f'avg_point_diff_{window}g'] = (team_games['team_score'] - team_games['opponent_score']).rolling(window=window).mean()
                
                # Calculate streak features
                team_games[f'streak_{window}g'] = team_games['team_won'].rolling(window=window).sum()
            
            # Store team stats
            team_stats[team_id] = team_games
        
        logger.info(f"Calculated statistics for {len(team_stats)} teams")
        return team_stats
    
    def create_matchup_features(self, games_df: pd.DataFrame, team_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features for team matchups
        
        Args:
            games_df: DataFrame with game data
            team_stats: Dictionary with team stats DataFrames
            
        Returns:
            pd.DataFrame: DataFrame with matchup features
        """
        logger.info("Creating matchup features")
        
        if games_df.empty:
            logger.error("No game data available for creating matchup features")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = games_df.copy()
        
        # List for all matchup records
        matchup_records = []
        
        # Process each game
        for _, game in df.iterrows():
            try:
                game_id = game['game_id']
                date = game['date']
                home_team_id = game['home_team_id']
                away_team_id = game['away_team_id']
                
                # Skip if we don't have stats for either team
                if home_team_id not in team_stats or away_team_id not in team_stats:
                    continue
                
                # Get team stats before this game
                home_team_games = team_stats[home_team_id]
                away_team_games = team_stats[away_team_id]
                
                # Find the last game before this one for each team
                home_previous_games = home_team_games[home_team_games['date'] < date]
                away_previous_games = away_team_games[away_team_games['date'] < date]
                
                # Skip if either team hasn't played any games yet
                if home_previous_games.empty or away_previous_games.empty:
                    continue
                
                # Get the most recent stats for each team
                home_latest = home_previous_games.iloc[-1]
                away_latest = away_previous_games.iloc[-1]
                
                # Create matchup record with game info
                matchup = {
                    'game_id': game_id,
                    'date': date,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_score': game['home_score'],
                    'away_score': game['away_score'],
                    'home_won': game['home_won'],
                    
                    # Home team cumulative stats
                    'home_games_played': home_latest['games_played'],
                    'home_win_rate': home_latest['cum_win_rate'],
                    'home_avg_points': home_latest['cum_avg_points_scored'],
                    'home_avg_points_allowed': home_latest['cum_avg_points_allowed'],
                    
                    # Away team cumulative stats
                    'away_games_played': away_latest['games_played'],
                    'away_win_rate': away_latest['cum_win_rate'],
                    'away_avg_points': away_latest['cum_avg_points_scored'],
                    'away_avg_points_allowed': away_latest['cum_avg_points_allowed'],
                }
                
                # Add rolling window stats if available
                for window in self.rolling_window_sizes:
                    win_rate_col = f'win_rate_{window}g'
                    avg_points_col = f'avg_points_{window}g'
                    avg_points_allowed_col = f'avg_points_allowed_{window}g'
                    avg_point_diff_col = f'avg_point_diff_{window}g'
                    streak_col = f'streak_{window}g'
                    
                    # Add home team rolling stats if available
                    if win_rate_col in home_latest:
                        matchup[f'home_win_rate_{window}g'] = home_latest[win_rate_col]
                        matchup[f'home_avg_points_{window}g'] = home_latest[avg_points_col]
                        matchup[f'home_avg_points_allowed_{window}g'] = home_latest[avg_points_allowed_col]
                        matchup[f'home_avg_point_diff_{window}g'] = home_latest[avg_point_diff_col]
                        matchup[f'home_streak_{window}g'] = home_latest[streak_col]
                    
                    # Add away team rolling stats if available
                    if win_rate_col in away_latest:
                        matchup[f'away_win_rate_{window}g'] = away_latest[win_rate_col]
                        matchup[f'away_avg_points_{window}g'] = away_latest[avg_points_col]
                        matchup[f'away_avg_points_allowed_{window}g'] = away_latest[avg_points_allowed_col]
                        matchup[f'away_avg_point_diff_{window}g'] = away_latest[avg_point_diff_col]
                        matchup[f'away_streak_{window}g'] = away_latest[streak_col]
                
                # Calculate matchup-specific features
                matchup['win_rate_diff'] = matchup['home_win_rate'] - matchup['away_win_rate']
                matchup['avg_points_diff'] = matchup['home_avg_points'] - matchup['away_avg_points']
                matchup['defensive_rating_diff'] = matchup['away_avg_points_allowed'] - matchup['home_avg_points_allowed']
                
                # Add to records
                matchup_records.append(matchup)
            except Exception as e:
                logger.error(f"Error creating matchup features for game {game.get('game_id', 'unknown')}: {str(e)}")
        
        # Create DataFrame from matchup records
        matchup_df = pd.DataFrame(matchup_records)
        
        # Sort by date
        if 'date' in matchup_df.columns:
            matchup_df = matchup_df.sort_values('date')
        
        logger.info(f"Created matchup features for {len(matchup_df)} games")
        return matchup_df
    
    def add_game_stats_features(self, matchup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game stats features to matchup DataFrame
        
        Args:
            matchup_df: DataFrame with matchup features
            
        Returns:
            pd.DataFrame: DataFrame with added game stats features
        """
        logger.info("Adding game stats features")
        
        if matchup_df.empty or not self.game_stats:
            logger.warning("No matchup data or game stats available")
            return matchup_df
        
        # Create a copy to avoid modifying the original
        df = matchup_df.copy()
        
        # Add stats-based features if available
        game_ids = df['game_id'].unique()
        stats_added = 0
        
        for game_id in game_ids:
            if game_id not in self.game_stats:
                continue
            
            # Get game stats
            stats = self.game_stats[game_id]
            
            # Process stats and add team-level aggregates
            try:
                # Extract player stats
                if isinstance(stats, dict) and 'data' in stats:
                    player_stats = stats['data']
                elif isinstance(stats, list):
                    player_stats = stats
                else:
                    player_stats = []
                
                if not player_stats:
                    continue
                
                # Get the game row(s)
                game_rows = df[df['game_id'] == game_id].index
                if game_rows.empty:
                    continue
                
                # Get the team IDs for this game
                home_team_id = df.loc[game_rows[0], 'home_team_id']
                away_team_id = df.loc[game_rows[0], 'away_team_id']
                
                # Aggregate player stats by team
                home_team_stats = {'ast': 0, 'blk': 0, 'dreb': 0, 'oreb': 0, 'reb': 0, 'stl': 0, 'turnover': 0, 'pf': 0}
                away_team_stats = {'ast': 0, 'blk': 0, 'dreb': 0, 'oreb': 0, 'reb': 0, 'stl': 0, 'turnover': 0, 'pf': 0}
                
                for player in player_stats:
                    # Skip if no team info
                    if 'team' not in player or 'id' not in player['team']:
                        continue
                    
                    # Determine which team this player belongs to
                    player_team_id = str(player['team']['id'])
                    if player_team_id == home_team_id:
                        team_stats = home_team_stats
                    elif player_team_id == away_team_id:
                        team_stats = away_team_stats
                    else:
                        continue
                    
                    # Aggregate stats
                    for stat in team_stats:
                        if stat in player:
                            team_stats[stat] += player.get(stat, 0) or 0
                
                # Add features to the DataFrame
                for stat, value in home_team_stats.items():
                    df.loc[game_rows, f'home_{stat}'] = value
                
                for stat, value in away_team_stats.items():
                    df.loc[game_rows, f'away_{stat}'] = value
                
                # Calculate stat differences
                for stat in home_team_stats:
                    df.loc[game_rows, f'{stat}_diff'] = home_team_stats[stat] - away_team_stats[stat]
                
                stats_added += 1
                
            except Exception as e:
                logger.error(f"Error processing game stats for game {game_id}: {str(e)}")
        
        logger.info(f"Added game stats features for {stats_added} games")
        return df
    
    def engineer_features(self, game_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main method to engineer features
        
        Args:
            game_df: DataFrame with game data (optional). If provided, will use this instead of loading data.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logger.info("Starting feature engineering process")
        
        # Either use provided game data or load it
        games_df = None
        if game_df is not None and not game_df.empty:
            logger.info(f"Using provided game data with {len(game_df)} records")
            games_df = game_df.copy()
        else:
            # Load the data
            if not self.load_data():
                logger.error("Failed to load data for feature engineering")
                return pd.DataFrame()
                
            # Create game DataFrame
            games_df = self.create_game_dataframe()
            
        if games_df.empty:
            logger.error("Failed to create game DataFrame")
            return pd.DataFrame()
        
        # Calculate team statistics
        team_stats = self.calculate_team_stats(games_df)
        if not team_stats:
            logger.error("Failed to calculate team statistics")
            return pd.DataFrame()
        
        # Create matchup features
        matchup_df = self.create_matchup_features(games_df, team_stats)
        if matchup_df.empty:
            logger.error("Failed to create matchup features")
            return pd.DataFrame()
        
        # Add game stats features
        features_df = self.add_game_stats_features(matchup_df)
        
        # Save engineered features
        try:
            features_file = self.output_dir / 'engineered_features.csv'
            features_df.to_csv(features_file, index=False)
            logger.info(f"Saved engineered features to {features_file}")
        except Exception as e:
            logger.error(f"Error saving engineered features: {str(e)}")
        
        logger.info("Feature engineering process completed")
        return features_df


# Main function to run feature engineering
def run_feature_engineering(data_dir: str = None, lookback_days: int = 30) -> pd.DataFrame:
    """
    Run the feature engineering process
    
    Args:
        data_dir: Directory containing historical data
        lookback_days: Number of days to look back for historical data
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    engineer = FeatureEngineer(data_dir, lookback_days)
    return engineer.engineer_features()


# Run feature engineering if script is executed directly
if __name__ == "__main__":
    # Get data directory from command line argument if provided
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run feature engineering
    features_df = run_feature_engineering(data_dir)
    
    # Print summary
    if not features_df.empty:
        print(f"\nFeature Engineering Summary:")
        print(f"Number of games processed: {len(features_df)}")
        print(f"Number of features created: {len(features_df.columns)}")
        print(f"Sample features: {list(features_df.columns)[:10]}")
        print(f"\nFeatures saved to: data/features/engineered_features.csv")
