#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Feature Engineering Module

This module transforms raw NBA data into engineered features for model training.
It creates comprehensive feature sets for NBA game prediction, including:
- Team performance metrics (recent form, scoring trends, etc.)
- Head-to-head history between teams
- Player availability and performance
- Advanced metrics (pace, efficiency, etc.)
- Odds and market indicators
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path('data/historical')
FEATURE_DIR = Path('data/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def serialize_dataframe(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for JSON serialization by converting numpy types to Python types
    
    Args:
        data: Dictionary containing pandas/numpy data
        
    Returns:
        Dictionary with serializable values
    """
    serializable_data = {}
    
    for key, value in data.items():
        if isinstance(value, np.integer):
            serializable_data[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_data[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif hasattr(value, 'to_dict'):
            # Handle pandas objects
            serializable_data[key] = value.to_dict()
        else:
            # Try standard serialization
            serializable_data[key] = value
            
    return serializable_data


class NBAFeatureEngineer:
    """Engineer features from NBA data for prediction models"""
    
    def __init__(self):
        """Initialize the feature engineer"""
        # Load team metadata
        self.teams_file = DATA_DIR / 'teams' / 'teams.json'
        self.teams = {}
        if self.teams_file.exists():
            with self.teams_file.open('r') as f:
                teams_data = json.load(f)
                for team in teams_data:
                    self.teams[str(team['id'])] = team
        
        # Create subdirectories for different feature types
        self.team_features_dir = FEATURE_DIR / 'team'
        self.game_features_dir = FEATURE_DIR / 'game'
        self.player_features_dir = FEATURE_DIR / 'player'
        
        self.team_features_dir.mkdir(exist_ok=True)
        self.game_features_dir.mkdir(exist_ok=True)
        self.player_features_dir.mkdir(exist_ok=True)
        
        # Cache for computed features
        self.cached_team_features = {}
        self.cached_game_features = {}
    
    def load_games(self) -> pd.DataFrame:
        """Load all collected games into a DataFrame"""
        logger.info("Loading games data")
        
        games_data = []
        game_files = list(DATA_DIR.glob('games/game_*.json'))
        
        if not game_files:
            logger.warning("No game files found in data directory")
            return pd.DataFrame()
        
        for game_file in game_files:
            try:
                with game_file.open('r') as f:
                    game = json.load(f)
                    games_data.append(game)
            except Exception as e:
                logger.error(f"Error loading game file {game_file}: {str(e)}")
        
        if not games_data:
            logger.warning("No games data loaded")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(games_data)
        
        # Add additional columns for easier processing
        df['game_date'] = pd.to_datetime(df['date'])
        df['season'] = df['season']
        df['home_team_id'] = df['home_team'].apply(lambda x: str(x['id']) if isinstance(x, dict) and 'id' in x else None)
        df['away_team_id'] = df['visitor_team'].apply(lambda x: str(x['id']) if isinstance(x, dict) and 'id' in x else None)
        df['home_team_name'] = df['home_team'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else None)
        df['away_team_name'] = df['visitor_team'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else None)
        
        # Handle scores
        df['home_score'] = df['home_team_score']
        df['away_score'] = df['visitor_team_score']
        
        # Create target variables
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['spread'] = df['home_score'] - df['away_score']
        df['total'] = df['home_score'] + df['away_score']
        
        # Sort by date
        df = df.sort_values('game_date')
        
        logger.info(f"Loaded {len(df)} games into DataFrame")
        return df
    
    def load_odds_data(self) -> Dict[str, Any]:
        """Load all odds data"""
        logger.info("Loading odds data")
        
        odds_data = {}
        odds_files = list(DATA_DIR.glob('odds/odds_*.json'))
        
        for odds_file in odds_files:
            try:
                date_str = odds_file.stem.replace('odds_', '')
                with odds_file.open('r') as f:
                    daily_odds = json.load(f)
                    for game_odds in daily_odds:
                        game_id = game_odds.get('id')
                        if game_id:
                            odds_data[str(game_id)] = game_odds
            except Exception as e:
                logger.error(f"Error loading odds file {odds_file}: {str(e)}")
        
        logger.info(f"Loaded odds data for {len(odds_data)} games")
        return odds_data
    
    def calculate_team_form(self, df: pd.DataFrame, team_id: str, date: pd.Timestamp,
                           window: int = 10) -> Dict[str, float]:
        """Calculate team form metrics based on recent games"""
        # Get games before the target date where team participated
        team_games = df[(df['game_date'] < date) & 
                       ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id))]
        
        # Sort by date (most recent first) and take the last 'window' games
        team_games = team_games.sort_values('game_date', ascending=False).head(window)
        
        if team_games.empty:
            return {
                'recent_win_pct': 0.5,  # Default to 50% if no data
                'avg_points_scored': 0,
                'avg_points_allowed': 0,
                'win_streak': 0,
                'games_played': 0
            }
        
        # Initialize metrics
        wins = 0
        points_scored = []
        points_allowed = []
        win_streak = 0
        current_streak = True  # True means we're still in a streak
        
        # Calculate metrics
        for _, game in team_games.iterrows():
            is_home = game['home_team_id'] == team_id
            team_score = game['home_score'] if is_home else game['away_score']
            opponent_score = game['away_score'] if is_home else game['home_score']
            
            points_scored.append(team_score)
            points_allowed.append(opponent_score)
            
            # Check if team won
            team_won = (is_home and game['home_win'] == 1) or (not is_home and game['home_win'] == 0)
            if team_won:
                wins += 1
                if current_streak:
                    win_streak += 1
            else:
                current_streak = False
        
        # Compute averages and percentages
        games_played = len(team_games)
        win_pct = wins / games_played if games_played > 0 else 0.5
        avg_scored = sum(points_scored) / games_played if games_played > 0 else 0
        avg_allowed = sum(points_allowed) / games_played if games_played > 0 else 0
        
        return {
            'recent_win_pct': win_pct,
            'avg_points_scored': avg_scored,
            'avg_points_allowed': avg_allowed,
            'win_streak': win_streak,
            'games_played': games_played
        }
    
    def calculate_head_to_head(self, df: pd.DataFrame, home_id: str, away_id: str,
                              date: pd.Timestamp, window: int = 10) -> Dict[str, float]:
        """Calculate head-to-head metrics between two teams"""
        # Get previous matchups between these teams
        h2h_games = df[(df['game_date'] < date) & 
                     (((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id)) |
                      ((df['home_team_id'] == away_id) & (df['away_team_id'] == home_id)))]
        
        # Sort and take recent games
        h2h_games = h2h_games.sort_values('game_date', ascending=False).head(window)
        
        if h2h_games.empty:
            return {
                'h2h_home_win_pct': 0.5,  # Default to 50% if no data
                'h2h_games_played': 0,
                'avg_point_diff': 0
            }
        
        # Initialize metrics
        home_wins = 0
        point_diffs = []
        
        # Calculate metrics
        for _, game in h2h_games.iterrows():
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            
            # Normalize so home_id is always considered the 'home' team
            if home_team_id == home_id and away_team_id == away_id:
                # Standard case (home vs away)
                home_wins += game['home_win']
                point_diff = game['home_score'] - game['away_score']
            else:
                # Reverse case (away vs home)
                home_wins += 1 - game['home_win']  # Invert win result
                point_diff = game['away_score'] - game['home_score']  # Invert point diff
            
            point_diffs.append(point_diff)
        
        games_played = len(h2h_games)
        home_win_pct = home_wins / games_played if games_played > 0 else 0.5
        avg_point_diff = sum(point_diffs) / games_played if games_played > 0 else 0
        
        return {
            'h2h_home_win_pct': home_win_pct,
            'h2h_games_played': games_played,
            'avg_point_diff': avg_point_diff
        }
    
    def calculate_rest_days(self, df: pd.DataFrame, team_id: str, game_date: pd.Timestamp) -> int:
        """Calculate rest days for a team before a game"""
        # Find the most recent game before game_date
        prev_games = df[(df['game_date'] < game_date) & 
                       ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id))]
        
        if prev_games.empty:
            return 3  # Default to 3 days of rest if no previous game found
        
        # Get the most recent game
        last_game_date = prev_games['game_date'].max()
        
        # Calculate days difference
        rest_days = (game_date - last_game_date).days
        
        return rest_days
    
    def create_game_features(self, df: pd.DataFrame, game_id: str) -> Dict[str, Any]:
        """Create comprehensive features for a specific game"""
        # Check if features already cached
        if game_id in self.cached_game_features:
            return self.cached_game_features[game_id]
        
        # Find the game in the dataframe
        game_row = df[df['id'] == game_id]
        if game_row.empty:
            logger.warning(f"Game ID {game_id} not found in dataframe")
            return {}
        
        game = game_row.iloc[0]
        game_date = game['game_date']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        
        # Calculate team form metrics
        home_form = self.calculate_team_form(df, home_team_id, game_date)
        away_form = self.calculate_team_form(df, away_team_id, game_date)
        
        # Calculate head-to-head metrics
        h2h_metrics = self.calculate_head_to_head(df, home_team_id, away_team_id, game_date)
        
        # Calculate rest days
        home_rest_days = self.calculate_rest_days(df, home_team_id, game_date)
        away_rest_days = self.calculate_rest_days(df, away_team_id, game_date)
        
        # Create feature dictionary
        features = {
            'game_id': game_id,
            'game_date': game_date.strftime('%Y-%m-%d'),
            'season': game['season'],
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': game['home_team_name'],
            'away_team_name': game['away_team_name'],
            
            # Home team metrics
            'home_recent_win_pct': home_form['recent_win_pct'],
            'home_avg_points_scored': home_form['avg_points_scored'],
            'home_avg_points_allowed': home_form['avg_points_allowed'],
            'home_win_streak': home_form['win_streak'],
            'home_games_played': home_form['games_played'],
            'home_rest_days': home_rest_days,
            
            # Away team metrics
            'away_recent_win_pct': away_form['recent_win_pct'],
            'away_avg_points_scored': away_form['avg_points_scored'],
            'away_avg_points_allowed': away_form['avg_points_allowed'],
            'away_win_streak': away_form['win_streak'],
            'away_games_played': away_form['games_played'],
            'away_rest_days': away_rest_days,
            
            # Head-to-head metrics
            'h2h_home_win_pct': h2h_metrics['h2h_home_win_pct'],
            'h2h_games_played': h2h_metrics['h2h_games_played'],
            'h2h_avg_point_diff': h2h_metrics['avg_point_diff'],
            
            # Comparative metrics
            'win_pct_diff': home_form['recent_win_pct'] - away_form['recent_win_pct'],
            'scoring_diff': home_form['avg_points_scored'] - away_form['avg_points_scored'],
            'defense_diff': away_form['avg_points_allowed'] - home_form['avg_points_allowed'],
            'rest_diff': home_rest_days - away_rest_days,
            
            # Home court advantage (can be tuned based on historical analysis)
            'home_court_advantage': 3.0
        }
        
        # Add outcomes if available (for training data)
        if not pd.isna(game['home_score']) and not pd.isna(game['away_score']):
            features.update({
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'home_win': game['home_win'],
                'spread': game['spread'],
                'total': game['total']
            })
        
        # Cache the features
        self.cached_game_features[game_id] = features
        
        return features
    
    def generate_features_for_all_games(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate features for all games in the dataset"""
        logger.info("Generating features for all games")
        
        all_features = []
        for game_id in df['id'].unique():
            try:
                features = self.create_game_features(df, game_id)
                if features:
                    all_features.append(features)
            except Exception as e:
                logger.error(f"Error generating features for game {game_id}: {str(e)}")
        
        logger.info(f"Generated features for {len(all_features)} games")
        
        # Save all features to file
        features_file = self.game_features_dir / 'all_game_features.json'
        with features_file.open('w') as f:
            json.dump(all_features, f, indent=2, cls=NumpyEncoder)
        
        return all_features
    
    def prepare_training_data(self, features: List[Dict[str, Any]], target: str = 'home_win') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from features
        
        Args:
            features: List of feature dictionaries
            target: Target variable ('home_win', 'spread', or 'total')
            
        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        logger.info(f"Preparing training data for target: {target}")
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Filter out games without outcomes (future games)
        df = df.dropna(subset=[target])
        
        # Select features
        feature_cols = [
            'home_recent_win_pct', 'home_avg_points_scored', 'home_avg_points_allowed',
            'home_win_streak', 'home_games_played', 'home_rest_days',
            'away_recent_win_pct', 'away_avg_points_scored', 'away_avg_points_allowed',
            'away_win_streak', 'away_games_played', 'away_rest_days',
            'h2h_home_win_pct', 'h2h_games_played', 'h2h_avg_point_diff',
            'win_pct_diff', 'scoring_diff', 'defense_diff', 'rest_diff',
            'home_court_advantage'
        ]
        
        # Make sure all columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Extract features and target
        X = df[feature_cols]
        y = df[target]
        
        logger.info(f"Prepared {len(X)} samples for training")
        
        return X, y


# Main function for testing
if __name__ == "__main__":
    # Test feature engineering
    engineer = NBAFeatureEngineer()
    
    # Load games data
    games_df = engineer.load_games()
    
    if not games_df.empty:
        # Generate features for all games
        features = engineer.generate_features_for_all_games(games_df)
        
        # Prepare training data
        X_train, y_train = engineer.prepare_training_data(features, target='home_win')
        print(f"Training data shape: {X_train.shape}")
        print(f"Feature columns: {X_train.columns.tolist()}")
        
        # Print sample predictions
        if len(X_train) > 0:
            print("\nFirst 5 samples:")
            print(X_train.head(5))
    else:
        print("No games data available. Please run the data collection process first.")
