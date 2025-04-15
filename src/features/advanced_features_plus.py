#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Feature Engineering for NBA Prediction System

This module extends the base feature engineering with additional advanced metrics
that capture more nuanced aspects of team and player performance, contextual factors,
and advanced statistical insights that can improve prediction accuracy.

New features include:
- Fatigue modeling based on schedule density and travel distance
- Team chemistry metrics based on lineup consistency
- Home court advantage refinements by venue
- Injury impact assessment beyond binary indicators
- Momentum and streak-based features with statistical significance
- Matchup-specific historical performance metrics
- Advanced pace and style compatibility indicators
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic

from src.features.advanced_features import FeatureEngineer as BaseFeatureEngineer

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer(BaseFeatureEngineer):
    """
    Enhanced Feature Engineer for NBA games that extends the base engineer with
    additional advanced metrics and contextual factors
    """
    
    def __init__(self, lookback_days: int = 30, include_advanced_metrics: bool = True):
        """
        Initialize the enhanced feature engineer
        
        Args:
            lookback_days: Number of days to look back for historical performance metrics
            include_advanced_metrics: Whether to include computationally expensive advanced metrics
        """
        # Initialize base feature engineer
        super().__init__(lookback_days=lookback_days)
        
        # Additional settings
        self.include_advanced_metrics = include_advanced_metrics
        
        # NBA arena locations for travel distance calculation (latitude, longitude)
        self.arena_locations = {
            'Atlanta Hawks': (33.7573, -84.3963),  # State Farm Arena
            'Boston Celtics': (42.3662, -71.0621),  # TD Garden
            'Brooklyn Nets': (40.6828, -73.9758),  # Barclays Center
            'Charlotte Hornets': (35.2251, -80.8392),  # Spectrum Center
            'Chicago Bulls': (41.8807, -87.6742),  # United Center
            'Cleveland Cavaliers': (41.4967, -81.6881),  # Rocket Mortgage FieldHouse
            'Dallas Mavericks': (32.7905, -96.8103),  # American Airlines Center
            'Denver Nuggets': (39.7487, -105.0077),  # Ball Arena
            'Detroit Pistons': (42.3410, -83.0550),  # Little Caesars Arena
            'Golden State Warriors': (37.7680, -122.3877),  # Chase Center
            'Houston Rockets': (29.7508, -95.3621),  # Toyota Center
            'Indiana Pacers': (39.7640, -86.1555),  # Gainbridge Fieldhouse
            'Los Angeles Clippers': (34.0430, -118.2673),  # Crypto.com Arena
            'Los Angeles Lakers': (34.0430, -118.2673),  # Crypto.com Arena
            'Memphis Grizzlies': (35.1382, -90.0505),  # FedExForum
            'Miami Heat': (25.7814, -80.1870),  # Kaseya Center
            'Milwaukee Bucks': (43.0451, -87.9173),  # Fiserv Forum
            'Minnesota Timberwolves': (44.9795, -93.2760),  # Target Center
            'New Orleans Pelicans': (29.9490, -90.0821),  # Smoothie King Center
            'New York Knicks': (40.7505, -73.9934),  # Madison Square Garden
            'Oklahoma City Thunder': (35.4634, -97.5151),  # Paycom Center
            'Orlando Magic': (28.5393, -81.3839),  # Kia Center
            'Philadelphia 76ers': (39.9012, -75.1720),  # Wells Fargo Center
            'Phoenix Suns': (33.4458, -112.0712),  # Footprint Center
            'Portland Trail Blazers': (45.5316, -122.6668),  # Moda Center
            'Sacramento Kings': (38.5803, -121.4996),  # Golden 1 Center
            'San Antonio Spurs': (29.4271, -98.4375),  # Frost Bank Center
            'Toronto Raptors': (43.6435, -79.3791),  # Scotiabank Arena
            'Utah Jazz': (40.7683, -111.9011),  # Delta Center
            'Washington Wizards': (38.8981, -77.0209),  # Capital One Arena
        }
        
        # Configure advanced statistical approaches
        self.scaler = StandardScaler()

    def engineer_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer enhanced features for NBA game prediction
        
        Args:
            games_df: DataFrame containing raw game data
            
        Returns:
            DataFrame with all base and enhanced features
        """
        # First, generate all base features
        features_df = super().engineer_features(games_df)
        
        logger.info("Adding enhanced features")
        
        try:            
            # Add fatigue metrics
            features_df = self._add_fatigue_metrics(features_df, games_df)
            
            # Add team chemistry metrics
            features_df = self._add_team_chemistry_metrics(features_df, games_df)
            
            # Add venue-specific advantages
            features_df = self._add_venue_advantage(features_df, games_df)
            
            # Add enhanced injury metrics
            features_df = self._add_injury_impact(features_df, games_df)
            
            # Add momentum and streak metrics
            features_df = self._add_momentum_metrics(features_df, games_df)
            
            # Add advanced matchup-specific metrics
            features_df = self._add_matchup_metrics(features_df, games_df)
            
            # Add advanced pace and style compatibility metrics
            features_df = self._add_style_compatibility(features_df, games_df)
            
            # Handle missing values with appropriate imputation
            features_df = self._handle_missing_values(features_df)
            
            logger.info(f"Enhanced feature engineering complete: {len(features_df.columns)} total features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error in enhanced feature engineering: {str(e)}")
            # Return base features if enhanced engineering fails
            return features_df

    def _add_fatigue_metrics(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fatigue metrics based on schedule density and travel distance
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame with schedule information
            
        Returns:
            DataFrame with fatigue metrics added
        """
        try:
            # Sort games by date for each team
            sorted_games = games_df.sort_values(['date'])
            
            # Initialize new columns
            features_df['home_games_last_7d'] = 0
            features_df['away_games_last_7d'] = 0
            features_df['home_travel_distance_km'] = 0.0
            features_df['away_travel_distance_km'] = 0.0
            features_df['home_back_to_back'] = 0
            features_df['away_back_to_back'] = 0
            features_df['home_three_in_four'] = 0
            features_df['away_three_in_four'] = 0
            
            # Calculate schedule density metrics
            for idx, game in features_df.iterrows():
                game_date = game['date'] if isinstance(game['date'], datetime) else pd.to_datetime(game['date'])
                start_date = game_date - timedelta(days=7)
                
                # Get all games in the last 7 days for each team
                home_recent_games = sorted_games[
                    ((sorted_games['home_team'] == game['home_team']) | 
                     (sorted_games['away_team'] == game['home_team'])) & 
                    (sorted_games['date'] >= start_date) & 
                    (sorted_games['date'] < game_date)
                ]
                
                away_recent_games = sorted_games[
                    ((sorted_games['home_team'] == game['away_team']) | 
                     (sorted_games['away_team'] == game['away_team'])) & 
                    (sorted_games['date'] >= start_date) & 
                    (sorted_games['date'] < game_date)
                ]
                
                # Games in the last 7 days
                features_df.at[idx, 'home_games_last_7d'] = len(home_recent_games)
                features_df.at[idx, 'away_games_last_7d'] = len(away_recent_games)
                
                # Check for back-to-back games
                if len(home_recent_games) > 0 and game_date - pd.to_datetime(home_recent_games['date'].iloc[-1]) <= timedelta(days=1):
                    features_df.at[idx, 'home_back_to_back'] = 1
                    
                if len(away_recent_games) > 0 and game_date - pd.to_datetime(away_recent_games['date'].iloc[-1]) <= timedelta(days=1):
                    features_df.at[idx, 'away_back_to_back'] = 1
                
                # Check for three games in four days
                if len(home_recent_games) >= 2 and game_date - pd.to_datetime(home_recent_games['date'].iloc[-2]) <= timedelta(days=3):
                    features_df.at[idx, 'home_three_in_four'] = 1
                    
                if len(away_recent_games) >= 2 and game_date - pd.to_datetime(away_recent_games['date'].iloc[-2]) <= timedelta(days=3):
                    features_df.at[idx, 'away_three_in_four'] = 1
                
                # Calculate travel distances if possible
                if self.include_advanced_metrics and len(home_recent_games) > 0 and len(away_recent_games) > 0:
                    # Get last game location for home team
                    if home_recent_games['home_team'].iloc[-1] == game['home_team']:
                        last_home_loc = self.arena_locations.get(game['home_team'], (0, 0))
                    else:
                        last_home_loc = self.arena_locations.get(home_recent_games['home_team'].iloc[-1], (0, 0))
                    
                    # Get last game location for away team
                    if away_recent_games['home_team'].iloc[-1] == game['away_team']:
                        last_away_loc = self.arena_locations.get(game['away_team'], (0, 0))
                    else:
                        last_away_loc = self.arena_locations.get(away_recent_games['home_team'].iloc[-1], (0, 0))
                    
                    # Current game location (home team's arena)
                    current_loc = self.arena_locations.get(game['home_team'], (0, 0))
                    
                    # Calculate distances
                    if sum(last_home_loc) > 0 and sum(current_loc) > 0:
                        features_df.at[idx, 'home_travel_distance_km'] = geodesic(last_home_loc, current_loc).kilometers
                        
                    if sum(last_away_loc) > 0 and sum(current_loc) > 0:
                        features_df.at[idx, 'away_travel_distance_km'] = geodesic(last_away_loc, current_loc).kilometers
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding fatigue metrics: {str(e)}")
            return features_df

    def _add_team_chemistry_metrics(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team chemistry metrics based on lineup consistency
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with team chemistry metrics added
        """
        # This is a placeholder - in a real implementation, we would use
        # lineup data to calculate consistency scores and chemistry metrics
        
        # Since we don't have actual lineup data in this example, we'll use dummy values
        # that could be replaced with real calculations in production
        features_df['home_lineup_consistency'] = 0.75  # Default dummy value 
        features_df['away_lineup_consistency'] = 0.75  # Default dummy value
        
        return features_df

    def _add_venue_advantage(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add venue-specific home court advantage metrics
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with venue advantage metrics added
        """
        # Calculate historical home court advantage by team
        # This could be enhanced with actual historical data analysis
        
        # Default home court advantage values
        home_advantage = {
            'Boston Celtics': 3.5,      # Strong home court
            'Golden State Warriors': 3.7,  # Historically strong at home
            'Denver Nuggets': 3.6,      # Altitude advantage
            'Utah Jazz': 3.6,           # Altitude advantage
            'Milwaukee Bucks': 3.4,
            'Phoenix Suns': 3.3,
            'Miami Heat': 3.2,
            # Add estimated values for all teams
        }
        
        # Apply home court advantage values
        features_df['home_court_advantage'] = features_df['home_team'].map(
            lambda x: home_advantage.get(x, 3.0)  # Default to 3.0 if team not found
        )
        
        # Add attendance percentage if available (placeholder)
        features_df['attendance_pct'] = 0.9  # Default dummy value
        
        return features_df

    def _add_injury_impact(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced injury impact assessments beyond binary indicators
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with enhanced injury metrics added
        """
        # In a real implementation, we would use actual injury data with player value metrics
        # Placeholder implementation that could be replaced with real calculations
        
        # Default impact values - these would normally come from actual injury reports
        # and player impact metrics like VORP, RPM, etc.
        features_df['home_star_player_injured'] = 0  # Binary indicator for demonstration
        features_df['away_star_player_injured'] = 0  # Binary indicator for demonstration
        features_df['home_injury_impact_score'] = 0.0  # Scaled impact score
        features_df['away_injury_impact_score'] = 0.0  # Scaled impact score
        
        return features_df

    def _add_momentum_metrics(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum and streak-based metrics with statistical significance
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with momentum metrics added
        """
        # Sort games by date for streak calculations
        sorted_games = games_df.sort_values(['date'])
        
        # Initialize streak columns
        features_df['home_win_streak'] = 0
        features_df['away_win_streak'] = 0
        features_df['home_momentum_score'] = 0.0
        features_df['away_momentum_score'] = 0.0
        
        # Calculate win streaks and momentum scores
        for idx, game in features_df.iterrows():
            game_date = game['date'] if isinstance(game['date'], datetime) else pd.to_datetime(game['date'])
            
            # Get recent games for home team (last 10 games)
            home_recent_games = sorted_games[
                ((sorted_games['home_team'] == game['home_team']) | 
                 (sorted_games['away_team'] == game['home_team'])) & 
                (sorted_games['date'] < game_date)
            ].tail(10)
            
            # Get recent games for away team (last 10 games)
            away_recent_games = sorted_games[
                ((sorted_games['home_team'] == game['away_team']) | 
                 (sorted_games['away_team'] == game['away_team'])) & 
                (sorted_games['date'] < game_date)
            ].tail(10)
            
            # Calculate win streaks
            home_wins = []
            for _, recent_game in home_recent_games.iterrows():
                if recent_game['home_team'] == game['home_team']:
                    # Home team was home in this game
                    home_wins.append(1 if recent_game['home_score'] > recent_game['away_score'] else 0)
                else:
                    # Home team was away in this game
                    home_wins.append(1 if recent_game['away_score'] > recent_game['home_score'] else 0)
            
            away_wins = []
            for _, recent_game in away_recent_games.iterrows():
                if recent_game['home_team'] == game['away_team']:
                    # Away team was home in this game
                    away_wins.append(1 if recent_game['home_score'] > recent_game['away_score'] else 0)
                else:
                    # Away team was away in this game
                    away_wins.append(1 if recent_game['away_score'] > recent_game['home_score'] else 0)
            
            # Calculate current win streak
            home_streak = 0
            for win in reversed(home_wins):
                if win == 1:
                    home_streak += 1
                else:
                    break
            
            away_streak = 0
            for win in reversed(away_wins):
                if win == 1:
                    away_streak += 1
                else:
                    break
            
            features_df.at[idx, 'home_win_streak'] = home_streak
            features_df.at[idx, 'away_win_streak'] = away_streak
            
            # Calculate weighted momentum score (more recent games have higher weights)
            if home_wins:
                weights = np.linspace(0.5, 1.0, len(home_wins))
                features_df.at[idx, 'home_momentum_score'] = np.average(home_wins, weights=weights)
            
            if away_wins:
                weights = np.linspace(0.5, 1.0, len(away_wins))
                features_df.at[idx, 'away_momentum_score'] = np.average(away_wins, weights=weights)
        
        return features_df

    def _add_matchup_metrics(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced matchup-specific historical performance metrics
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with matchup-specific metrics added
        """
        # Sort games by date
        sorted_games = games_df.sort_values(['date'])
        
        # Initialize matchup columns
        features_df['matchup_home_win_pct'] = 0.5  # Default to 50%
        features_df['matchup_games_count'] = 0
        features_df['matchup_avg_point_diff'] = 0.0
        
        # Calculate head-to-head metrics
        for idx, game in features_df.iterrows():
            game_date = game['date'] if isinstance(game['date'], datetime) else pd.to_datetime(game['date'])
            
            # Find all previous matchups between these teams (regardless of home/away)
            matchups = sorted_games[
                (((sorted_games['home_team'] == game['home_team']) & 
                  (sorted_games['away_team'] == game['away_team'])) | 
                 ((sorted_games['home_team'] == game['away_team']) & 
                  (sorted_games['away_team'] == game['home_team']))) & 
                (sorted_games['date'] < game_date)
            ]
            
            matchup_count = len(matchups)
            features_df.at[idx, 'matchup_games_count'] = matchup_count
            
            if matchup_count > 0:
                # Calculate home team's win percentage in matchups
                home_team_wins = 0
                point_diffs = []
                
                for _, matchup in matchups.iterrows():
                    if matchup['home_team'] == game['home_team']:
                        # Current home team was home in this matchup
                        point_diff = matchup['home_score'] - matchup['away_score']
                        home_team_wins += 1 if point_diff > 0 else 0
                        point_diffs.append(point_diff)
                    else:
                        # Current home team was away in this matchup
                        point_diff = matchup['away_score'] - matchup['home_score']
                        home_team_wins += 1 if point_diff > 0 else 0
                        point_diffs.append(point_diff)
                
                features_df.at[idx, 'matchup_home_win_pct'] = home_team_wins / matchup_count
                
                if point_diffs:
                    features_df.at[idx, 'matchup_avg_point_diff'] = np.mean(point_diffs)
        
        return features_df

    def _add_style_compatibility(self, features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced pace and style compatibility indicators
        
        Args:
            features_df: DataFrame with base features
            games_df: Original games DataFrame
            
        Returns:
            DataFrame with style compatibility metrics added
        """
        # This would typically use team pace, shot selection, and play type data
        # Placeholder implementation with dummy values
        
        # Team style characteristics (placeholders)
        team_styles = {
            'Boston Celtics': {'pace': 99.5, 'three_pt_pct': 37.8, 'post_up_freq': 5.2},
            'Golden State Warriors': {'pace': 102.8, 'three_pt_pct': 38.5, 'post_up_freq': 3.1},
            # Add data for other teams
        }
        
        # Default values for teams not in the dictionary
        default_style = {'pace': 100.0, 'three_pt_pct': 36.0, 'post_up_freq': 5.0}
        
        # Calculate style compatibility (simplistic model for illustration)
        for idx, game in features_df.iterrows():
            home_style = team_styles.get(game['home_team'], default_style)
            away_style = team_styles.get(game['away_team'], default_style)
            
            # Calculate pace differential (affects total points)
            pace_diff = home_style.get('pace', 100.0) - away_style.get('pace', 100.0)
            features_df.at[idx, 'pace_differential'] = pace_diff
            
            # Calculate shooting style compatibility
            shooting_diff = home_style.get('three_pt_pct', 36.0) - away_style.get('three_pt_pct', 36.0)
            features_df.at[idx, 'shooting_style_diff'] = shooting_diff
            
            # Calculate post play vs perimeter play compatibility
            post_diff = home_style.get('post_up_freq', 5.0) - away_style.get('post_up_freq', 5.0)
            features_df.at[idx, 'post_play_diff'] = post_diff
        
        return features_df

    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate imputation strategies
        
        Args:
            features_df: DataFrame with potentially missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Fill missing values for numeric columns with median
        numeric_cols = features_df.select_dtypes(include=['int', 'float']).columns
        for col in numeric_cols:
            if features_df[col].isnull().any():
                median_value = features_df[col].median()
                features_df[col].fillna(median_value, inplace=True)
        
        # Fill missing values for categorical columns with mode
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if features_df[col].isnull().any():
                mode_value = features_df[col].mode()[0]
                features_df[col].fillna(mode_value, inplace=True)
        
        return features_df
