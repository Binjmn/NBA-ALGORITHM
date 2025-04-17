"""
Database Models for NBA Prediction System

This module provides ORM-style model classes for working with the PostgreSQL database.
Each class represents a table in the database and provides methods for CRUD operations.

The models are designed to support the prediction system's requirements for storing
game data, player data, model weights, and model performance metrics.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import psycopg2
from psycopg2.extras import Json

from src.database.connection import get_db_connection, get_dict_cursor

# Configure logging
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for all database models providing common functionality
    """
    
    table_name = None
    primary_key = 'id'
    
    @classmethod
    def find_by_id(cls, item_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Find a record by its primary key
        
        Args:
            item_id: The primary key value to search for
            
        Returns:
            Dict containing the record data or None if not found
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {cls.table_name} WHERE {cls.primary_key} = %s",
                    (item_id,)
                )
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error finding {cls.table_name} by ID: {str(e)}")
            return None
    
    @classmethod
    def find_all(cls, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Find all records with pagination
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of dictionaries containing the record data
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {cls.table_name} ORDER BY {cls.primary_key} LIMIT %s OFFSET %s",
                    (limit, offset)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error finding all {cls.table_name}: {str(e)}")
            return []
    
    @staticmethod
    def _build_where_clause(filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build a WHERE clause from a dictionary of filters
        
        Args:
            filters: Dictionary of column names and values to filter by
            
        Returns:
            Tuple of (where_clause, params)
        """
        if not filters:
            return "", []
            
        where_parts = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                where_parts.append(f"{key} IN %s")
                params.append(tuple(value))
            elif value is None:
                where_parts.append(f"{key} IS NULL")
            else:
                where_parts.append(f"{key} = %s")
                params.append(value)
                
        where_clause = " AND ".join(where_parts)
        return where_clause, params


class Game(BaseModel):
    """
    Model for the games table
    
    Stores information about NBA games, including team data, odds, features, and predictions
    """
    
    table_name = 'games'
    primary_key = 'id'
    
    @classmethod
    def create(cls, game_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new game record
        
        Args:
            game_data: Dictionary containing game data
            
        Returns:
            ID of the created record or None if creation failed
        """
        required_fields = ['game_id', 'season_year', 'date', 'home_team_id', 'away_team_id', 'status']
        
        # Validate required fields
        for field in required_fields:
            if field not in game_data:
                logger.error(f"Missing required field '{field}' for game creation")
                return None
        
        # Ensure JSON fields are properly formatted
        for json_field in ['data', 'odds', 'features', 'predictions']:
            if json_field in game_data and not isinstance(game_data[json_field], dict):
                logger.error(f"Field '{json_field}' must be a dictionary")
                return None
            elif json_field not in game_data:
                game_data[json_field] = {}
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO games 
                    (game_id, season_year, date, home_team_id, away_team_id, status, data, odds, features, predictions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """, (
                        game_data['game_id'],
                        game_data['season_year'],
                        game_data['date'],
                        game_data['home_team_id'],
                        game_data['away_team_id'],
                        game_data['status'],
                        Json(game_data.get('data', {})),
                        Json(game_data.get('odds', {})),
                        Json(game_data.get('features', {})),
                        Json(game_data.get('predictions', {}))
                    ))
                    
                    result = cursor.fetchone()
                    conn.commit()
                    
                    if result and result[0]:
                        logger.info(f"Created game record with ID {result[0]}")
                        return result[0]
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error creating game record: {str(e)}")
            return None
    
    @classmethod
    def update(cls, game_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a game record by game_id
        
        Args:
            game_id: The game_id to update
            updates: Dictionary of field names and values to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not updates:
            logger.warning("No updates provided for game update")
            return False
            
        # Handle JSON fields properly
        json_fields = ['data', 'odds', 'features', 'predictions']
        for field in json_fields:
            if field in updates and not isinstance(updates[field], dict):
                logger.error(f"Field '{field}' must be a dictionary")
                return False
        
        try:
            set_parts = []
            params = []
            
            for key, value in updates.items():
                if key in json_fields:
                    # For JSON fields, use jsonb_set to update
                    set_parts.append(f"{key} = %s")
                    params.append(Json(value))
                else:
                    set_parts.append(f"{key} = %s")
                    params.append(value)
            
            set_clause = ", ".join(set_parts)
            params.append(game_id)  # For the WHERE clause
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    UPDATE games 
                    SET {set_clause}
                    WHERE game_id = %s
                    """, params)
                    
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"Updated game with game_id {game_id}")
                        return True
                    else:
                        logger.warning(f"No game found with game_id {game_id}")
                        return False
        except Exception as e:
            logger.error(f"Error updating game record: {str(e)}")
            return False
    
    @classmethod
    def find_by_game_id(cls, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a game by its game_id
        
        Args:
            game_id: The game_id to search for
            
        Returns:
            Dict containing the game data or None if not found
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM games WHERE game_id = %s",
                    (game_id,)
                )
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error finding game by game_id: {str(e)}")
            return None
    
    @classmethod
    def find_by_date_range(cls, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Find games within a date range
        
        Args:
            start_date: Start of the date range
            end_date: End of the date range
            
        Returns:
            List of dictionaries containing game data
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM games WHERE date >= %s AND date <= %s ORDER BY date",
                    (start_date, end_date)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error finding games by date range: {str(e)}")
            return []
    
    @classmethod
    def find_today_games(cls) -> List[Dict[str, Any]]:
        """
        Find games scheduled for today
        
        Returns:
            List of dictionaries containing game data
        """
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
        end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=timezone.utc)
        
        return cls.find_by_date_range(start_of_day, end_of_day)
    
    @classmethod
    def find_by_season(cls, season_year: int, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Find games for a specific season
        
        Args:
            season_year: The NBA season year to find games for
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of dictionaries containing game data
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM games WHERE season_year = %s ORDER BY date DESC LIMIT %s OFFSET %s",
                    (season_year, limit, offset)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error finding games by season: {str(e)}")
            return []


class Player(BaseModel):
    """
    Model for the players table
    
    Stores information about NBA players, including stats, features, and predictions
    """
    
    table_name = 'players'
    primary_key = 'id'
    
    @classmethod
    def create(cls, player_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new player record
        
        Args:
            player_data: Dictionary containing player data
            
        Returns:
            ID of the created record or None if creation failed
        """
        required_fields = ['player_id', 'name']
        
        # Validate required fields
        for field in required_fields:
            if field not in player_data:
                logger.error(f"Missing required field '{field}' for player creation")
                return None
        
        # Ensure JSON fields are properly formatted
        for json_field in ['data', 'features', 'predictions']:
            if json_field in player_data and not isinstance(player_data[json_field], dict):
                logger.error(f"Field '{json_field}' must be a dictionary")
                return None
            elif json_field not in player_data:
                player_data[json_field] = {}
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO players 
                    (player_id, name, team_id, position, data, features, predictions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """, (
                        player_data['player_id'],
                        player_data['name'],
                        player_data.get('team_id'),
                        player_data.get('position'),
                        Json(player_data.get('data', {})),
                        Json(player_data.get('features', {})),
                        Json(player_data.get('predictions', {}))
                    ))
                    
                    result = cursor.fetchone()
                    conn.commit()
                    
                    if result and result[0]:
                        logger.info(f"Created player record with ID {result[0]}")
                        return result[0]
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error creating player record: {str(e)}")
            return None
    
    @classmethod
    def update(cls, player_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a player record by player_id
        
        Args:
            player_id: The player_id to update
            updates: Dictionary of field names and values to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not updates:
            logger.warning("No updates provided for player update")
            return False
            
        # Handle JSON fields properly
        json_fields = ['data', 'features', 'predictions']
        for field in json_fields:
            if field in updates and not isinstance(updates[field], dict):
                logger.error(f"Field '{field}' must be a dictionary")
                return False
        
        try:
            set_parts = []
            params = []
            
            for key, value in updates.items():
                if key in json_fields:
                    # For JSON fields, use jsonb_set to update
                    set_parts.append(f"{key} = %s")
                    params.append(Json(value))
                else:
                    set_parts.append(f"{key} = %s")
                    params.append(value)
            
            set_clause = ", ".join(set_parts)
            params.append(player_id)  # For the WHERE clause
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    UPDATE players 
                    SET {set_clause}
                    WHERE player_id = %s
                    """, params)
                    
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"Updated player with player_id {player_id}")
                        return True
                    else:
                        logger.warning(f"No player found with player_id {player_id}")
                        return False
        except Exception as e:
            logger.error(f"Error updating player record: {str(e)}")
            return False
    
    @classmethod
    def find_by_player_id(cls, player_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a player by player_id
        
        Args:
            player_id: The player_id to search for
            
        Returns:
            Dict containing the player data or None if not found
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM players WHERE player_id = %s",
                    (player_id,)
                )
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error finding player by player_id: {str(e)}")
            return None
    
    @classmethod
    def find_by_team(cls, team_id: str) -> List[Dict[str, Any]]:
        """
        Find all players on a specific team
        
        Args:
            team_id: The team_id to search for
            
        Returns:
            List of dictionaries containing player data
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM players WHERE team_id = %s ORDER BY name",
                    (team_id,)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error finding players by team: {str(e)}")
            return []
    
    @classmethod
    def search_by_name(cls, name_query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for players by name
        
        Args:
            name_query: The name to search for (partial match)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing player data
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM players WHERE name ILIKE %s ORDER BY name LIMIT %s",
                    (f"%{name_query}%", limit)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error searching players by name: {str(e)}")
            return []


class ModelWeight(BaseModel):
    """
    Model for the model_weights table
    
    Stores weights and parameters for trained prediction models
    """
    
    table_name = 'model_weights'
    primary_key = 'id'
    
    @classmethod
    def create(cls, model_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new model weight record
        
        Args:
            model_data: Dictionary containing model data
            
        Returns:
            ID of the created record or None if creation failed
        """
        required_fields = ['model_name', 'model_type', 'weights', 'version']
        
        # Validate required fields
        for field in required_fields:
            if field not in model_data:
                logger.error(f"Missing required field '{field}' for model weight creation")
                return None
        
        # Ensure JSON fields are properly formatted
        for json_field in ['weights', 'params']:
            if json_field in model_data and not isinstance(model_data[json_field], dict):
                logger.error(f"Field '{json_field}' must be a dictionary")
                return None
            elif json_field not in model_data and json_field == 'params':
                model_data[json_field] = {}
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO model_weights 
                    (model_name, model_type, weights, params, version, trained_at, active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """, (
                        model_data['model_name'],
                        model_data['model_type'],
                        Json(model_data['weights']),
                        Json(model_data.get('params', {})),
                        model_data['version'],
                        model_data.get('trained_at', datetime.now(timezone.utc)),
                        model_data.get('active', True)
                    ))
                    
                    result = cursor.fetchone()
                    conn.commit()
                    
                    if result and result[0]:
                        logger.info(f"Created model weight record with ID {result[0]}")
                        return result[0]
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error creating model weight record: {str(e)}")
            return None
    
    @classmethod
    def find_latest_active(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Find the latest active version of a model
        
        Args:
            model_name: The name of the model to search for
            
        Returns:
            Dict containing the model weight data or None if not found
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    """SELECT * FROM model_weights 
                       WHERE model_name = %s AND active = true 
                       ORDER BY version DESC LIMIT 1""",
                    (model_name,)
                )
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error finding latest active model: {str(e)}")
            return None
    
    @classmethod
    def deactivate_old_versions(cls, model_name: str, current_version: int) -> bool:
        """
        Deactivate all versions of a model except the current one
        
        Args:
            model_name: The name of the model
            current_version: The version to keep active
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """UPDATE model_weights 
                           SET active = false 
                           WHERE model_name = %s AND version != %s""",
                        (model_name, current_version)
                    )
                    conn.commit()
                    
                    logger.info(f"Deactivated old versions of model {model_name}")
                    return True
        except Exception as e:
            logger.error(f"Error deactivating old model versions: {str(e)}")
            return False


class ModelPerformance(BaseModel):
    """
    Model for the model_performance table
    
    Stores performance metrics for prediction models
    """
    
    table_name = 'model_performance'
    primary_key = 'id'
    
    @classmethod
    def create(cls, performance_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new model performance record
        
        Args:
            performance_data: Dictionary containing performance data
            
        Returns:
            ID of the created record or None if creation failed
        """
        required_fields = ['model_name', 'prediction_target', 'metrics']
        
        # Validate required fields
        for field in required_fields:
            if field not in performance_data:
                logger.error(f"Missing required field '{field}' for model performance creation")
                return None
        
        # Ensure metrics is a dictionary
        if not isinstance(performance_data['metrics'], dict):
            logger.error("Field 'metrics' must be a dictionary")
            return None
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO model_performance 
                    (model_name, prediction_target, metrics, is_baseline, "time_window")
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """, (
                        performance_data['model_name'],
                        performance_data['prediction_target'],
                        Json(performance_data['metrics']),
                        performance_data.get('is_baseline', False),
                        performance_data.get('time_window', '7d')
                    ))
                    
                    result = cursor.fetchone()
                    conn.commit()
                    
                    if result and result[0]:
                        logger.info(f"Created model performance record with ID {result[0]}")
                        return result[0]
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error creating model performance record: {str(e)}")
            return None
    
    @classmethod
    def get_latest_performance(cls, model_name: str, prediction_target: str, window: str = '7d', time_window_column: str = "time_window") -> Optional[Dict[str, Any]]:
        """
        Get the latest performance metrics for a model
        
        Args:
            model_name: The name of the model
            prediction_target: Type of prediction (e.g., 'moneyline', 'spread', 'player_points')
            window: The time window for the metrics ('7d', '30d', 'season')
            time_window_column: Name of the column that stores the time window
            
        Returns:
            Dict containing the performance data or None if not found
        """
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    f"""SELECT * FROM model_performance 
                       WHERE model_name = %s AND prediction_target = %s AND \"{time_window_column}\" = %s 
                       ORDER BY created_at DESC LIMIT 1""",
                    (model_name, prediction_target, window)
                )
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting latest model performance: {str(e)}")
            return None
    
    @classmethod
    def get_performance_trend(cls, model_name: str, prediction_target: str, window: str = '7d', days: int = 30, time_window_column: str = "time_window") -> List[Dict[str, Any]]:
        """
        Get the performance trend for a model over time
        
        Args:
            model_name: The name of the model
            prediction_target: Type of prediction
            window: The time window for the metrics
            days: Number of days of history to retrieve
            time_window_column: Name of the column that stores the time window
            
        Returns:
            List of dictionaries containing performance data ordered by date
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        try:
            with get_dict_cursor() as cursor:
                cursor.execute(
                    f"""SELECT * FROM model_performance 
                       WHERE model_name = %s AND prediction_target = %s AND \"{time_window_column}\" = %s 
                       AND created_at BETWEEN %s AND %s
                       ORDER BY created_at ASC""",
                    (model_name, prediction_target, window, start_date, end_date)
                )
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting model performance trend: {str(e)}")
            return []
