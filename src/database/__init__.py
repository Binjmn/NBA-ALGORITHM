"""Database module for NBA Prediction System

This module provides database connectivity and models for the NBA prediction system.
It handles PostgreSQL connection management, schema definitions, and CRUD operations.
"""

from src.database.connection import get_db_connection, init_db
from src.database.models import (
    Game,
    Player,
    ModelWeight,
    ModelPerformance
)

__all__ = [
    'get_db_connection',
    'init_db',
    'Game',
    'Player',
    'ModelWeight',
    'ModelPerformance'
]
