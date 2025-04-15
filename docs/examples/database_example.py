"""
Database Module Example

This example demonstrates how to use the database module to interact with
the PostgreSQL database for the NBA prediction system.

It shows how to:
1. Initialize the database
2. Create and retrieve game data
3. Create and retrieve player data
4. Store and retrieve model weights and performance metrics

Usage:
    python -m src.examples.database_example
"""

import logging
import os
import sys
from datetime import datetime, timezone, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to make imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import database modules
from src.database.init_db import initialize_database
from src.database.models import Game, Player, ModelWeight, ModelPerformance


def example_game_operations():
    """
    Example of creating and retrieving game data
    """
    logger.info("\n=== Game Operations Example ===")
    
    # Create a sample game
    now = datetime.now(timezone.utc)
    game_data = {
        'game_id': f'sample_game_{now.strftime("%Y%m%d%H%M%S")}',
        'season_year': 2025,
        'date': now,
        'home_team_id': 'BOS',
        'away_team_id': 'LAL',
        'status': 'scheduled',
        'data': {
            'arena': 'TD Garden',
            'home_team_score': None,
            'away_team_score': None
        },
        'odds': {
            'moneyline': {
                'home': -150,
                'away': +130
            },
            'spread': {
                'home': -3.5,
                'away': +3.5,
                'home_odds': -110,
                'away_odds': -110
            }
        },
        'features': {
            'home_recent_form': 0.75,
            'away_recent_form': 0.65,
            'home_rest_days': 2,
            'away_rest_days': 1
        }
    }
    
    # Create the game in the database
    game_id = Game.create(game_data)
    if game_id:
        logger.info(f"Created game with ID: {game_id}")
    else:
        logger.error("Failed to create game")
        return
    
    # Find the game by game_id
    game = Game.find_by_game_id(game_data['game_id'])
    if game:
        logger.info(f"Retrieved game: {game['game_id']} - {game['home_team_id']} vs {game['away_team_id']}")
        logger.info(f"Game date: {game['date']}")
        logger.info(f"Odds: {game['odds']}")
    else:
        logger.error("Game not found")
    
    # Update the game with predictions
    update_result = Game.update(game_data['game_id'], {
        'predictions': {
            'win_probability': {
                'home': 0.62,
                'away': 0.38
            },
            'recommended_bet': {
                'type': 'moneyline',
                'team': 'home',
                'odds': -150,
                'confidence': 'medium',
                'bankroll_percent': 2.5
            }
        }
    })
    
    if update_result:
        logger.info("Game updated with predictions")
        
        # Retrieve the updated game
        updated_game = Game.find_by_game_id(game_data['game_id'])
        if updated_game:
            logger.info(f"Prediction win probability: {updated_game['predictions']['win_probability']}")
            logger.info(f"Recommended bet: {updated_game['predictions']['recommended_bet']}")
    else:
        logger.error("Failed to update game")
    
    # Find games for today
    today_games = Game.find_today_games()
    logger.info(f"Found {len(today_games)} games scheduled for today")
    
    # Find games by season
    season_games = Game.find_by_season(2025, limit=5)
    logger.info(f"Found {len(season_games)} games for the 2025 season")


def example_player_operations():
    """
    Example of creating and retrieving player data
    """
    logger.info("\n=== Player Operations Example ===")
    
    # Create sample players
    players = [
        {
            'player_id': 'player_123',
            'name': 'LeBron James',
            'team_id': 'LAL',
            'position': 'F',
            'data': {
                'height': '6-9',
                'weight': 250,
                'jersey': 23,
                'stats': {
                    'ppg': 25.8,
                    'rpg': 7.2,
                    'apg': 8.5
                }
            }
        },
        {
            'player_id': 'player_456',
            'name': 'Jayson Tatum',
            'team_id': 'BOS',
            'position': 'F',
            'data': {
                'height': '6-8',
                'weight': 210,
                'jersey': 0,
                'stats': {
                    'ppg': 28.3,
                    'rpg': 8.7,
                    'apg': 4.6
                }
            }
        }
    ]
    
    # Create players in the database
    for player_data in players:
        player_id = Player.create(player_data)
        if player_id:
            logger.info(f"Created player with ID: {player_id}")
        else:
            logger.error(f"Failed to create player: {player_data['name']}")
    
    # Find a player by player_id
    player = Player.find_by_player_id('player_123')
    if player:
        logger.info(f"Retrieved player: {player['name']} ({player['team_id']})")
        logger.info(f"Stats: {player['data']['stats']}")
    else:
        logger.error("Player not found")
    
    # Update a player with predictions
    update_result = Player.update('player_123', {
        'predictions': {
            'next_game': {
                'points': {
                    'prediction': 24.5,
                    'over_odds': -110,
                    'under_odds': -110,
                    'confidence': 0.68
                },
                'rebounds': {
                    'prediction': 7.5,
                    'over_odds': -115,
                    'under_odds': -105,
                    'confidence': 0.62
                },
                'assists': {
                    'prediction': 9.0,
                    'over_odds': -120,
                    'under_odds': +100,
                    'confidence': 0.73
                }
            }
        }
    })
    
    if update_result:
        logger.info("Player updated with predictions")
        
        # Retrieve updated player
        updated_player = Player.find_by_player_id('player_123')
        if updated_player and 'predictions' in updated_player:
            logger.info(f"Points prediction: {updated_player['predictions']['next_game']['points']}")
    else:
        logger.error("Failed to update player")
    
    # Find players by team
    team_players = Player.find_by_team('BOS')
    logger.info(f"Found {len(team_players)} players on the Boston Celtics")
    for player in team_players:
        logger.info(f"  - {player['name']} ({player['position']})")
    
    # Search players by name
    search_results = Player.search_by_name('Tat')
    logger.info(f"Found {len(search_results)} players matching 'Tat'")
    for player in search_results:
        logger.info(f"  - {player['name']} ({player['team_id']})")


def example_model_operations():
    """
    Example of creating and retrieving model data
    """
    logger.info("\n=== Model Operations Example ===")
    
    # Create a sample model weight record
    model_data = {
        'model_name': 'RandomForest',
        'model_type': 'classification',
        'weights': {
            'feature_importances': {
                'home_recent_form': 0.24,
                'away_recent_form': 0.22,
                'home_rest_days': 0.18,
                'away_rest_days': 0.15,
                'home_offensive_rating': 0.12,
                'away_offensive_rating': 0.09
            },
            'trees': [  # Simplified for example
                {'depth': 5, 'nodes': 31},
                {'depth': 4, 'nodes': 15},
                {'depth': 6, 'nodes': 63}
            ]
        },
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 5,
            'random_state': 42
        },
        'version': 1,
        'trained_at': datetime.now(timezone.utc),
        'active': True
    }
    
    # Create the model weight record
    model_id = ModelWeight.create(model_data)
    if model_id:
        logger.info(f"Created model weight record with ID: {model_id}")
    else:
        logger.error("Failed to create model weight record")
        return
    
    # Find the latest active version of the model
    model = ModelWeight.find_latest_active('RandomForest')
    if model:
        logger.info(f"Retrieved model: {model['model_name']} (version {model['version']})")
        logger.info(f"Trained at: {model['trained_at']}")
        logger.info(f"Parameters: {model['params']}")
    else:
        logger.error("Model not found")
    
    # Record model performance metrics
    performance_data = {
        'model_name': 'RandomForest',
        'date': datetime.now(timezone.utc),
        'metrics': {
            'accuracy': 0.73,
            'precision': 0.68,
            'recall': 0.71,
            'f1_score': 0.69,
            'auc_roc': 0.78,
            'profit_factor': 1.15
        },
        'prediction_type': 'moneyline',
        'num_predictions': 42,
        'window': '7d'
    }
    
    # Create the model performance record
    perf_id = ModelPerformance.create(performance_data)
    if perf_id:
        logger.info(f"Created model performance record with ID: {perf_id}")
    else:
        logger.error("Failed to create model performance record")
    
    # Get the latest performance metrics
    performance = ModelPerformance.get_latest_performance('RandomForest', 'moneyline', '7d')
    if performance:
        logger.info(f"Latest performance metrics for {performance['model_name']}:")
        logger.info(f"  Accuracy: {performance['metrics']['accuracy']}")
        logger.info(f"  Precision: {performance['metrics']['precision']}")
        logger.info(f"  Profit Factor: {performance['metrics']['profit_factor']}")
    else:
        logger.error("Model performance not found")
    
    # Get a 30-day performance trend (simulated for the example)
    # In a real scenario, we would have multiple records over time
    # Here we create some mock data points for demonstration
    now = datetime.now(timezone.utc)
    for i in range(5):
        day_offset = i * 7  # One data point per week
        date = now - timedelta(days=day_offset)
        
        # Slight variations in metrics
        accuracy = 0.70 + (i * 0.01)
        
        trend_data = {
            'model_name': 'RandomForest',
            'date': date,
            'metrics': {
                'accuracy': accuracy,
                'precision': 0.67 + (i * 0.005),
                'recall': 0.70 + (i * 0.008),
                'profit_factor': 1.12 + (i * 0.02)
            },
            'prediction_type': 'moneyline',
            'num_predictions': 40 + i,
            'window': '7d'
        }
        ModelPerformance.create(trend_data)
    
    # Get the performance trend
    trend = ModelPerformance.get_performance_trend('RandomForest', 'moneyline', '7d', days=30)
    logger.info(f"Performance trend contains {len(trend)} data points")
    for point in sorted(trend, key=lambda x: x['date']):
        logger.info(f"  {point['date'].strftime('%Y-%m-%d')}: Accuracy = {point['metrics']['accuracy']:.2f}, Profit Factor = {point['metrics']['profit_factor']:.2f}")


def main():
    """
    Main function to run all examples
    """
    logger.info("Starting Database Module Example")
    
    # Initialize the database
    success = initialize_database(verbose=True)
    if not success:
        logger.error("Database initialization failed. Exiting example.")
        return
    
    # Run all example operations
    example_game_operations()
    example_player_operations()
    example_model_operations()
    
    logger.info("\nDatabase Example completed successfully")


if __name__ == "__main__":
    main()
