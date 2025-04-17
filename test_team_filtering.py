#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test NBA Team Filtering

Verifies that our NBA team filtering correctly identifies current NBA teams
and filters out historical teams.
"""

import logging
logging.basicConfig(level=logging.INFO)

from nba_algorithm.data.team_data import fetch_all_teams
from nba_algorithm.data.nba_teams import get_active_nba_teams

# Get all teams from the API
teams = fetch_all_teams()
print(f"\nRetrieved {len(teams)} total teams from API")

# Filter to active NBA teams only
active_teams = get_active_nba_teams(teams)

# Show the results
print(f"\nFound {len(active_teams)} active NBA teams:")
for team_id, team_name in sorted(active_teams.items(), key=lambda x: x[1]):
    print(f"  - {team_name} (ID: {team_id})")

# Check for historical teams that were filtered out
all_team_ids = {team['id'] for team in teams if 'id' in team}
active_team_ids = set(active_teams.keys())
historical_team_ids = all_team_ids - active_team_ids

historical_teams = [team for team in teams if team.get('id') in historical_team_ids]

print(f"\nFiltered out {len(historical_teams)} historical or non-NBA teams:")
for team in sorted(historical_teams, key=lambda x: x.get('full_name', ''))[:10]:  # Show first 10
    print(f"  - {team.get('full_name', 'Unknown')} (ID: {team.get('id', 'Unknown')})")

if len(historical_teams) > 10:
    print(f"  - ... and {len(historical_teams) - 10} more")
