def generate_mapped_data(game_boxscores, starting_lineup_dict):
    """
    Generates a mapped data structure containing player stats organized by game ID,
    differentiating between home and away players based on the starting lineup.

    Parameters:
    - game_boxscores (dict): The game boxscores data.
    - starting_lineup_dict (dict): A dictionary containing the starting lineup for each game.

    Returns:
    - dict: The mapped data with player statistics organized by game ID.
    """
    mapped_data = {}

    # Iterate through each game's lineup in starting_lineup_dict
    for game_id_key, lineup in starting_lineup_dict.items():
        # Initialize storage for this game's data
        game_data = {'home_players': [], 'away_players': []}
        
        # Find the matching game in game_boxscores by gameId
        for game_num, game_info in game_boxscores.items():
            if game_info['gameId'] == game_id_key:  # Match found
                # Iterate through each player position in the lineup, excluding non-ID columns
                for position, player_id in lineup.items():
                    if position not in ['HmTm', 'VisTm']:  # Skip non-ID columns
                        # Determine team type based on position prefix
                        team_type = 'home' if 'Hm' in position else 'away'
                        
                        # Attempt to find the player's stats
                        player_stats = game_info[team_type]['players'].get(player_id, {}).get('stats', 'No stats found')
                        
                        # Store the player's stats
                        if team_type == 'home':
                            game_data['home_players'].append({player_id: player_stats})
                        else:
                            game_data['away_players'].append({player_id: player_stats})
                
                # Once the matching game is processed, break out of the loop
                break
        
        # Add the game's data to the mapped data
        mapped_data[game_id_key] = game_data

    return mapped_data