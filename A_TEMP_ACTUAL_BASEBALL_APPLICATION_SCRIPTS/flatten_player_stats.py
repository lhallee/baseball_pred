def flatten_player_stats(game_id, mapped_data, starting_lineup_dict):
    """
    Flattens and sorts player stats based on the lineup order for a given game, for both home and away players.

    Parameters:
    - game_id (str): The ID of the game.
    - mapped_data (dict): A dictionary containing detailed stats for each player.
    - starting_lineup_dict (dict): A dictionary containing the lineup order for the game.

    Returns:
    - dict: A dictionary containing the flattened and sorted player stats for both home and away players.
    """
    home_players_stats = mapped_data[game_id]['home_players']
    away_players_stats = mapped_data[game_id]['away_players']
    starting_lineup = starting_lineup_dict[game_id]

    # Extract the lineup order for home and away players
    home_lineup_order = [
        starting_lineup['HmStPchID'],  # Starting pitcher
        starting_lineup['HmBat1ID'],   # Batter 1
        starting_lineup['HmBat2ID'],   # Batter 2
        starting_lineup['HmBat3ID'],   # Batter 3
        starting_lineup['HmBat4ID'],   # Batter 4
        starting_lineup['HmBat5ID'],   # Batter 5
        starting_lineup['HmBat6ID'],   # Batter 6
        starting_lineup['HmBat7ID'],   # Batter 7
        starting_lineup['HmBat8ID'],   # Batter 8
        starting_lineup['HmBat9ID']    # Batter 9
    ]

    away_lineup_order = [
        starting_lineup['VisStPchID'],  # Starting pitcher
        starting_lineup['VisBat1ID'],   # Batter 1
        starting_lineup['VisBat2ID'],   # Batter 2
        starting_lineup['VisBat3ID'],   # Batter 3
        starting_lineup['VisBat4ID'],   # Batter 4
        starting_lineup['VisBat5ID'],   # Batter 5
        starting_lineup['VisBat6ID'],   # Batter 6
        starting_lineup['VisBat7ID'],   # Batter 7
        starting_lineup['VisBat8ID'],   # Batter 8
        starting_lineup['VisBat9ID']    # Batter 9
    ]

    # Convert the list of player stats dictionaries into a single dictionary for both home and away
    home_players_stats_dict = {list(player.keys())[0]: list(player.values())[0] for player in home_players_stats}
    away_players_stats_dict = {list(player.keys())[0]: list(player.values())[0] for player in away_players_stats}

    # Flatten and sort the player stats based on the lineup order
    flattened_stats = {'Game_ID': game_id}  # Initialize with the game ID
    for idx, player_id in enumerate(home_lineup_order, start=1):
        if player_id in home_players_stats_dict:
            player_stats = home_players_stats_dict[player_id]
            for stat_type, stats in player_stats.items():
                for stat, value in stats.items():
                    # Construct the key to include player position, stat type, and stat name for home players
                    key = f'home_player{idx}_{stat_type}_{stat}'
                    # Store the stat value directly
                    flattened_stats[key] = value
                    # Additionally, store the player ID at the same level of nesting
                    flattened_stats[f'home_player{idx}_ID'] = player_id

    for idx, player_id in enumerate(away_lineup_order, start=1):
        if player_id in away_players_stats_dict:
            player_stats = away_players_stats_dict[player_id]
            for stat_type, stats in player_stats.items():
                for stat, value in stats.items():
                    # Construct the key to include player position, stat type, and stat name for away players
                    key = f'away_player{idx}_{stat_type}_{stat}'
                    # Store the stat value directly
                    flattened_stats[key] = value
                    # Additionally, store the player ID at the same level of nesting
                    flattened_stats[f'away_player{idx}_ID'] = player_id

    return flattened_stats