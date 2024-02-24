import pandas as pd

def map_stats_to_lineup(mapped_data, starting_lineup_df):
    # Prepare a list to collect all game stats
    all_game_stats = []

    # Iterate through each game in mapped_data
    for game_id, game_info in mapped_data.items():
        # Find the corresponding starting lineup row
        lineup_row = starting_lineup_df[starting_lineup_df['Game_ID'] == game_id]

        if not lineup_row.empty:
            # Initialize a dictionary to hold the stats for this game
            game_stats = {'Game_ID': game_id}

            # Process both home and away teams
            for team_type in ['home', 'away']:
                team_prefix = 'Hm' if team_type == 'home' else 'Vis'
                
                # Process each player in the team
                for player_id, player_info in game_info[f'{team_type}_players'].items():
                    # Find the player's position in the lineup
                    position = None
                    for col in lineup_row.columns:
                        if lineup_row.iloc[0][col] == player_id:
                            position = col
                            break
                    
                    # If the player's position is found, map their stats
                    if position:
                        for stat_type, stats in player_info.items():
                            for stat, value in stats.items():
                                # Construct a unique key for each stat
                                stat_key = f"{position}_{stat_type}_{stat}"
                                game_stats[stat_key] = value
            
            # Add the game's stats to the list
            all_game_stats.append(game_stats)

    # Convert the list of game stats to a DataFrame
    return pd.DataFrame(all_game_stats)
# import pandas as pd

# def map_stats_to_lineup(mapped_data, starting_lineup_df):
#     # Prepare a list to collect all game stats
#     all_game_stats = []

#     # Iterate through each game in starting_lineup_df
#     for _, lineup_row in starting_lineup_df.iterrows():
#         game_id = lineup_row['Game_ID']
#         game_info = mapped_data.get(game_id, {})

#         # Initialize a dictionary to hold the stats for this game
#         game_stats = {'Game_ID': game_id}

#         # Process each player in the lineup
#         for col in lineup_row.index:
#             player_id = lineup_row[col]
#             position = col

#             # Initialize player stats with player ID, ensuring it's always included
#             game_stats[f"{position}_ID"] = player_id

#             # Check if the player has stats in mapped_data
#             team_type = 'home' if 'Hm' in position else 'away'
#             player_info = game_info.get(f'{team_type}_players', {}).get(player_id, {})

#             # Map their stats, if any
#             for stat_type, stats in player_info.items():
#                 for stat, value in stats.items():
#                     # Construct a unique key for each stat
#                     stat_key = f"{position}_{stat_type}_{stat}"
#                     game_stats[stat_key] = value

#         # Add the game's stats to the list
#         all_game_stats.append(game_stats)

#     # Convert the list of game stats to a DataFrame
#     return pd.DataFrame(all_game_stats)