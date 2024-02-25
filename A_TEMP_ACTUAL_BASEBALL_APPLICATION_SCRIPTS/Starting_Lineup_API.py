

# import requests
# import pandas as pd

# def Starting_Lineup_API(gamePk: int):
#     """
#     Fetches and formats the lineup for a given MLB game identified by its gamePk, with players' IDs as rows and their batting order positions as columns.
    
#     Parameters:
#     - gamePk (int): The unique identifier for the MLB game.
    
#     Returns:
#     - A tuple of DataFrames: (df_home_lineup, df_away_lineup) with modified structure.
#     """
#     # URL for the MLB API request
#     url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={gamePk}&language=en&hydrate=story,xrefId,lineups,broadcasts(all),probablePitcher(note),game(content(media(epg)),tickets)&useLatestGames=true&fields=dates,games,teams,probablePitcher,note,id,dates,games,broadcasts,type,name,homeAway,language,isNational,callSign,mediaState,mediaStateCode,availableForStreaming,freeGame,mediaId,dates,games,game,tickets,ticketType,ticketLinks,dates,games,content,media,epg,dates,games,lineups,homePlayers,awayPlayers,useName,lastName,primaryPosition,abbreviation,dates,games,xrefIds,xrefId,xrefType,story"

#     # Fetch data from the API
#     response = requests.get(url)
#     data = response.json()

#     # Extract the game data
#     game_data = data['dates'][0]['games'][0]

#     # Extract probable pitchers for home and away teams
#     home_pitcher = game_data.get('teams', {}).get('home', {}).get('probablePitcher', {})
#     away_pitcher = game_data.get('teams', {}).get('away', {}).get('probablePitcher', {})

#     # Extracting home and away team lineups
#     home_lineup = game_data.get('lineups', {}).get('homePlayers', [])
#     away_lineup = game_data.get('lineups', {}).get('awayPlayers', [])

#     # Function to format lineup data, including the pitcher
#     def format_lineup(pitcher, players):
#         lineup = {'HmStPitcherID' if pitcher.get('homeAway') == 'home' else 'VisStPitcherID': [pitcher.get('id', '')]}
#         for i, player in enumerate(players, start=1):
#             key = f"HmBat{i}ID" if player.get('homeAway') == 'home' else f"VisBat{i}ID"
#             if key not in lineup:
#                 lineup[key] = []
#             lineup[key].append(player.get('id', ''))
#         return lineup

#     # Format the lineups including the pitchers
#     formatted_home_lineup = format_lineup(home_pitcher, home_lineup)
#     formatted_away_lineup = format_lineup(away_pitcher, away_lineup)

#     # Convert to pandas DataFrame for easier viewing/manipulation, transpose for desired format
#     df_home_lineup = pd.DataFrame(formatted_home_lineup, index=[gamePk]).transpose()
#     df_away_lineup = pd.DataFrame(formatted_away_lineup, index=[gamePk]).transpose()

#     return df_home_lineup, df_away_lineup


import requests
import pandas as pd

def Starting_Lineup_API(gamePk: int):
    """
    Fetches and formats the lineup for a given MLB game identified by its gamePk, with players' IDs as rows and their batting order positions as columns.
    
    Parameters:
    - gamePk (int): The unique identifier for the MLB game.
    
    Returns:
    - A tuple of DataFrames: (df_home_lineup, df_away_lineup) with modified structure.
    """
    # URL for the MLB API request
    url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={gamePk}&language=en&hydrate=story,xrefId,lineups,broadcasts(all),probablePitcher(note),game(content(media(epg)),tickets)&useLatestGames=true&fields=dates,games,teams,probablePitcher,note,id,dates,games,broadcasts,type,name,homeAway,language,isNational,callSign,mediaState,mediaStateCode,availableForStreaming,freeGame,mediaId,dates,games,game,tickets,ticketType,ticketLinks,dates,games,content,media,epg,dates,games,lineups,homePlayers,awayPlayers,useName,lastName,primaryPosition,abbreviation,dates,games,xrefIds,xrefId,xrefType,story"

    # Fetch data from the API
    response = requests.get(url)
    data = response.json()

    # Extract the game data
    game_data = data['dates'][0]['games'][0]

    # Extract probable pitchers for home and away teams
    home_pitcher = game_data.get('teams', {}).get('home', {}).get('probablePitcher', {})
    away_pitcher = game_data.get('teams', {}).get('away', {}).get('probablePitcher', {})

    # Extracting home and away team lineups
    home_lineup = game_data.get('lineups', {}).get('homePlayers', [])
    away_lineup = game_data.get('lineups', {}).get('awayPlayers', [])

    # Function to format lineup data, including the pitcher
    def format_lineup(pitcher, players, homeAway):
        lineup = {'HmStPchID' if homeAway == 'home' else 'VisStPchID': [pitcher.get('id', '')]}
        for i, player in enumerate(players, start=1):
            key = f"HmBat{i}ID" if homeAway == 'home' else f"VisBat{i}ID"
            if key not in lineup:
                lineup[key] = []
            lineup[key].append(player.get('id', ''))
        return lineup

    # Format the lineups including the pitchers
    formatted_home_lineup = format_lineup(home_pitcher, home_lineup, 'home')
    formatted_away_lineup = format_lineup(away_pitcher, away_lineup, 'away')

    # Convert to pandas DataFrame for easier viewing/manipulation, transpose for desired format
    # df_home_lineup = pd.DataFrame(formatted_home_lineup, index=[gamePk]).transpose()
    # df_away_lineup = pd.DataFrame(formatted_away_lineup, index=[gamePk]).transpose()
# Assuming formatted_home_lineup and formatted_away_lineup are already defined as shown in your snippet

    # Convert to pandas DataFrame for easier viewing/manipulation, transpose for desired format
    df_home_lineup = pd.DataFrame(formatted_home_lineup, index=[gamePk]).transpose()
    df_away_lineup = pd.DataFrame(formatted_away_lineup, index=[gamePk]).transpose()

    # Combine the home and away lineups into one DataFrame
    # Combine the home and away lineups into one DataFrame in the same row
    # Append df_away_lineup onto the end of df_home_lineup
    appended_lineup = pd.concat([df_home_lineup, df_away_lineup], axis=0)

    # Transpose the resulting DataFrame
    transposed_lineup = appended_lineup.transpose()
    transposed_lineup.set_index(transposed_lineup.columns[0],inplace=True)

    return transposed_lineup