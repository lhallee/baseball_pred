def fetch_mlb_api(date):
    import statsapi
    """
    Fetches the starting lineups for all MLB games on a specific date.
    
    Args:
    - date (str): The date for which to fetch the starting lineups, in 'YYYY-MM-DD' format.
    
    Returns:
    - dict: A dictionary with game IDs as keys and starting lineups as values.
    """
    # Fetch the schedule for the given date
    raw_schedule = statsapi.schedule(start_date=date, end_date=date)
    
    # Convert the list-based schedule to a dictionary with game IDs as keys
    schedule = {game['game_id']: game for game in raw_schedule}
    
    # Initialize an empty dictionary to store the boxscores
    boxscores = {}
    
    # Iterate through each game in the schedule
    for game_id in schedule:
        # Fetch the boxscore data for each game using its game_id
        boxscores[game_id] = statsapi.boxscore_data(game_id)
    
    return schedule, boxscores