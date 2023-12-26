library(baseballr)

# Get today's date
today <- Sys.Date()

# Calculate the date 3 years ago
start_date = "2020-01-01"

# Get batting data
batting_data <- statcast_search_batters(start_date = start_date, end_date = today)

# Get pitching data
pitching_data <- statcast_search_pitchers(start_date = start_date, end_date = today)

