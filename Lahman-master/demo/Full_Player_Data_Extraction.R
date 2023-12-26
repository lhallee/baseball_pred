# Load the Lahman package
library(Lahman)

# Install and load the retrosheet package
library(retrosheet)

# Initialize an empty list to store the data frames
data_list <- list()

# Loop over the years of interest
for(year in 2000:2023) {
  # Get the data for the current year
  data <- per_game_data("game", year)
  
  # Store the data frame in the list
  data_list[[as.character(year)]] <- data
}

# Combine all the data frames into one
per_game_data <- do.call(rbind, data_list)

people_data <- People
pitching_data <- Pitching

# Extracting MLB fielding data
fielding_data <- Fielding

# Extracting MLB batting data
batting_data <- Batting
# write.csv(master_data, file = "Lahman_MLB_People/Lahman_MLB_People.csv")
# write.csv(pitching_data, file = "Lahman_MLB_Pitching/Lahman_MLB_pitching.csv")
# write.csv(fielding_data, file = "Lahman_MLB_Fielding/Lahman_MLB_fielding.csv")
# write.csv(batting_data, file = "Lahman_MLB_Batting/Lahman_MLB_batting.csv")
write.csv(per_game_data, file = "Lahman_MLB_per_game_data/Lahman_MLB_per_game_data.csv")

# Assuming you have the game logs data in a data frame called `game_logs`
# and the player data in a data frame called `people_data`

# Merge the game logs and player data on the `retroID` column


# Now, each row in `merged_data` represents a player-game pair

# data(LahmanData)