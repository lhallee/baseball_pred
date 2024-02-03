# Install devtools if you haven't already
if (!require(devtools)) install.packages("devtools")

# Install baseballr from GitHub
devtools::install_github("BillPetti/baseballr")

# Load the baseballr package
library(baseballr)

# Retrieve the entire Chadwick player dataset
all_players <- try(chadwick_player_lu())

# Get the current year
current_year <- as.numeric(format(Sys.Date(), "%Y"))

# Filter for players with MLB experience and who played in the last year
mlb_players <- all_players[!is.na(all_players$key_mlbam) & 
                           all_players$mlb_played_first <= current_year & 
                           all_players$mlb_played_last >= (current_year - 1), ]

# Extract the MLB player IDs as strings
mlb_player_ids <- as.vector(mlb_players[!is.na(mlb_players$key_fangraphs), "key_fangraphs"])
mlb_player_ids <- as.character(mlb_player_ids)

# Assuming mlb_player is a vector like this:
# mlb_player <- c("1 2 3", "4 5 6", "7 8 9")
# Split each element of the vector
split_list <- strsplit(mlb_player_ids, " ")
# Unlist the split_list to create a single vector
split_vector <- unlist(split_list)

split_vector <- gsub("c\\(|\\)|,|\\n", "", split_vector)
# Convert the character vector to numeric
# numeric_vector <- as.numeric(split_vector)
library(dplyr)
print(split_vector)
all_data_batter <- tibble()  # Initialize an empty tibble for batters
all_data_pitcher <- tibble() # Initialize an empty tibble for pitchers
player_id <- split_vector
current_year <- as.numeric(format(Sys.Date(), "%Y"))  # Get the current year
print(player_id)
for (id in player_id) {
  print(paste("Processing player ID:", id))  # Debugging line to show current player ID
  
  for (year in (current_year-17):current_year) {
    # Attempt to retrieve batter data for the year
    batter_data <- tryCatch({
      baseballr::fg_batter_game_logs(playerid = id, year = as.character(year))
    }, error = function(e) {
      NULL  # Return NULL if an error occurs
    })
    
    # Attempt to retrieve pitcher data for the year
    pitcher_data <- tryCatch({
      baseballr::fg_pitcher_game_logs(playerid = id, year = as.character(year))
    }, error = function(e) {
      NULL  # Return NULL if an error occurs
    })
    
    # Append the data to the respective tibble if it's not NULL
    if (!is.null(batter_data)) {
      all_data_batter <- bind_rows(all_data_batter, batter_data)
    }
    if (!is.null(pitcher_data)) {
      all_data_pitcher <- bind_rows(all_data_pitcher, pitcher_data)
    }
  }
}

# Write the combined tibble for batters to a CSV file
write.csv(all_data_batter, file = "mlb_batter_game_logs_2007_2023.csv", row.names = FALSE)
# Write the combined tibble for MLB pitchers to a different CSV file
write.csv(all_data_pitcher, file = "mlb_pitcher_game_logs_2007_2023.csv", row.names = FALSE)

# Print the number of players processed for each type
# print(paste("Total batters processed:", nrow(all_data_batter)))
# print(paste("Total pitchers processed:", nrow(all_data_pitcher)))
# for (id in player_id) {
#   print(id)}
# print(results)

