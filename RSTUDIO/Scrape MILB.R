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
print(sum(!is.na(all_players$key_bbref_minors)))
milb_players <- all_players[all_players$key_bbref_minors != "" & 
                            all_players$mlb_played_first >= (current_year - 1), ]

# Extract the MiLB player IDs as strings
# Assuming milb_player_ids is a data frame or matrix column, convert it to a vector
milb_player_ids <- as.vector(milb_players[!is.na(milb_players$key_fangraphs), "key_fangraphs"])
# Loop through the vector and print each element on a new line
milb_player_ids <- as.character(milb_player_ids)

# Assuming mlb_player is a vector like this:
# mlb_player <- c("1 2 3", "4 5 6", "7 8 9")
# Split each element of the vector
split_list <- strsplit(milb_player_ids, " ")
# Unlist the split_list to create a single vector
split_vector <- unlist(split_list)

split_vector <- gsub("c\\(|\\)|,|\\n", "", split_vector)
# Convert the character vector to numeric
# numeric_vector <- as.numeric(split_vector)
library(dplyr)

all_data_batter <- tibble()  # Initialize an empty tibble for batters
all_data_pitcher <- tibble() # Initialize an empty tibble for pitchers
player_id <- split_vector
current_year <- as.numeric(format(Sys.Date(), "%Y"))  # Get the current year

for (id in player_id) {
  # print(paste("Processing player ID:", id))  # Debugging line to show current player ID
  
  for (year in (current_year-17):current_year) {
    # Attempt to retrieve batter data for the year
    batter_data <- tryCatch({
      baseballr::fg_milb_batter_game_logs(id, as.character(year))
    }, error = function(e) {
      NULL  # Return NULL if an error occurs
    })
    
    # Attempt to retrieve pitcher data for the year
    pitcher_data <- tryCatch({
      baseballr::fg_milb_pitcher_game_logs(id, as.character(year))
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
write.csv(all_data_batter, file = "milb_batter_game_logs_2007_2023.csv", row.names = FALSE)
# Write the combined tibble for pitchers to a different CSV file
write.csv(all_data_pitcher, file = "milb_pitcher_game_logs_2007_2023.csv", row.names = FALSE)

# Print the number of players processed for each type
# print(paste("Total batters processed:", nrow(all_data_batter)))
# print(paste("Total pitchers processed:", nrow(all_data_pitcher)))
for (id in player_id) {
  print(id)}
print(results)
# # Function to get batter game logs
# get_batter_logs <- function(player_id, year) {
#   tryCatch({
#     baseballr::fg_milb_batter_game_logs(player_id, year)
#   }, error = function(e) {
#     NULL  # Return NULL if an error occurs
#   })
# }

# # Function to get pitcher game logs
# get_pitcher_logs <- function(player_id, year) {
#   tryCatch({
#     baseballr::fg_milb_pitcher_game_logs(player_id, year)
#   }, error = function(e) {
#     NULL  # Return NULL if an error occurs
#   })
# }

# get_player_game_logs <- function(player_id, year) {
#   # Try to get batter logs first
#   batter_data <- get_batter_logs(player_id, year)
#   if (!is.null(batter_data)) {
#     return(batter_data)
#   }
  
#   # If batter logs are null, try to get pitcher logs
#   pitcher_data <- get_pitcher_logs(player_id, year)
#   if (!is.null(pitcher_data)) {
#     return(pitcher_data)
#   }
  
#   # If both are null, return NULL
#   return(NULL)
# }


# # Function to get game logs, trying batter first, then pitcher
# # Function to get game logs, trying batter first, then pitcher
# results <- list()

# for (id in split_vector) {
#   for (yr in (current_year-1):current_year) {
#     data <- get_player_game_logs(id, yr)
#     if (!is.null(data)) {
#       results[[paste(id, yr, sep = "_")]] <- data
#     }
#   }
# }

# # Combine all data frames in the results list into one data frame
# all_data <- do.call(rbind, results)

# # Export the combined data frame to a CSV file
# write.csv(all_data, file = "player_game_logs.csv", row.names = FALSE)
# # Example usage

# # player_id <- split_vector[1]  # Replace with the actual player ID
# # year <- current_year-20 # Replace with the actual year
# # Function to get game logs, trying batter first, then pitcher
