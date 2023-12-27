# Load the Lahman package
library(Lahman)

# Install and load the retrosheet package
library(retrosheet)
# Load the Lahman package
game_logs <- getRetrosheet("event", 2000:2023)
# Import the datasets
data("AllstarFull")
data("Appearances")
data("AwardsManagers")
data("AwardsPlayers")
data("AwardsShareManagers")
data("AwardsSharePlayers")
data("Batting")
data("BattingPost")
data("CollegePlaying")
data("Fielding")
data("FieldingOF")
data("FieldingOFsplit")
data("FieldingPost")
data("HallOfFame")
data("HomeGames")
data("Managers")
data("ManagersHalf")
data("Parks")
data("People")
data("Pitching")
data("PitchingPost")
data("Salaries")
data("Schools")
data("SeriesPost")
data("Teams")
data("TeamsFranchises")
data("TeamsHalf")
per_game_data=getRetrosheet()

people_data <- People
pitching_data <- Pitching

# Extracting MLB fielding data
fielding_data <- Fielding

# Extracting MLB batting data
batting_data <- Batting
college_data <- CollegePlaying
# Assuming you have these dataframes: people_data, pitching_data, fielding_data, batting_data, per_game_data

# Create directories if they don't exist
dir.create("Lahman_MLB_People", showWarnings = FALSE)
dir.create("Lahman_MLB_Pitching", showWarnings = FALSE)
dir.create("Lahman_MLB_Fielding", showWarnings = FALSE)
dir.create("Lahman_MLB_Batting", showWarnings = FALSE)
dir.create("Lahman_MLB_per_game_data", showWarnings = FALSE)
dir.create("Lahman_CollegePlaying", showWarnings = FALSE)


# Assuming you have these dataframes: AllstarFull, Appearances, AwardsManagers, AwardsPlayers, AwardsShareManagers, AwardsSharePlayers, BattingPost, CollegePlaying, FieldingOF, FieldingOFsplit, FieldingPost, HallOfFame, HomeGames, Managers, ManagersHalf, Parks, PitchingPost, Salaries, Schools, SeriesPost, Teams, TeamsFranchises, TeamsHalf

# Create directories if they don't exist
dir.create("Lahman_AllstarFull", showWarnings = FALSE)
dir.create("Lahman_Appearances", showWarnings = FALSE)
dir.create("Lahman_AwardsManagers", showWarnings = FALSE)
dir.create("Lahman_AwardsPlayers", showWarnings = FALSE)
dir.create("Lahman_AwardsShareManagers", showWarnings = FALSE)
dir.create("Lahman_AwardsSharePlayers", showWarnings = FALSE)
dir.create("Lahman_BattingPost", showWarnings = FALSE)

dir.create("Lahman_FieldingOF", showWarnings = FALSE)
dir.create("Lahman_FieldingOFsplit", showWarnings = FALSE)
dir.create("Lahman_FieldingPost", showWarnings = FALSE)
dir.create("Lahman_HallOfFame", showWarnings = FALSE)
dir.create("Lahman_HomeGames", showWarnings = FALSE)
dir.create("Lahman_Managers", showWarnings = FALSE)
dir.create("Lahman_ManagersHalf", showWarnings = FALSE)
dir.create("Lahman_Parks", showWarnings = FALSE)
dir.create("Lahman_PitchingPost", showWarnings = FALSE)
dir.create("Lahman_Salaries", showWarnings = FALSE)
dir.create("Lahman_Schools", showWarnings = FALSE)
dir.create("Lahman_SeriesPost", showWarnings = FALSE)
dir.create("Lahman_Teams", showWarnings = FALSE)
dir.create("Lahman_TeamsFranchises", showWarnings = FALSE)
dir.create("Lahman_TeamsHalf", showWarnings = FALSE)

# Write data to CSV files in corresponding directories
write.csv(AllstarFull, file = "Lahman_AllstarFull/Lahman_AllstarFull.csv")
write.csv(Appearances, file = "Lahman_Appearances/Lahman_Appearances.csv")
write.csv(AwardsManagers, file = "Lahman_AwardsManagers/Lahman_AwardsManagers.csv")
write.csv(AwardsPlayers, file = "Lahman_AwardsPlayers/Lahman_AwardsPlayers.csv")
write.csv(AwardsShareManagers, file = "Lahman_AwardsShareManagers/Lahman_AwardsShareManagers.csv")
write.csv(AwardsSharePlayers, file = "Lahman_AwardsSharePlayers/Lahman_AwardsSharePlayers.csv")
write.csv(BattingPost, file = "Lahman_BattingPost/Lahman_BattingPost.csv")
write.csv(CollegePlaying, file = "Lahman_CollegePlaying/Lahman_CollegePlaying.csv")
write.csv(FieldingOF, file = "Lahman_FieldingOF/Lahman_FieldingOF.csv")
write.csv(FieldingOFsplit, file = "Lahman_FieldingOFsplit/Lahman_FieldingOFsplit.csv")
write.csv(FieldingPost, file = "Lahman_FieldingPost/Lahman_FieldingPost.csv")
write.csv(HallOfFame, file = "Lahman_HallOfFame/Lahman_HallOfFame.csv")
write.csv(HomeGames, file = "Lahman_HomeGames/Lahman_HomeGames.csv")
write.csv(Managers, file = "Lahman_Managers/Lahman_Managers.csv")
write.csv(ManagersHalf, file = "Lahman_ManagersHalf/Lahman_ManagersHalf.csv")
write.csv(Parks, file = "Lahman_Parks/Lahman_Parks.csv")
write.csv(PitchingPost, file = "Lahman_PitchingPost/Lahman_PitchingPost.csv")
write.csv(Salaries, file = "Lahman_Salaries/Lahman_Salaries.csv")
write.csv(Schools, file = "Lahman_Schools/Lahman_Schools.csv")
write.csv(SeriesPost, file = "Lahman_SeriesPost/Lahman_SeriesPost.csv")
write.csv(Salaries, file = "Lahman_Salaries/Lahman_Salaries.csv")
write.csv(Schools, file = "Lahman_Schools/Lahman_Schools.csv")
write.csv(SeriesPost, file = "Lahman_SeriesPost/Lahman_SeriesPost.csv")
write.csv(Teams, file = "Lahman_Teams/Lahman_Teams.csv")
write.csv(TeamsFranchises, file = "Lahman_TeamsFranchises/Lahman_TeamsFranchises.csv")
write.csv(TeamsHalf, file = "Lahman_TeamsHalf/Lahman_TeamsHalf.csv")
# Write data to CSV files in corresponding directories

# Assuming you have the game logs data in a data frame called `game_logs`
# and the player data in a data frame called `people_data`

# Merge the game logs and player data on the `retroID` column


# Now, each row in `merged_data` represents a player-game pair

# data(LahmanData)