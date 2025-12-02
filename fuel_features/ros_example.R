############################################################
# ROS / FBP example for cffdrs 1.9.0
# - Uses the documented API: fbp(input, output = "Primary")
# - No internal FBP object, no fire_behaviour_prediction()
############################################################

# 0. Clean environment -------------------------------------
rm(list = ls())

# 1. Load cffdrs (install if missing) ----------------------
if (!requireNamespace("cffdrs", quietly = TRUE)) {
  install.packages("cffdrs")
}
library(cffdrs)

cat("\n=== cffdrs version:", as.character(packageVersion("cffdrs")), "===\n\n")

# 2. Build a realistic input row ---------------------------
# Column names MUST match the docs (case-insensitive, but let’s be exact):
# Required: FuelType, LAT, LONG, FFMC, BUI, WS, GS, Dj, Aspect
# Optional but useful: PC, PDF, cc, etc.

fbp_input <- data.frame(
  FuelType = "C-1",   # "C-1","C-2","C-3","C-4","C-5","C-6","C-7", "D-1","M-1","M-2", "O-1a","O-1b", "S-1","S-2","S-3","NF","W", "U" # nolint # nolint: line_length_linter.
  LAT      = 56.257371,    # latitude (Canada-ish)
  LONG     = -115.114533,  # longitude
  FFMC     = 90,      # Fine Fuel Moisture Code - available in weather_list
  BUI      = 60,      # Build-Up Index - available in weather_list
  WS       = 20,      # Wind speed (km/h at 10m) - available in weather_list
  GS       = 0,       # Ground slope (%) – NOT grass stage here
  Dj       = 200,     # Julian day (1–365) # It has to be changed per season say, center of the season # nolint: line_length_linter.
  Aspect   = 0,        # Aspect (deg) # leave as 0
  PC = 65             # Percent Conifer, between 0 and 100.
)

cat("Input to fbp():\n")
print(fbp_input)
cat("\n")

# 3. Run FBP ------------------------------------------------
# Primary = 8 primary outputs (ROS, HFI, etc.)
result <- fbp(input = fbp_input, output = "Primary")

cat("=============== FBP PRIMARY RESULTS ===============\n")
print(result)

# 4. Extract key outputs (if present) ----------------------
safe_get <- function(x, name) if (name %in% names(x)) x[[name]] else NA

ROS <- safe_get(result, "ROS")  # m/min
HFI <- safe_get(result, "HFI")  # kW/m
CFB <- safe_get(result, "CFB")
FI  <- safe_get(result, "FI")
SFC <- safe_get(result, "SFC")
TFC <- safe_get(result, "TFC")

cat("\nKey outputs:\n")
cat("  • Rate of Spread (ROS): ", ROS, " m/min\n", sep = "")
cat("  • Head Fire Intensity (HFI): ", HFI, " kW/m\n", sep = "")
cat("  • Crown Fraction Burned (CFB): ", CFB, "\n", sep = "")
cat("  • Fire Intensity (FI): ", FI, " kW/m\n", sep = "")
cat("  • Surface Fuel Consumption (SFC): ", SFC, " kg/m^2\n", sep = "")
cat("  • Total Fuel Consumption (TFC): ", TFC, " kg/m^2\n\n", sep = "")
cat("===================================================\n")


# fuel_map <- list(
#   '1'   = list(FuelType="C-1"),
#   '2'   = list(FuelType="C-2"),
#   '3'   = list(FuelType="C-3"),
#   '4'   = list(FuelType="C-4"),
#   '7'   = list(FuelType="C-7"),

#   '11'  = list(FuelType="D-1"),
#   '12'  = list(FuelType="D-1"),
#   '13'  = list(FuelType="D-1"),

#   '31'  = list(FuelType="O-1A"),   # or O-1B if you classify as matted

#   '101' = list(FuelType="NF"),
#   '102' = list(FuelType="W"),
#   '106' = list(FuelType="U"),

#   '425' = list(FuelType="M-1", PC=25),
#   '525' = list(FuelType="M-2", PC=25),
#   '635' = list(FuelType="M-1", PC=35),  # or M-2, pick seasonal
#   '650' = list(FuelType="M-1", PC=50), # or M-2, pick seasonal
#   '665' = list(FuelType="M-2", PC=65) # or M-2, pick seasonal
# ) 