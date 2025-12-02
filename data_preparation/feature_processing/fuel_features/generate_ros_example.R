############################################################
# An example on how to use cffdrs library
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

ROS <- safe_get(result, "ROS") # nolint
HFI <- safe_get(result, "HFI")  # nolint: object_name_linter.
CFB <- safe_get(result, "CFB") # nolint: object_name_linter.
FI  <- safe_get(result, "FI") # nolint: object_name_linter, object_name_linter.
SFC <- safe_get(result, "SFC") # nolint: object_name_linter.
TFC <- safe_get(result, "TFC") # nolint: object_name_linter.

cat("\nKey outputs:\n")
cat("  • Rate of Spread (ROS): ", ROS, " m/min\n", sep = "")
cat("  • Head Fire Intensity (HFI): ", HFI, " kW/m\n", sep = "")
cat("  • Crown Fraction Burned (CFB): ", CFB, "\n", sep = "")
cat("  • Fire Intensity (FI): ", FI, " kW/m\n", sep = "")
cat("  • Surface Fuel Consumption (SFC): ", SFC, " kg/m^2\n", sep = "")
cat("  • Total Fuel Consumption (TFC): ", TFC, " kg/m^2\n\n", sep = "")
cat("===================================================\n")
