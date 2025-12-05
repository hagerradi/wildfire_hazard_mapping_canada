############################################################
#### WARNING: lots of AI generated code
# requirments: install R, open R terminal then,
# remotes::install_github("cffdrs/cffdrs_r") # nolint: commented_code_linter.
## For Dev version, this is what i used:
## remotes::install_github("cffdrs/cffdrs_r", ref="dev") # nolint
################
# To run this file: source('data_preparation/feature_processing/fuel_features/fbp_features.R') # nolint
# fbp_features.R: main R script to generate features of fuel, at the raster level # nolint
# raster inputs required:
# elevation: elev.asc
# FBP: fbp.asc
# Weather - ffmc.asc
# Weather - bui.asc
# Weather - ws.asc
# Constant values required:
# use_constant_fwi <- FALSE # otherwise, it will use constant values fixed in the script # nolint
# latitude: value from the log file
# longitude: value from the log file
# julien_day: julien day (value from the log file) - median day of a season
############################################################

rm(list = ls())

# ---------------- CONFIG -----------------

input_dir <- "data/fuel_input_data"
base_dir <- "../yan_bp3/hex05/mapped_inputs"

dem_path  <- file.path(base_dir, "elev.asc")
fuel_path <- file.path(base_dir, "fbp.asc")

# manuallty set
use_constant_fwi <- FALSE
# this is from the log file
latitude    <- 56.257371
longitude   <- -115.114533
aspect_const <- 0

# these are rasters of mean of weather list projected on fire zones
ffmc_rast_path <- file.path(input_dir, "ffmc.asc")
bui_rast_path  <- file.path(input_dir, "bui.asc")
ws_rast_path   <- file.path(input_dir, "ws.asc")

mixedwood_season <- "green"

# ---------------- Libraries -----------------

library(terra)
library(cffdrs)

cat("cffdrs version:", as.character(packageVersion("cffdrs")), "\n\n")

# ---------------- Load rasters -----------------
load_raster_fix_rows <- function(path, expected_rows) {
  r <- rast(path)
  res_y <- res(r)[2]
  e <- ext(r)
  e$ymax <- e$ymin + expected_rows * res_y
  crop(r, e)
}
# MANUAL FIX: FUEL is loaded with extra row by terra.
fuel <- load_raster_fix_rows(fuel_path, expected_rows = 1669)

dem_raw <- rast(dem_path)

if (!hasValues(fuel)) stop("Fuel raster has no values.")
if (!hasValues(dem_raw)) stop("DEM raster has no values.")

# -----------------------------------------------------
# Project DEM EXACTLY to fuel
# -----------------------------------------------------

cat("Aligning DEM to fuel...\n")
dem <- project(dem_raw, fuel)
stopifnot(all(dim(fuel) == dim(dem)))

if (!compareGeom(fuel, dem, stopOnError = FALSE)) {
  stop("DEM still does not match fuel after projection.")
}

# ---------------- Compute slope -----------------

slope_deg <- terrain(dem, v = "slope", unit = "degrees")
slope_pct <- tan(slope_deg * pi / 180) * 100

fuel_vals  <- values(fuel)
slope_vals <- values(slope_pct)

# ---------------- Check lengths -----------------

ncells <- length(fuel_vals)
if (is.na(ncells) || ncells == 0) stop("Fuel raster failed to load or is empty.") # nolint

if (length(slope_vals) != ncells) {
  stop("Slope and fuel lengths differ unexpectedly.")
}

if (use_constant_fwi) {
  # manuallty set
  ffmc_const <- 90
  bui_const  <- 60
  ws_const   <- 20

  ffmc_vals <- rep(ffmc_const, ncells)
  bui_vals  <- rep(bui_const, ncells)
  ws_vals   <- rep(ws_const, ncells)
}else {
  # ---------------- Weather rasters -----------------
  ffmc_rast <- rast(ffmc_rast_path)
  bui_rast  <- rast(bui_rast_path)
  ws_rast   <- rast(ws_rast_path)

  ffmc_rast <- project(ffmc_rast, fuel)
  bui_rast  <- project(bui_rast, fuel)
  ws_rast   <- project(ws_rast, fuel)

  stopifnot(all(dim(fuel) == dim(ffmc_rast)))

  if (!compareGeom(fuel, ffmc_rast, stopOnError = FALSE)) {
    stop("FFMC still does not match fuel after resampling/projection.")
  }

  ffmc_vals <- values(ffmc_rast)
  bui_vals  <- values(bui_rast)
  ws_vals   <- values(ws_rast)
}


# ---------------- Fuel lookup -----------------
mixedwood_type_for <- function(default) {
  if (mixedwood_season == "leafless") "M-1"
  else if (mixedwood_season == "green") "M-2"
  else default
}
season_day_map <- list(
  green = 152,      # June 1
  leafless = 121    # May 1
)
julien_day <- season_day_map[[mixedwood_season]]

fuel_lookup <- list(
  # ---------------- Conifer (C) ----------------
  "1"  = list(FuelType = "C-1", PC = NA),  # C-1 Spruce-Lichen Woodland
  "2"  = list(FuelType = "C-2", PC = NA),  # C-2 Boreal Spruce
  "3"  = list(FuelType = "C-3", PC = NA),  # C-3 Mature Jack / Lodgepole Pine
  "4"  = list(FuelType = "C-4", PC = NA),  # C-4 Immature Jack / Lodgepole Pine
  "5"  = list(FuelType = "C-5", PC = NA),  # C-5 Red and White Pine
  "6"  = list(FuelType = "C-6", PC = NA),  # C-6 Conifer Plantation
  "7"  = list(FuelType = "C-7", PC = NA),  # C-7 Ponderosa Pine / Douglas-fir

  # ---------------- Deciduous (D) ----------------
  "11" = list(FuelType = "D-1", PC = NA),       # D-1 Leafless Aspen
  "12" = list(FuelType = "D-2", PC = NA),       # D-2 Green Aspen (BUI thresholding handled elsewhere if needed) # nolint
  "13" = list(FuelType = "D-1", PC = NA),       # D-1/D-2 Aspen -> map to D-1 for FBP # nolint

  # ---------------- Grass (O) ----------------
  "31" = list(FuelType = "O-1A", PC = NA),      # O-1a Matted Grass
  "32" = list(FuelType = "O-1B", PC = NA),      # O-1b Standing Grass

  # ---------------- Non-fuel ----------------
  "101" = list(FuelType = NA, PC = NA),         # Non-fuel
  "102" = list(FuelType = NA, PC = NA),         # Water
  "106" = list(FuelType = NA, PC = NA),         # Urban

  # ---------------- Mixedwood (M) ----------------
  # Note: codes that are explicitly M-1 or M-2 keep that,
  # codes that are M-1/M-2 use mixedwood_type_for() so
  # they switch between M-1 / M-2 based on mixedwood_season.
  "425" = list(FuelType = "M-1",                  PC = 25),  # M-1 leafless (25% conifer) # nolint
  "525" = list(FuelType = "M-2",                  PC = 25),  # M-2 green (25% conifer) # nolint

  "625" = list(FuelType = mixedwood_type_for("M-2"), PC = 25),  # M-1/M-2 (25% conifer) # nolint
  "635" = list(FuelType = mixedwood_type_for("M-2"), PC = 35),  # M-1/M-2 (35% conifer) # nolint
  "650" = list(FuelType = mixedwood_type_for("M-2"), PC = 50),  # M-1/M-2 (50% conifer) # nolint
  "665" = list(FuelType = mixedwood_type_for("M-2"), PC = 65)   # M-1/M-2 (65% conifer) # nolint
)


# ---------------- Map fuel to FBP -----------------
FuelType <- rep(NA_character_, ncells) # nolint
PC       <- rep(NA_real_,      ncells) # nolint

unique_codes <- unique(fuel_vals)

for (code in unique_codes) {
  key <- as.character(code)
  if (!key %in% names(fuel_lookup)) next
  idx <- which(fuel_vals == code)
  FuelType[idx] <- fuel_lookup[[key]]$FuelType # nolint: object_name_linter.
  if (!is.na(fuel_lookup[[key]]$PC)) {
    PC[idx] <- fuel_lookup[[key]]$PC # nolint
  }
}

# Default PC for mixedwood if still missing
missing_pc <- which(FuelType %in% c("M-1", "M-2") & is.na(PC))
if (length(missing_pc) > 0) {
  PC[missing_pc] <- 50 # nolint
}

# ---------------- Build FBP dataframe -----------------

fbp_input <- data.frame(
  FuelType = FuelType,
  LAT      = rep(latitude,   ncells),
  LONG     = rep(longitude,  ncells),
  FFMC     = ffmc_vals,
  BUI      = bui_vals,
  WS       = ws_vals,
  GS       = slope_vals,
  Dj       = rep(julien_day,    ncells),
  Aspect   = rep(aspect_const, ncells),
  PC       = PC
)

valid <- !is.na(fbp_input$FuelType)

if (!any(valid)) {
  stop("No valid FuelType values found. Check fuel codes and mapping.")
}

# ---------------- Run FBP -----------------

cat("Running FBP...\n")

res <- fbp(fbp_input[valid, ], output = "Primary")
cat(names(res))
# ---------------- Build ROS raster -----------------
fbp_vars <- c("ROS", "HFI", "CFB", "SFC", "TFC")  # pick what you want

for (v in fbp_vars) {
  if (!v %in% names(res)) {
    cat("Skipping", v, "- not found in FBP output.\n")
    next
  }

  vals <- rep(NA_real_, ncells)
  vals[valid] <- res[[v]]

  r <- fuel
  values(r) <- vals

  out_path <- file.path(input_dir, paste0(v, ".asc"))
  writeRaster(r, out_path, overwrite = TRUE, NAflag = -9999)
  cat("Wrote", v, "to", out_path, "\n")
}
cat("ROS summary:\n")
print(summary(res$ROS))

cat("Weather summaries:\n")
summary(ffmc_vals)
summary(bui_vals)
summary(ws_vals)

cat("Slope summary:\n")
summary(slope_vals)

cat("Fuel types in run:\n")
table(fbp_input$FuelType)
