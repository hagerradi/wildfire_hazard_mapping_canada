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
# Weather - wd.asc
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

# manually set
use_constant_fwi <- FALSE
# this is from the log file
latitude    <- 56.257371
longitude   <- -115.114533
aspect_const <- 0

# these are rasters of mean of weather list projected on fire zones
ffmc_rast_path <- file.path(input_dir, "ffmc.asc")
bui_rast_path  <- file.path(input_dir, "bui.asc")
ws_rast_path   <- file.path(input_dir, "ws.asc")
wd_rast_path   <- file.path(input_dir, "wd.asc")

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

# Calculate aspect (Terra outputs degrees: 0=North, 90=East, matching cffdrs)
aspect_deg <- terrain(dem, v = "aspect", unit = "degrees")
names(aspect_deg) <- "Aspect"

# Handle flat terrain (Slope ~ 0 => Aspect is irrelevant/undefined)
# We can default Aspect to 0 where slope is 0 to avoid NAs
aspect_deg[is.na(aspect_deg)] <- 0
# ---------------- Check lengths -----------------

ncells <- length(fuel_vals)
if (is.na(ncells) || ncells == 0) stop("Fuel raster failed to load or is empty.") # nolint

if (length(slope_vals) != ncells) {
  stop("Slope and fuel lengths differ unexpectedly.")
}

if (use_constant_fwi) {
  # manually set
  ffmc_const <- 90
  bui_const  <- 60
  ws_const   <- 20
  wd_const   <- 0

  ffmc_rast <- fuel
  bui_rast  <- fuel
  ws_rast   <- fuel
  wd_rast   <- fuel

  values(ffmc_rast) <- ffmc_const
  values(bui_rast)  <- bui_const
  values(ws_rast)   <- ws_const
  values(wd_rast)   <- wd_const
} else {
  # ---------------- Weather rasters -----------------
  ffmc_rast <- rast(ffmc_rast_path)
  bui_rast  <- rast(bui_rast_path)
  ws_rast   <- rast(ws_rast_path)
  wd_rast   <- rast(wd_rast_path)

  ffmc_rast <- project(ffmc_rast, fuel)
  bui_rast  <- project(bui_rast, fuel)
  ws_rast   <- project(ws_rast, fuel)
  wd_rast   <- project(wd_rast, fuel)

  stopifnot(all(dim(fuel) == dim(ffmc_rast)))

  if (!compareGeom(fuel, ffmc_rast, stopOnError = FALSE)) {
    stop("FFMC still does not match fuel after resampling/projection.")
  }
}

# Just for summaries later
ffmc_vals <- values(ffmc_rast)
bui_vals  <- values(bui_rast)
ws_vals   <- values(ws_rast)
wd_vals   <- values(wd_rast)

# ---------------- Fuel lookup -----------------

mixedwood_type_for <- function(default) {
  if (mixedwood_season == "leafless") "M-1"
  else if (mixedwood_season == "green") "M-2"
  else default
}
season_day_map <- list(
  green    = 152,  # June 1
  leafless = 121   # May 1
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
  "101" = list(FuelType = "NF", PC = NA),         # Non-fuel
  "102" = list(FuelType = "WA", PC = NA),         # Water
  "106" = list(FuelType = "NF", PC = NA),         # Urban

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

# ---------------- Map fuel to FBP (per cell) -----------------
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

# For all non-mixedwood fuels, PC should not be NA
PC[is.na(PC)] <- 0

if (all(is.na(FuelType))) {
  stop("No valid FuelType values found. Check fuel codes and mapping.")
}

# FuelType raster as factor with character levels (preferred by fbpRaster)
fuel_rast <- fuel
values(fuel_rast) <- NA_character_
values(fuel_rast)[!is.na(FuelType)] <- FuelType[!is.na(FuelType)]
fuel_rast <- as.factor(fuel_rast)
names(fuel_rast) <- "FuelType"
fuel_vals   <- values(fuel_rast)

# LAT & LONG rasters (constant over grid)
lat_rast <- fuel
values(lat_rast) <- latitude
names(lat_rast)  <- "LAT"

long_rast <- fuel
values(long_rast) <- longitude
names(long_rast)  <- "LONG"

# Julian day raster
dj_rast <- fuel
values(dj_rast) <- julien_day
names(dj_rast)  <- "Dj"

# PC raster (Percent Conifer – only used for M-1/M-2)
pc_rast <- fuel
values(pc_rast) <- PC
names(pc_rast)  <- "PC"

# Name weather & slope layers correctly
names(ffmc_rast) <- "FFMC"
names(bui_rast)  <- "BUI"
names(ws_rast)   <- "WS"
names(wd_rast)   <- "WD"
names(slope_pct) <- "GS"

# Build the full input SpatRaster for fbpRaster
fbp_input_rast <- c(
  fuel_rast,
  lat_rast,
  long_rast,
  ffmc_rast,
  bui_rast,
  ws_rast,
  wd_rast,
  slope_pct,
  dj_rast,
  aspect_deg,
  pc_rast
)

# Sanity check
if (any(is.na(values(fuel_rast)))) {
  cat("Warning: some cells have NA FuelType codes; outputs will be NA there.\n")
}

cat("Running fbpRaster on", ncell(fbp_input_rast), "cells...\n")

# ---------------- Run FBP on rasters -----------------

fbp_vars <- c("ROS", "HFI", "CFB", "SFC", "TFC")  # pick what you want

# Use Primary outputs; you can switch to "All" if needed
res_rast <- fbpRaster(
  input  = fbp_input_rast,
  output = "Primary"
  # you can add: select = fbp_vars, cores = 4, m = 1000, etc.
)

cat("fbpRaster outputs:\n")
print(names(res_rast))

# =======================
# Build a validity mask
# =======================

valid_cell <- !is.na(FuelType) &              # have a mapped fuel type
              !is.na(ffmc_vals) & # nolint
              !is.na(bui_vals)  &
              !is.na(ws_vals)   &
              !is.na(wd_vals) &
              !is.na(slope_vals)

# Do the same for other outputs if you want consistency:
for (v in c("ROS", "HFI", "CFB", "SFC", "TFC")) {
  if (!v %in% names(res_rast)) next
  vals <- values(res_rast[[v]])
  vals[!valid_cell] <- NA
  values(res_rast[[v]]) <- vals
}

# ---------------- Write output rasters -----------------

for (v in fbp_vars) {
  if (!v %in% names(res_rast)) {
    cat("Skipping", v, "- not found in fbpRaster output.\n")
    next
  }

  out_r   <- res_rast[[v]]
  out_path <- file.path(input_dir, paste0(v, ".asc"))

  writeRaster(out_r, out_path, overwrite = TRUE, NAflag = -9999)
  cat("Wrote", v, "to", out_path, "\n")
}

# ---------------- Summaries -----------------

if ("ROS" %in% names(res_rast)) {
  cat("ROS summary:\n")
  print(summary(values(res_rast[["ROS"]])))
}

cat("Weather summaries:\n")
print(summary(ffmc_vals))
print(summary(bui_vals))
print(summary(ws_vals))
print(summary(wd_vals))

cat("Slope summary:\n")
print(summary(slope_vals))

cat("PC summary:\n")
print(summary(values(pc_rast)))

cat("Fuel types in run (string):\n")
print(table(FuelType, useNA = "ifany"))


# Choose some pixels with valid fuels and weather
idx_test <- which(
  !is.na(FuelType) &
  FuelType %in% c("C-1","C-2","C-3","C-4","C-7","D-1","M-2") &
  !is.na(ffmc_vals) &
  !is.na(bui_vals) &
  !is.na(ws_vals) &
  !is.na(slope_vals)
)

idx_test <- idx_test[seq_len(min(5, length(idx_test)))]

cat("Comparing fbp() vs fbpRaster() for", length(idx_test), "pixels\n")

wd_vals   <- values(wd_rast)   # add this once, near other *_vals

for (i in idx_test) {
  df_i <- data.frame(
    FuelType = FuelType[i],
    LAT      = latitude,       # or values(lat_rast)[i]
    LONG     = longitude,      # or values(long_rast)[i]
    FFMC     = ffmc_vals[i],
    BUI      = bui_vals[i],
    WS       = ws_vals[i],
    WD       = wd_vals[i],     # ← ensure same WD as raster
    GS       = slope_vals[i],
    Dj       = julien_day,
    Aspect   = aspect_deg[i],
    PC       = PC[i]
  )

  res_df  <- fbp(df_i, output = "Primary")
  ros_df  <- res_df$ROS
  sfc_df  <- res_df$SFC

  ros_rast <- values(res_rast[["ROS"]])[i]
  sfc_rast <- values(res_rast[["SFC"]])[i]

  cat("Pixel i =", i, "\n")
  print(df_i)
  cat(
    "i =", i,
    "| FuelType =", FuelType[i],
    "\n",
    "  fbp()      ROS =", ros_df, "  SFC =", sfc_df, "\n",
    "  fbpRaster  ROS =", ros_rast, "  SFC =", sfc_rast, "\n\n"
  )
}

#TODO: ROS has valid values for both Water, Urban and NoFuel.
#TODO: fbp and fbp-raster outputs don't match at the pixel level.