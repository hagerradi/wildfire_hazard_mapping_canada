############################################################
# run_fbp_raster.R  — UPDATED VERSION
############################################################

rm(list = ls())

# ---------------- CONFIG -----------------

input_dir <- "data/fuel_data_temporary"

dem_path  <- file.path(input_dir, "elev.asc")
fuel_path <- file.path(input_dir, "fbp.asc")

# these are rasters of mean of weather list projected on fire zones
ffmc_rast_path <- file.path(input_dir, "ffmc.asc")
bui_rast_path  <- file.path(input_dir, "bui.asc")
ws_rast_path   <- file.path(input_dir, "ws.asc")

# this is from the log file
LAT_const    <- 56.257371
LONG_const   <- -115.114533
Dj_const     <- 152
Aspect_const <- 0

mixedwood_season <- "green"

ros_output_path <- file.path(input_dir, "ROS.asc")

# ---------------- Libraries -----------------

library(terra)
library(cffdrs)

cat("cffdrs version:", as.character(packageVersion("cffdrs")), "\n\n")

# ---------------- Load rasters -----------------

fuel <- rast(fuel_path)
dem_raw <- rast(dem_path)

if (!hasValues(fuel)) stop("Fuel raster has no values.")
if (!hasValues(dem_raw)) stop("DEM raster has no values.")

# -----------------------------------------------------
# Align DEM EXACTLY to fuel (with robust handling)
# -----------------------------------------------------

cat("Aligning DEM to fuel...\n")

dem <- tryCatch(
  {
    # First, try simple crop + resample
    dem_crop <- crop(dem_raw, fuel)
    resample(dem_crop, fuel, method = "bilinear")
  },
  error = function(e) {
    cat("Crop failed:", conditionMessage(e), "\n")
    cat("Attempting to project DEM to fuel CRS...\n")
    # Project DEM to fuel CRS, then resample
    dem_proj <- project(dem_raw, fuel)
    resample(dem_proj, fuel, method = "bilinear")
  }
)

if (!compareGeom(fuel, dem, stopOnError = FALSE)) {
  stop("DEM still does not match fuel after resampling/projection.")
}

# ---------------- Compute slope -----------------

slope_deg <- terrain(dem, v = "slope", unit = "degrees")
slope_pct <- tan(slope_deg * pi / 180) * 100

fuel_vals  <- values(fuel)
slope_vals <- values(slope_pct)

# ---------------- Check lengths -----------------

ncells <- length(fuel_vals)
if (is.na(ncells) || ncells == 0) stop("Fuel raster failed to load or is empty.")

if (length(slope_vals) != ncells) {
  stop("Slope and fuel lengths differ unexpectedly.")
}

# ---------------- Weather setup -----------------
ffmc_rast <- rast(ffmc_rast_path)
bui_rast  <- rast(bui_rast_path)
ws_rast   <- rast(ws_rast_path)

ffmc_rast <- resample(ffmc_rast, fuel)
bui_rast  <- resample(bui_rast, fuel)
ws_rast   <- resample(ws_rast, fuel)

ffmc_vals <- values(ffmc_rast)
bui_vals  <- values(bui_rast)
ws_vals   <- values(ws_rast)



# ---------------- Fuel lookup -----------------

mixedwood_type_for <- function(default) {
  if (mixedwood_season == "leafless") "M-1"
  else if (mixedwood_season == "green") "M-2"
  else default
}

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
  "12" = list(FuelType = "D-2", PC = NA),       # D-2 Green Aspen (BUI thresholding handled elsewhere if needed)
  "13" = list(FuelType = "D-1", PC = NA),       # D-1/D-2 Aspen -> map to D-1 for FBP

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
  "425" = list(FuelType = "M-1",                  PC = 25),  # M-1 leafless (25% conifer)
  "525" = list(FuelType = "M-2",                  PC = 25),  # M-2 green (25% conifer)

  "625" = list(FuelType = mixedwood_type_for("M-2"), PC = 25),  # M-1/M-2 (25% conifer)
  "635" = list(FuelType = mixedwood_type_for("M-2"), PC = 35),  # M-1/M-2 (35% conifer)
  "650" = list(FuelType = mixedwood_type_for("M-2"), PC = 50),  # M-1/M-2 (50% conifer)
  "665" = list(FuelType = mixedwood_type_for("M-2"), PC = 65)   # M-1/M-2 (65% conifer)
)


# ---------------- Map fuel to FBP -----------------

FuelType <- rep(NA_character_, ncells)
PC       <- rep(NA_real_,      ncells)

unique_codes <- unique(fuel_vals)

for (code in unique_codes) {
  key <- as.character(code)
  if (!key %in% names(fuel_lookup)) next
  idx <- which(fuel_vals == code)
  FuelType[idx] <- fuel_lookup[[key]]$FuelType
  if (!is.na(fuel_lookup[[key]]$PC)) {
    PC[idx] <- fuel_lookup[[key]]$PC
  }
}

# Default PC for mixedwood if still missing
missing_pc <- which(FuelType %in% c("M-1", "M-2") & is.na(PC))
if (length(missing_pc) > 0) {
  PC[missing_pc] <- 50
}

# ---------------- Build FBP dataframe -----------------

fbp_input <- data.frame(
  FuelType = FuelType,
  LAT      = rep(LAT_const,   ncells),
  LONG     = rep(LONG_const,  ncells),
  FFMC     = ffmc_vals,
  BUI      = bui_vals,
  WS       = ws_vals,
  GS       = slope_vals,
  Dj       = rep(Dj_const,    ncells),
  Aspect   = rep(Aspect_const, ncells),
  PC       = PC
)

valid <- !is.na(fbp_input$FuelType)

if (!any(valid)) {
  stop("No valid FuelType values found. Check fuel codes and mapping.")
}

# ---------------- Run FBP -----------------

cat("Running FBP...\n")

res <- fbp(fbp_input[valid, ], output = "Primary")
names(res)
# ---------------- Build ROS raster -----------------

ros_vals <- rep(NA_real_, ncells)
ros_vals[valid] <- res$ROS

ros_rast <- fuel
values(ros_rast) <- ros_vals

# Use a numeric NAflag so ASCII is GDAL-friendly
writeRaster(ros_rast, ros_output_path, overwrite = TRUE, NAflag = -9999)

cat("ROS saved to:", ros_output_path, "\n")
cat("Done.\n")
############################################################
