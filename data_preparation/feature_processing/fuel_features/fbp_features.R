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
# julien_day: julien day (value from the log file) - median day of a season
############################################################

rm(list = ls())

# ---------------- CONFIG -----------------

input_dir <- "data/fuel_input_data"
base_dir <- "../yan_bp3/hex05/mapped_inputs"

dem_path  <- file.path(base_dir, "elev.asc")
fuel_path <- file.path(base_dir, "fbp.asc")

# this is from the log file
latitude    <- 56.257371
longitude   <- -115.114533

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
fuel_raw <- rast(fuel_path)
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

# ---------------- Compute slope and aspect -----------------

slope_deg <- terrain(dem, v = "slope", unit = "degrees")
slope_pct <- tan(slope_deg * pi / 180) * 100

slope_vals <- values(slope_pct)

# Calculate aspect (Terra outputs degrees: 0=North, 90=East, matching cffdrs)
aspect_deg <- terrain(dem, v = "aspect", unit = "degrees")

# Handle flat terrain (Slope ~ 0 => Aspect is irrelevant/undefined)
# We can default Aspect to 0 where slope is 0 to avoid NAs
aspect_deg[is.na(aspect_deg)] <- 0
# ---------------- Check lengths -----------------

ncells <- length(values(fuel))
if (is.na(ncells) || ncells == 0) stop("Fuel raster failed to load or is empty.") # nolint

if (length(slope_vals) != ncells) {
  stop("Slope and fuel lengths differ unexpectedly.")
}

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

# Name weather & slope layers correctly
names(ffmc_rast) <- "FFMC"
names(bui_rast)  <- "BUI"
names(ws_rast)   <- "WS"
names(wd_rast)   <- "WD"
names(slope_pct) <- "GS"
names(aspect_deg) <- "Aspect"

# ---------------- LAT & LONG rasters from fuel + .prj -----------------
library(sf)

if (is.na(terra::crs(fuel))) {
  stop("Fuel raster has no CRS. Check that fbp.prj is present and readable.")
}

# 1) Get cell-center coordinates in the fuel CRS
xy <- terra::crds(fuel, df = TRUE)   # columns: x, y. Row order == cell index order # nolint

# 2) Turn them into sf points with the fuel CRS
pts <- sf::st_as_sf(
  xy,
  coords = c("x", "y"),
  crs = terra::crs(fuel)  # WKT string from your PROJCRS
)

# 3) Reproject to WGS84 (EPSG:4326)
pts_ll <- sf::st_transform(pts, 4326)

# 4) Extract lon/lat coordinates
lonlat <- sf::st_coordinates(pts_ll)   # matrix [ncell x 2], cols: X=lon, Y=lat

# 5) Build LONG and LAT SpatRasters aligned with 'fuel'
long_rast <- fuel
lat_rast  <- fuel

terra::values(long_rast) <- lonlat[, 1]  # longitude
terra::values(lat_rast)  <- lonlat[, 2]  # latitude

names(long_rast) <- "LONG"
names(lat_rast)  <- "LAT"

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

# Julian day raster
dj_rast <- fuel
values(dj_rast) <- julien_day
names(dj_rast)  <- "Dj"

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
  "12" = list(FuelType = "D-1", PC = NA),       # D-2 Green Aspen (BUI thresholding handled elsewhere if needed) # nolint
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
fuel_vals <- values(fuel)

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

# PC raster (Percent Conifer – only used for M-1/M-2)
pc_rast <- fuel
values(pc_rast) <- PC
names(pc_rast)  <- "PC"

# FuelType raster as factor with character levels (preferred by fbpRaster)
fuel_rast <- fuel
values(fuel_rast) <- NA_character_
values(fuel_rast)[!is.na(FuelType)] <- FuelType[!is.na(FuelType)]
fuel_rast <- as.factor(fuel_rast)
names(fuel_rast) <- "FuelType"

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

fbp_vars <- c("ROS", "HFI", "CFB", "SFC", "TFC")

# Use Primary outputs; you can switch to "All" if needed
res_rast <- fbpRaster(
  input  = fbp_input_rast,
  output = "Primary",
  select = fbp_vars,
)

cat("fbpRaster outputs:\n")
print(names(res_rast))

# =======================
# Build a validity mask
# =======================
ffmc_vals <- values(ffmc_rast)
bui_vals <- values(bui_rast)
ws_vals <- values(ws_rast)
wd_vals <- values(wd_rast)

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

cat("Fuel summary:\n")
print(summary(values(fuel_rast)))

cat("Weather summaries:\n")
print(summary(values(ffmc_rast)))
print(summary(values(bui_rast)))
print(summary(values(ws_rast)))
print(summary(values(wd_rast)))

cat("Slope summary:\n")
print(summary(slope_vals))

cat("Aspect summary:\n")
print(summary(values(aspect_deg)))

cat("PC summary:\n")
print(summary(values(pc_rast)))

cat("DJ summary:\n")
print(summary(values(dj_rast)))


# Choose some pixels with valid fuels and weather
idx_test <- which(
  !is.na(FuelType) &
  !is.na(ffmc_vals) &
  !is.na(bui_vals) &
  !is.na(ws_vals) &
  !is.na(slope_vals)
)

idx_test <- idx_test[seq_len(min(5, length(idx_test)))]

cat("Comparing fbp() vs fbpRaster() for", length(idx_test), "pixels\n")

wd_vals   <- values(wd_rast)   # add this once, near other *_vals
aspect_vals <- values(aspect_deg)  # numeric vector length = ncells

for (i in idx_test) {
  pixel_df <- extract(fbp_input_rast, i)   # 'i' is a cell index
  pixel_df <- pixel_df[1, ]               # drop row index column

  pixel_df$FuelType <- "D-1"

  res_df   <- fbp(pixel_df, output = "Primary")
  ros_df  <- res_df$ROS
  sfc_df  <- res_df$SFC

  ros_rast <- values(res_rast[["ROS"]])[i]
  sfc_rast <- values(res_rast[["SFC"]])[i]

  cat("Pixel i =", i, "\n")
  print(pixel_df)
  cat(
    "i =", i,
    "| FuelType =", FuelType[i],
    "\n",
    "  fbp()      ROS =", ros_df, "  SFC =", sfc_df, "\n",
    "  fbpRaster  ROS =", ros_rast, "  SFC =", sfc_rast, "\n\n"
  )
}

#TODO: fbp and fbp-raster outputs don't match at the pixel level.
#TODO: ROS has valid values for water and non-fuel
