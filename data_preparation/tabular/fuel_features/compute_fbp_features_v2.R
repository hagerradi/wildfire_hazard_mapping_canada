############################################################
#### Supports BurnP3+
#### WARNING: lots of AI generated code
# requirments: install R, open R terminal then,
# remotes::install_github("cffdrs/cffdrs_r") # nolint: commented_code_linter.
## For Dev version, this is what i used:
## remotes::install_github("cffdrs/cffdrs_r", ref="dev") # nolint
################
# To run this file: source('data_preparation/feature_processing/fuel_features/compute_fbp_features.R') # nolint
# compute_fbp_features.R: main R script to generate features of fuel, at the raster level # nolint
# raster inputs required:
# elevation: elev.tif
# FBP: fbp.tif
# Weather - ffmc.tif
# Weather - bui.tif
# Weather - ws.tif
# Weather - wd.tif
# Constant values required:
# julien_day: julien day (value from the log file) - median day of a season
############################################################

rm(list = ls())

# ---------------- CONFIG -----------------

input_dir <- "data/fuel_input_data"
base_dir <- "../burnp3plus/hex05/spatial/"

dem_path  <- file.path(base_dir, "hex05_dem.tif")
fuel_path <- file.path(base_dir, "hex05_fbp.tif")

# these are rasters of mean of weather list projected on fire zones
ffmc_rast_path <- file.path(input_dir, "ffmc.tif")
bui_rast_path  <- file.path(input_dir, "bui.tif")
ws_rast_path   <- file.path(input_dir, "ws.tif")
wd_rast_path   <- file.path(input_dir, "wd.tif")

mixedwood_season <- "green"

# ---------------- Libraries -----------------

library(terra)
library(cffdrs)
library(sf)

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
# fuel <- load_raster_fix_rows(fuel_path, expected_rows = 1669)
fuel <- terra::rast(fuel_path)
dem <- rast(dem_path)
fuel_raw <- rast(fuel_path)

cat("Fuel dimensions:\n")
print(dim(fuel))
print(ext(fuel))
print(res(fuel))

cat("DEM dimensions:\n")
print(dim(dem))
print(ext(dem))
print(res(dem))

if (!hasValues(fuel)) stop("Fuel raster has no values.")
if (!hasValues(dem)) stop("DEM raster has no values.")

# -----------------------------------------------------
# Project DEM EXACTLY to fuel
# -----------------------------------------------------

# cat("Aligning DEM to fuel...\n")
# dem <- project(dem_raw, fuel)
# stopifnot(all(dim(fuel) == dim(dem)))

# if (!compareGeom(fuel, dem, stopOnError = FALSE)) {
#   stop("DEM still does not match fuel after projection.")
# }

# ---------------- Julian day raster -----------------

season_day_map <- list(
  green    = 152,  # June 1
  leafless = 121   # May 1
)

julien_day <- season_day_map[[mixedwood_season]]

if (is.null(julien_day)) {
  stop(paste("Unknown mixedwood_season:", mixedwood_season))
}

dj_rast <- fuel
terra::values(dj_rast) <- julien_day
names(dj_rast) <- "Dj"

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

ncells <- terra::ncell(fuel)
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

buieff_rast <- fuel
terra::values(buieff_rast) <- 0
names(buieff_rast) <- "BUIEff"

# Name weather & slope layers correctly
names(ffmc_rast) <- "FFMC"
names(bui_rast)  <- "BUI"
names(ws_rast)   <- "WS"
names(wd_rast)   <- "WD"
names(slope_pct) <- "GS"
names(aspect_deg) <- "Aspect"
names(buieff_rast) <- "BUIEff"

# ---------------- LAT & LONG rasters from fuel + .prj -----------------

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

# 3) Lat / lon rasters
# ---------------- LAT & LONG rasters from fuel CRS -----------------

if (terra::crs(fuel) == "" || is.na(terra::crs(fuel))) {
  stop("Fuel raster has no CRS. Cannot compute LAT/LONG.")
}

make_lonlat_rasters <- function(template_rast) {
  cells <- 1:terra::ncell(template_rast)

  # Cell-center coordinates in the fuel raster CRS
  xy <- terra::xyFromCell(template_rast, cells)

  pts <- sf::st_as_sf(
    data.frame(cell = cells, x = xy[, 1], y = xy[, 2]),
    coords = c("x", "y"),
    crs = terra::crs(template_rast)
  )

  # FBP expects LAT/LONG in decimal degrees, EPSG:4326
  pts_ll <- sf::st_transform(pts, 4326)
  lonlat <- sf::st_coordinates(pts_ll)

  long_rast <- template_rast
  lat_rast  <- template_rast

  terra::values(long_rast) <- lonlat[, 1]
  terra::values(lat_rast)  <- lonlat[, 2]

  names(long_rast) <- "LONG"
  names(lat_rast)  <- "LAT"

  list(
    LONG = long_rast,
    LAT = lat_rast
  )
}

lonlat_rasters <- make_lonlat_rasters(fuel)

long_rast <- lonlat_rasters$LONG
lat_rast  <- lonlat_rasters$LAT
# ---------------- Fuel lookup -----------------

library(dplyr)
library(stringr)

fuel_lookup_csv <- "../burnp3plus/hex05/tabular/hex05_FuelTypes.csv"

mixedwood_type_for <- function(default = "M-2") {
  if (mixedwood_season == "leafless") {
    return("M-1")
  }

  if (mixedwood_season == "green") {
    return("M-2")
  }

  return(default)
}

extract_pc <- function(description) {
  pc <- stringr::str_match(description, "\\((\\d+)\\s*PC\\)")[, 2]
  ifelse(is.na(pc), NA_real_, as.numeric(pc))
}

normalize_fuel_type <- function(description) {
  description <- toupper(description)

  dplyr::case_when(
    description == "C-1" ~ "C-1",
    description == "C-2" ~ "C-2",
    description == "C-3" ~ "C-3",
    description == "C-4" ~ "C-4",
    description == "C-5" ~ "C-5",
    description == "C-6" ~ "C-6",
    description == "C-7" ~ "C-7",

    description == "D-1" ~ "D-1",
    description == "D-2" ~ "D-1",
    description == "D-1/D-2" ~ "D-1",

    description == "O-1A" ~ "O-1A",
    description == "O-1B" ~ "O-1B",

    description %in% c("NON-FUEL", "NF") ~ "NF",
    description == "WA" ~ "WA",

    stringr::str_detect(description, "^M-1\\s*\\(") ~ "M-1",
    stringr::str_detect(description, "^M-2\\s*\\(") ~ "M-2",
    stringr::str_detect(description, "^M-1/M-2\\s*\\(") ~ mixedwood_type_for("M-2"),

    description == "M-1" ~ "M-1",
    description == "M-2" ~ "M-2",
    description == "M-1/M-2" ~ mixedwood_type_for("M-2"),

    TRUE ~ NA_character_
  )
}

load_fuel_lookup <- function(fuel_lookup_csv) {
  fuel_table <- read.csv(
  fuel_lookup_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

  fuel_table <- fuel_table %>%
    mutate(
      ID = as.character(ID),
      FuelType = normalize_fuel_type(Description),
      PC = extract_pc(Description)
    ) %>%
    filter(!is.na(FuelType)) %>%
    select(ID, FuelType, PC)

  fuel_lookup <- setNames(
    lapply(seq_len(nrow(fuel_table)), function(i) {
      list(
        FuelType = fuel_table$FuelType[i],
        PC = fuel_table$PC[i]
      )
    }),
    fuel_table$ID
  )

  fuel_lookup
}

fuel_lookup <- load_fuel_lookup(fuel_lookup_csv)
fuel_vals <- terra::values(fuel, mat = FALSE)
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

for (r_name in c("ffmc_rast", "bui_rast", "ws_rast", "wd_rast")) {
  r <- get(r_name)

  if (!compareGeom(fuel, r, stopOnError = FALSE)) {
    stop(paste(r_name, "does not match fuel geometry."))
  }
}


# Build the full input SpatRaster for fbpRaster
# Option 1
# fbp_input_rast <- c(
#   fuel_rast,
#   lat_rast,
#   long_rast,
#   ffmc_rast,
#   bui_rast,
#   buieff_rast,
#   ws_rast,
#   wd_rast,
#   slope_pct,
#   dj_rast,
#   aspect_deg,
#   pc_rast
# )
# Option 2
# fbp_input_rast <- c(
#   fuel_rast,
#   lat_rast,
#   long_rast,
#   ffmc_rast,
#   buieff_rast,
#   ws_rast,
#   slope_pct,
#   dj_rast,
#   aspect_deg)
# Option 3
fbp_input_rast <- c(
  fuel_rast,
  lat_rast,
  long_rast,
  buieff_rast
)
# Option 4
# fbp_input_rast <- c(
#   fuel_rast
# )

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
lat_vals <- terra::values(lat_rast, mat = FALSE)
long_vals <- terra::values(long_rast, mat = FALSE)
aspect_vals <- terra::values(aspect_deg, mat = FALSE)

valid_cell <- !is.na(FuelType) &
              !is.na(lat_vals) &
              !is.na(long_vals) &
              !is.na(ffmc_vals) &
              !is.na(bui_vals) &
              !is.na(ws_vals) &
              !is.na(wd_vals) &
              !is.na(slope_vals) &
              !is.na(aspect_vals)

# Do the same for other outputs if you want consistency:
for (v in c("ROS", "HFI")) {
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
  out_path <- file.path(input_dir, paste0(v, ".tif"))

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

# cat("Comparing fbp() vs fbpRaster() for", length(idx_test), "pixels\n")

# wd_vals   <- values(wd_rast)   # add this once, near other *_vals
# aspect_vals <- values(aspect_deg)  # numeric vector length = ncells

# for (i in idx_test) {
#   pixel_df <- terra::extract(fbp_input_rast, i, ID = FALSE)

#   res_df <- fbp(pixel_df, output = "Primary")

#   ros_df <- res_df$ROS
#   sfc_df <- res_df$SFC

#   ros_rast <- terra::values(res_rast[["ROS"]], mat = FALSE)[i]
#   sfc_rast <- terra::values(res_rast[["SFC"]], mat = FALSE)[i]

#   cat("Pixel i =", i, "\n")
#   print(pixel_df)

#   cat(
#     "i =", i,
#     "| FuelType =", pixel_df$FuelType,
#     "\n",
#     "  fbp()      ROS =", ros_df, "  SFC =", sfc_df, "\n",
#     "  fbpRaster  ROS =", ros_rast, "  SFC =", sfc_rast, "\n\n"
#   )
# }

#TODO: fbp and fbp-raster outputs don't exactly match at the pixel level.
