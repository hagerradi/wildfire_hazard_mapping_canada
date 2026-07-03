# ============================================================
# FBP ISI vs ROSi curves
# open R terminal and run: source("data_preparation/tabular/fuel_features/compute_vector_values_national.R")
# ============================================================

cat("Starting script...\n\n")

library(cffdrs)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)

args <- commandArgs(trailingOnly = TRUE)

out_dir <- if (length(args) >= 1) {
  args[1]
} else {
  file.path(getwd(), "fbp_outputs")
}

out_dir <- normalizePath(
  out_dir,
  winslash = "/",
  mustWork = FALSE
)

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}


cat("Output directory:\n")
cat(out_dir, "\n\n")

# ------------------------------------------------------------
# ISI range
# ------------------------------------------------------------

isi_values <- c(1e-6, seq(5, 85, by = 5))

# ------------------------------------------------------------
# Curve specs
# ------------------------------------------------------------

curve_specs <- tibble::tribble(
  ~fbp_code, ~CurveLabel,                 ~FuelType, ~SeasonState, ~PC, ~PDF, ~cc, ~GFL,

  # ------------------------------------------------------------
  # Conifer fuel types
  # ------------------------------------------------------------
  1,         "C-1",                       "C-1",     "direct",     NA,  NA,   NA,  NA,
  2,         "C-2",                       "C-2",     "direct",     NA,  NA,   NA,  NA,
  3,         "C-3",                       "C-3",     "direct",     NA,  NA,   NA,  NA,
  4,         "C-4",                       "C-4",     "direct",     NA,  NA,   NA,  NA,
  5,         "C-5",                       "C-5",     "direct",     NA,  NA,   NA,  NA,
  7,         "C-7",                       "C-7",     "direct",     NA,  NA,   NA,  NA,

  # ------------------------------------------------------------
  # Deciduous fuel types
  # ------------------------------------------------------------
  11,        "D-1 leafless",              "D-1",     "leafless",   NA,  NA,   NA,  NA,
  12,        "D-2 green",                 "D-2",     "green",      NA,  NA,   NA,  NA,

  # Raster class 13 is D-1/D-2, not a direct FBP fuel type.
  # Represent it as two phenology scenarios.
  13,        "D-1/D-2 leafless as D-1",   "D-1",     "leafless",   NA,  NA,   NA,  NA,
  13,        "D-1/D-2 green as D-2",      "D-2",     "green",      NA,  NA,   NA,  NA,

  # ------------------------------------------------------------
  # Grass fuel types
  # cc = percent curing.
  # GFL = grass fuel load in kg/m².
  # 0.35 kg/m² = 3.5 t/ha standard assumption.
  # ------------------------------------------------------------
  31,        "O-1a 90%c, GFL 0.35",       "O-1a",    "direct",     NA,  NA,   90,  0.35,
  32,        "O-1b 90%c, GFL 0.35",       "O-1b",    "direct",     NA,  NA,   90,  0.35,

  # non-burning classes
  101, "Non-fuel", "NF", "nonfuel", NA, NA, NA, NA,
  102, "Water",    "WA", "water",   NA, NA, NA, NA,
  # ------------------------------------------------------------
  # Explicit M-1 / M-2 classes from your rasters
  # PC = percent conifer.
  # ------------------------------------------------------------
  420,       "M-1 20%C leafless",         "M-1",     "leafless",   20,  NA,   NA,  NA,
  450,       "M-1 50%C leafless",         "M-1",     "leafless",   50,  NA,   NA,  NA,

  535,       "M-2 35%C green",            "M-2",     "green",      35,  NA,   NA,  NA,

  # ------------------------------------------------------------
  # Combined M-1/M-2 raster classes.
  # Represent each as two seasonal scenarios.
  # ------------------------------------------------------------
  610,       "M-1/M-2 10%C leafless",     "M-1",     "leafless",   10,  NA,   NA,  NA,
  610,       "M-1/M-2 10%C green",        "M-2",     "green",      10,  NA,   NA,  NA,

  620,       "M-1/M-2 20%C leafless",     "M-1",     "leafless",   20,  NA,   NA,  NA,
  620,       "M-1/M-2 20%C green",        "M-2",     "green",      20,  NA,   NA,  NA,

  635,       "M-1/M-2 35%C leafless",     "M-1",     "leafless",   35,  NA,   NA,  NA,
  635,       "M-1/M-2 35%C green",        "M-2",     "green",      35,  NA,   NA,  NA,

  650,       "M-1/M-2 50%C leafless",     "M-1",     "leafless",   50,  NA,   NA,  NA,
  650,       "M-1/M-2 50%C green",        "M-2",     "green",      50,  NA,   NA,  NA,

  665,       "M-1/M-2 65%C leafless",     "M-1",     "leafless",   65,  NA,   NA,  NA,
  665,       "M-1/M-2 65%C green",        "M-2",     "green",      65,  NA,   NA,  NA
)

curve_specs <- curve_specs %>%
  mutate(
    PC  = as.numeric(PC),
    PDF = as.numeric(PDF),
    cc  = as.numeric(cc),
    GFL = as.numeric(GFL)
  )

curve_order <- curve_specs$CurveLabel

# ------------------------------------------------------------
# Build metadata table for plotting
# ------------------------------------------------------------

meta_df <- curve_specs %>%
  tidyr::crossing(ISI = isi_values) %>%
  mutate(
    row_id = row_number(),
    CurveLabel = factor(CurveLabel, levels = curve_order),

    # Defaults where not explicitly required.
    # For non-M fuel types, PC/PDF are ignored by FBP.
    # For non-grass fuel types, cc/GFL are ignored by FBP.
    PC  = ifelse(is.na(PC), 50, PC),
    PDF = ifelse(is.na(PDF), 35, PDF),
    cc  = ifelse(is.na(cc), 80, cc),
    GFL = ifelse(is.na(GFL), 0.35, GFL)
  )

# ------------------------------------------------------------
# Build clean FBP input table
# ------------------------------------------------------------

fbp_input <- meta_df %>%
  transmute(
    ID = row_id,

    FuelType = as.character(FuelType),

    # ISI supplied directly
    ISI = ISI,
    FFMC = 90,
    BUI = 60,
    WS = 0,
    GS = 0,
    Dj = 180,
    Aspect = 0,
    BUIEff = 0,

    PC = PC,
    PDF = PDF,
    cc = cc,
    GFL = GFL
  )

# ------------------------------------------------------------
# Run FBP
# ------------------------------------------------------------

fbp_out <- cffdrs::fbp(
  fbp_input,
  output = "All",
  m = nrow(fbp_input)
)

# ------------------------------------------------------------
# Join by ID, not row order
# ------------------------------------------------------------
plot_df <- meta_df %>%
  select(row_id, fbp_code, CurveLabel, FuelType, SeasonState, ISI) %>%
  left_join(
    fbp_out %>%
      select(
        ID,
        ROS,
        HFI,
        SFC,
        TFC,
        CFB,
        ISI_fbp = ISI
      ),
    by = c("row_id" = "ID")
  ) %>%
  mutate(
    ROS = tidyr::replace_na(ROS, 0),
    HFI = tidyr::replace_na(HFI, 0)
  ) %>%
  arrange(CurveLabel, ISI)
# ------------------------------------------------------------
# Check for duplicate points
# ------------------------------------------------------------

dup_check <- plot_df %>%
  count(CurveLabel, ISI) %>%
  filter(n > 1)

if (nrow(dup_check) > 0) {
  cat("WARNING: duplicate CurveLabel + ISI rows found:\n")
  print(dup_check)
} else {
  cat("No duplicate CurveLabel + ISI rows found.\n\n")
}

cat("Preview of plot_df:\n")
print(head(plot_df, 20))
cat("\n")

# ------------------------------------------------------------
# Styling: colours for the NEW CurveLabel names
# ------------------------------------------------------------

curve_colors <- c(
  # Conifer fuel types
  "C-1" = "#4D4D4D",
  "C-2" = "#000000",
  "C-3" = "#8B4513",
  "C-4" = "#D55E00",
  "C-5" = "#CC79A7",
  "C-7" = "#E69F00",

  # Deciduous fuel types
  "D-1 leafless" = "#009E73",
  "D-2 green" = "#56B4E9",
  "D-1/D-2 leafless as D-1" = "#0072B2",
  "D-1/D-2 green as D-2" = "#00BFC4",

  # Grass fuel types
  "O-1a 90%c, GFL 0.35" = "#F0E442",
  "O-1b 90%c, GFL 0.35" = "#B79F00",

  # Explicit M-1 / M-2 classes
  "M-1 20%C leafless" = "#A6CEE3",
  "M-1 50%C leafless" = "#1F78B4",
  "M-2 35%C green" = "#33A02C",

  # Combined M-1/M-2 raster classes
  "M-1/M-2 10%C leafless" = "#B2DF8A",
  "M-1/M-2 10%C green" = "#FB9A99",

  "M-1/M-2 20%C leafless" = "#E31A1C",
  "M-1/M-2 20%C green" = "#FDBF6F",

  "M-1/M-2 35%C leafless" = "#FF7F00",
  "M-1/M-2 35%C green" = "#CAB2D6",

  "M-1/M-2 50%C leafless" = "#6A3D9A",
  "M-1/M-2 50%C green" = "#FFFF99",

  "M-1/M-2 65%C leafless" = "#B15928",
  "M-1/M-2 65%C green" = "#999999",

  # Non-burning classes
  "Non-fuel" = "#666666",
  "Water" = "#005AB5"
)

curve_linetypes <- c(
  # Conifer fuel types
  "C-1" = "solid",
  "C-2" = "solid",
  "C-3" = "solid",
  "C-4" = "solid",
  "C-5" = "solid",
  "C-7" = "solid",

  # Deciduous fuel types
  "D-1 leafless" = "solid",
  "D-2 green" = "solid",
  "D-1/D-2 leafless as D-1" = "longdash",
  "D-1/D-2 green as D-2" = "dashed",

  # Grass fuel types
  "O-1a 90%c, GFL 0.35" = "longdash",
  "O-1b 90%c, GFL 0.35" = "solid",

  # Explicit M-1 / M-2 classes
  "M-1 20%C leafless" = "solid",
  "M-1 50%C leafless" = "solid",
  "M-2 35%C green" = "solid",

  # Combined M-1/M-2 raster classes
  "M-1/M-2 10%C leafless" = "solid",
  "M-1/M-2 10%C green" = "dashed",

  "M-1/M-2 20%C leafless" = "solid",
  "M-1/M-2 20%C green" = "dashed",

  "M-1/M-2 35%C leafless" = "solid",
  "M-1/M-2 35%C green" = "dashed",

  "M-1/M-2 50%C leafless" = "solid",
  "M-1/M-2 50%C green" = "dashed",

  "M-1/M-2 65%C leafless" = "solid",
  "M-1/M-2 65%C green" = "dashed",

  "Non-fuel" = "solid",
  "Water" = "solid"
)

curve_sizes <- setNames(
  rep(0.9, length(curve_order)),
  curve_order
)

# Make scenario / comparison curves slightly thinner
curve_sizes[grepl("D-1/D-2", names(curve_sizes))] <- 0.75
curve_sizes[grepl("M-1/M-2", names(curve_sizes))] <- 0.75

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

missing_colors <- setdiff(curve_order, names(curve_colors))
missing_linetypes <- setdiff(curve_order, names(curve_linetypes))
missing_sizes <- setdiff(curve_order, names(curve_sizes))

if (length(missing_colors) > 0) {
  stop("Missing colours for: ", paste(missing_colors, collapse = ", "))
}

if (length(missing_linetypes) > 0) {
  stop("Missing linetypes for: ", paste(missing_linetypes, collapse = ", "))
}

if (length(missing_sizes) > 0) {
  stop("Missing sizes for: ", paste(missing_sizes, collapse = ", "))
}

# ------------------------------------------------------------
# Clean plotting data
# ------------------------------------------------------------
plot_df_clean <- plot_df %>%
  filter(ISI > 0) %>%
  arrange(CurveLabel, ISI)

# Optional green points on green / M-2 curves
point_df <- plot_df_clean %>%
  filter(
    SeasonState == "green" |
      FuelType == "M-2" |
      grepl("green", as.character(CurveLabel))
  ) %>%
  filter(ISI %% 2 == 0)

# ------------------------------------------------------------
# Plot ROS
# ------------------------------------------------------------

p <- ggplot(plot_df_clean, aes(x = ISI, y = ROS, group = CurveLabel)) +
  geom_line(
    aes(
      color = CurveLabel,
      linetype = CurveLabel,
      linewidth = CurveLabel
    ),
    lineend = "butt"
  ) +
  geom_point(
    data = point_df,
    aes(
      x = ISI,
      y = ROS,
      color = CurveLabel
    ),
    size = 1.6,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = curve_colors,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_linetype_manual(
    values = curve_linetypes,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_linewidth_manual(
    values = curve_sizes,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_x_continuous(
    limits = c(0, 85),
    breaks = seq(0, 85, by = 10),
    minor_breaks = seq(0, 85, by = 2),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 190),
    breaks = seq(0, 190, by = 25),
    expand = c(0, 0)
  ) +
  guides(
    color = guide_legend(
      title = NULL,
      ncol = 1,
      override.aes = list(
        linetype = unname(curve_linetypes[curve_order]),
        linewidth = unname(curve_sizes[curve_order])
      )
    ),
    linetype = "none",
    linewidth = "none"
  ) +
  labs(
    x = "Initial Spread Index",
    y = "Rate of Spread, m/min"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(
      color = "steelblue",
      face = "bold",
      size = 12,
      hjust = 0
    ),
    plot.subtitle = element_text(
      color = "steelblue",
      face = "bold",
      size = 9,
      hjust = 0
    ),
    axis.title = element_text(face = "bold", color = "black"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "gray40", linewidth = 0.4),
    axis.ticks = element_line(color = "gray40"),
    axis.ticks.length = unit(0.16, "cm"),
    legend.position = "right",
    legend.text = element_text(size = 9),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.45, "cm"),
    panel.border = element_rect(
      color = "gray65",
      fill = NA,
      linewidth = 0.5
    ),
    plot.margin = margin(15, 20, 15, 15)
  )

print(p)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

png_path <- file.path(out_dir, "fbp_rosi_curves_national_fuel.png")
csv_path <- file.path(out_dir, "fbp_curves_national_fuel.csv")

ggsave(
  filename = png_path,
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

write.csv(
  plot_df_clean,
  file = csv_path,
  row.names = FALSE
)

cat("Saved plot to:\n")
cat(png_path, "\n\n")

cat("Saved CSV to:\n")
cat(csv_path, "\n\n")

# ------------------------------------------------------------
# Plot HFI
# ------------------------------------------------------------

p_hfi <- ggplot(plot_df_clean, aes(x = ISI, y = HFI, group = CurveLabel)) +
  geom_line(
    aes(
      color = CurveLabel,
      linetype = CurveLabel,
      linewidth = CurveLabel
    ),
    lineend = "butt"
  ) +
  geom_point(
    data = point_df,
    aes(
      x = ISI,
      y = HFI,
      color = CurveLabel
    ),
    size = 1.6,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = curve_colors,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_linetype_manual(
    values = curve_linetypes,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_linewidth_manual(
    values = curve_sizes,
    breaks = curve_order,
    limits = curve_order
  ) +
  scale_x_continuous(
    limits = c(0, 85),
    breaks = seq(0, 85, by = 10),
    minor_breaks = seq(0, 85, by = 2),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    limits = c(0, NA)
  ) +
  guides(
    color = guide_legend(
      title = NULL,
      ncol = 1,
      override.aes = list(
        linetype = unname(curve_linetypes[curve_order]),
        linewidth = unname(curve_sizes[curve_order])
      )
    ),
    linetype = "none",
    linewidth = "none"
  ) +
  labs(
    x = "Initial Spread Index",
    y = "Head Fire Intensity, kW/m"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(
      color = "steelblue",
      face = "bold",
      size = 12,
      hjust = 0
    ),
    plot.subtitle = element_text(
      color = "steelblue",
      face = "bold",
      size = 9,
      hjust = 0
    ),
    axis.title = element_text(face = "bold", color = "black"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "gray40", linewidth = 0.4),
    axis.ticks = element_line(color = "gray40"),
    axis.ticks.length = unit(0.16, "cm"),
    legend.position = "right",
    legend.text = element_text(size = 9),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.45, "cm"),
    panel.border = element_rect(
      color = "gray65",
      fill = NA,
      linewidth = 0.5
    ),
    plot.margin = margin(15, 20, 15, 15)
  )

print(p_hfi)

# ------------------------------------------------------------
# Save HFI outputs
# ------------------------------------------------------------

hfi_png_path <- file.path(out_dir, "fbp_hfi_curves_national_fuel.png")

ggsave(
  filename = hfi_png_path,
  plot = p_hfi,
  width = 12,
  height = 8,
  dpi = 300
)

cat("Saved HFI plot to:\n")
cat(hfi_png_path, "\n\n")


cat("Done.\n")
