# ============================================================
# FBP ISI vs ROSi curves - safer clean version
# open R terminal and run: source("data_preparation/tabular/fuel_features/compute_vector_values_BC.R")
# ============================================================

cat("Starting script...\n\n")

library(cffdrs)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)

out_dir <- file.path(getwd(), "fbp_outputs")

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat("Output directory:\n")
cat(out_dir, "\n\n")

# ------------------------------------------------------------
# ISI range
# ------------------------------------------------------------

isi_values <- seq(0, 70, by = 1)

# ------------------------------------------------------------
# Curve specs
# ------------------------------------------------------------

curve_specs <- tibble::tribble(
  ~CurveLabel,       ~FuelType, ~PC, ~PDF, ~cc, ~GFL,
  "C-1",             "C-1",     NA,  NA,   NA,  NA,
  "C-2",             "C-2",     NA,  NA,   NA,  NA,
  "C-3",             "C-3",     NA,  NA,   NA,  NA,
  "C-4",             "C-4",     NA,  NA,   NA,  NA,
  "C-5",             "C-5",     NA,  NA,   NA,  NA,
  "C-7",             "C-7",     NA,  NA,   NA,  NA,
  "D-1",             "D-1",     NA,  NA,   NA,  NA,
  "D-2",             "D-2",     NA,  NA,   NA,  NA,

  "M-1 / 50%C",      "M-1",     50,  NA,   NA,  NA,
  "M-1 / 75%C",      "M-1",     75,  NA,   NA,  NA,

  "M-2 / 50%C",      "M-2",     50,  NA,   NA,  NA,
  "M-2 / 75%C",      "M-2",     75,  NA,   NA,  NA,

  "M-3 / 65%DF",     "M-3",     NA,  65,   NA,  NA,
  "M-3 / 100%DF",    "M-3",     NA,  100,  NA,  NA,

  "S-1",             "S-1",     NA,  NA,   NA,  NA,
  "S-2",             "S-2",     NA,  NA,   NA,  NA,
  "S-3",             "S-3",     NA,  NA,   NA,  NA,

  "O-1a 90%c",       "O-1a",    NA,  NA,   90,  0.35,
  "O-1b 90%c",       "O-1b",    NA,  NA,   90,  0.35
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

    PC = ifelse(is.na(PC), 50, PC),
    PDF = ifelse(is.na(PDF), 35, PDF),
    cc = ifelse(is.na(cc), 80, cc),
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

    LAT = 44.6648,
    LONG = -63.5762,
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
  select(row_id, CurveLabel, FuelType, ISI) %>%
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
# Styling
# ------------------------------------------------------------

curve_colors <- c(
  "C-1" = "#7F7F7F",      # gray
  "C-2" = "#000000",      # black
  "C-3" = "#7A3B00",      # brown
  "C-4" = "#9A4A00",      # brown-orange
  "C-5" = "#A895C6",      # light purple
  "C-7" = "#C43B3B",      # muted red

  "D-1" = "#39C000",      # bright green
  "D-2" = "#39C000",      # same green

  "M-1 / 50%C" = "#1E88E5",   # dashed blue
  "M-1 / 75%C" = "#1E88E5",   # solid blue

  "M-2 / 50%C" = "#1696E8",   # blue with green points
  "M-2 / 75%C" = "#1696E8",   # same blue

  "M-3 / 65%DF" = "#FF0000",  # strong red
  "M-3 / 100%DF" = "#FF2A2A", # slightly lighter red

  "S-1" = "#8B4513",      # brown dotted
  "S-2" = "#000000",      # black dotted
  "S-3" = "#6E97D8",      # light steel blue dotted

  "O-1a 90%c" = "#D7DB4A", # yellow-green dashed
  "O-1b 90%c" = "#D7DB4A"  # same yellow-green solid
)

curve_linetypes <- c(
  "C-1" = "solid",
  "C-2" = "solid",
  "C-3" = "solid",
  "C-4" = "longdash",
  "C-5" = "solid",
  "C-7" = "solid",
  "D-1" = "solid",
  "M-1 / 50%C" = "longdash",
  "M-1 / 75%C" = "solid",
  "M-2 / 50%C" = "dotted",
  "M-2 / 75%C" = "solid",
  "M-3 / 65%DF" = "solid",
  "M-3 / 100%DF" = "solid",
  "S-1" = "dotted",
  "S-2" = "dotted",
  "S-3" = "dotted",
  "O-1a 90%c" = "longdash",
  "O-1b 90%c" = "solid"
)

curve_sizes <- c(
  "C-1" = 0.9,
  "C-2" = 0.9,
  "C-3" = 1.0,
  "C-4" = 0.9,
  "C-5" = 0.9,
  "C-7" = 0.9,
  "D-1" = 0.9,
  "M-1 / 50%C" = 0.9,
  "M-1 / 75%C" = 0.9,
  "M-2 / 50%C" = 0.9,
  "M-2 / 75%C" = 0.9,
  "M-3 / 65%DF" = 1.2,
  "M-3 / 100%DF" = 0.8,
  "S-1" = 0.9,
  "S-2" = 0.9,
  "S-3" = 0.9,
  "O-1a 90%c" = 0.9,
  "O-1b 90%c" = 1.0
)

point_df <- plot_df %>%
  filter(CurveLabel %in% c("M-2 / 50%C", "M-2 / 75%C")) %>%
  filter(ISI %% 2 == 0)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_df_clean <- plot_df %>%
  filter(ISI > 0) %>%
  bind_rows(
    plot_df %>%
      filter(ISI == 0) %>%
      mutate(ROS = 0)
  ) %>%
  arrange(CurveLabel, ISI)

p <- ggplot(plot_df_clean, aes(x = ISI, y = ROS, group = CurveLabel)) +
  geom_line(
    aes(color = CurveLabel, linetype = CurveLabel, linewidth = CurveLabel),
    lineend = "butt"
  ) +
  geom_point(
    data = point_df,
    color = "#00D83A",
    size = 2.1,
    show.legend = FALSE
  ) +
  scale_color_manual(values = curve_colors, breaks = curve_order) +
  scale_linetype_manual(values = curve_linetypes, breaks = curve_order) +
  scale_linewidth_manual(values = curve_sizes, breaks = curve_order) +
  scale_x_continuous(
    limits = c(0, 70),
    breaks = seq(0, 70, by = 10),
    minor_breaks = seq(0, 70, by = 2),
    expand = c(0, 0)
    ) +
  scale_y_continuous(
    limits = c(0, 175),
    breaks = seq(0, 175, by = 25),
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
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(color = "steelblue", face = "bold", size = 12, hjust = 0),
    plot.subtitle = element_text(color = "steelblue", face = "bold", size = 9, hjust = 0),
    axis.title = element_text(face = "bold", color = "black"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "gray40", linewidth = 0.4),
    axis.ticks = element_line(color = "gray40"),
    axis.ticks.length = unit(0.16, "cm"),
    legend.position = "right",
    legend.text = element_text(size = 10),
    legend.key.width = unit(1.2, "cm"),
    legend.key.height = unit(0.45, "cm"),
    panel.border = element_rect(color = "gray65", fill = NA, linewidth = 0.5),
    plot.margin = margin(15, 20, 15, 15)
  )

print(p)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

png_path <- file.path(out_dir, "fbp_isi_rosi_curves_clean.png")
csv_path <- file.path(out_dir, "fbp_isi_rosi_curves_clean.csv")

ggsave(
  filename = png_path,
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

write.csv(
  plot_df,
  file = csv_path,
  row.names = FALSE
)

cat("Saved plot to:\n")
cat(png_path, "\n\n")

cat("Saved CSV to:\n")
cat(csv_path, "\n\n")

cat("Done.\n")
