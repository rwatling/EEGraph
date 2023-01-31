library(ggplot2)

setwd("/home/rwatling/Academics/mtu/masters/programming/EEGraph")
options(scipen = 32)

# Controls
benchmarks = c("google", "higgs", "livejournal", "pokec", "road", "skitter")
variants = c("async_push_td", "async_push_dd", "sync_push_td", "sync_push_dd")
vals <- c(1)
energy_types <- c("regular-energy", "um-energy")

# Classic Files
async_push_dd_files <- list.files("./data/sssp/regular-energy/async_push_dd", pattern='readings', full.names=TRUE)
async_push_td_files <- list.files("./data/sssp/regular-energy/async_push_td", pattern='readings', full.names=TRUE)
sync_push_dd_files <- list.files("./data/sssp/regular-energy/sync_push_dd", pattern='readings', full.names=TRUE)
sync_push_td_files <- list.files("./data/sssp/regular-energy/sync_push_td", pattern='readings', full.names=TRUE)

# UM Files
um_async_push_dd_files <- list.files("./data/sssp/um-energy/async_push_dd", pattern='readings', full.names=TRUE)
um_async_push_td_files <- list.files("./data/sssp/um-energy/async_push_td", pattern='readings', full.names=TRUE)
um_sync_push_dd_files <- list.files("./data/sssp/um-energy/sync_push_dd", pattern='readings', full.names=TRUE)
um_sync_push_td_files <- list.files("./data/sssp/um-energy/sync_push_td", pattern='readings', full.names=TRUE)


## Add colors
my_colors <- c("#000000", "#6db6ff", "#24ff24", "#490092", "#000000", "#6db6ff", "#24ff24", "#490092")

# individual comparison
for (benchmark in benchmarks) {
  count <- 0

  for (variant in variants) {
    small_filename <- paste0("./", benchmark, "_", variant, ".png")
    png(small_filename)
    
    for (energy_type in energy_types) {
      if (energy_type == "regular-energy") {
        base_energy_dir <- dirname(dirname(async_push_td_files[1]))
      } else {
        base_energy_dir <- dirname(dirname(um_async_push_td_files[1]))
      }
      for (val in vals) {
        temp_path<-paste0(val, "-", benchmark, "-readings")
        temp_pts_path <-paste0(val, "-", benchmark, "-stats")
        
        read_path <- paste0(base_energy_dir, "/", variant, "/", temp_path)
        read_pts_path <- paste0(base_energy_dir, "/", variant, "/", temp_pts_path)
        
        temp_csv <- read.csv(read_path)
        temp_pts_csv <- read.csv(read_pts_path)
        temp_pts_csv <- replace(temp_pts_csv, is.na(temp_pts_csv), 0)
        
        min_timestamp <- min(temp_csv$timestamp)
        temp_csv$timestamp = temp_csv$timestamp - min_timestamp
        temp_info <- data.frame(temp_csv$timestamp, temp_csv$power_draw_mW)
        temp_pts <- data.frame(temp_pts_csv$pt_to, temp_pts_csv$pt_to_timestamp, temp_pts_csv$pt_to_power_mW)
        temp_pts$temp_pts_csv.pt_to_timestamp <- temp_pts$temp_pts_csv.pt_to_timestamp - min_timestamp
        
        if (energy_type == "regular-energy") {
          temp_info$type <- variant
          temp_pts$type <- variant
          temp_pts <- temp_pts[1:4,]
        } else {
          temp_info$type <- paste0("um_", variant)
          temp_pts$type <- paste0("um_", variant)
          temp_pts <- temp_pts[1:3,]
        }
        
        temp_info$energy_type <- energy_type
        temp_pts$energy_type <- energy_type
        colnames(temp_info) <- c("time", "power", "type", "energy_type")
        colnames(temp_pts) <- c("pt", "time", "power", "type", "energy_type")
        
        if (count %% 2 == 0) {
          small_bench_df <- temp_info
          small_bench_pts <- temp_pts
        } else {
          small_bench_df <- rbind(small_bench_df, temp_info)
          small_bench_pts <- rbind(small_bench_pts, temp_pts)
          
          title <- paste0("Power Curve: ", benchmark, " ", variant)
          p <- ggplot(small_bench_df, aes(x = time, y = power)) +
            geom_line(aes(group = type, color = type, linetype = energy_type)) +
            geom_point(data = small_bench_pts, aes(shape = energy_type, size = 10)) +
            ylab("Power (mW)") +
            xlab("Time") +
            ggtitle(title)+
            scale_linetype_manual(values=c("solid","longdash")) +
            scale_color_manual(values=c("black", "gray")) +
            scale_shape_manual(values=c(1, 4)) +
            guides(size = "none") +
            theme_minimal()
          print(p)
        }
        
        count <- count + 1
      }
    }
    
    dev.off()
  }
}

# combined
for (benchmark in benchmarks) {
  count <- 0

  for (variant in variants) {
    for (energy_type in energy_types) {
      if (energy_type == "regular-energy") {
        base_energy_dir <- dirname(dirname(async_push_td_files[1]))
      } else {
        base_energy_dir <- dirname(dirname(um_async_push_td_files[1]))
      }
      for (val in vals) {
        temp_path<-paste0(val, "-", benchmark, "-readings")
        read_path <- paste0(base_energy_dir, "/", variant, "/", temp_path)
        
        temp_csv <- read.csv(read_path)
        temp_csv$timestamp = temp_csv$timestamp - min(temp_csv$timestamp)
        temp_info <- data.frame(temp_csv$timestamp, temp_csv$power_draw_mW)
        
        if (energy_type == "regular-energy") {
          temp_info$type <- variant
        } else {
          temp_info$type <- paste0("um_", variant)
        }
        
        temp_info$energy_type <- energy_type
        
        colnames(temp_info) <- c("time", "power", "type", "energy_type")
        
        if (count == 0) {
          bench_df <- temp_info
        } else {
          bench_df <- rbind(bench_df, temp_info)
        }
        
        count <- count + 1
      }
    }
  }
  
  filename<-paste0("./", benchmark, ".png")
  png(filename)
  title <- paste0("Power Curve: ", benchmark)
  p <- ggplot(bench_df, aes(x = time, y = power, group = type, color = type, linetype = energy_type)) +
    geom_line() +
    ylab("Power (mW)") +
    xlab("Time") +
    ggtitle(title)+
    scale_linetype_manual(values=c("solid","longdash")) +
    scale_color_manual(values=my_colors) +
    theme_minimal()
  
  print(p)
  dev.off()
}




