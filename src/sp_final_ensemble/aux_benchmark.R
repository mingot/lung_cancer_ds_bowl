# strip dsb_/luna_ and .npz
strip_dsb <- function(x) {
  gsub("(dsb_|luna_)([[:alnum:]]+)(\\.npz)", "\\2", x)
}

# compute feature-wise auc
# df_feature and df_cancer must contain patientid column
library(ggplot2)
benchmark_feat <- function(df_feature, df_cancer)  {
  df_feature <- data.table::as.data.table(df_feature)
  df_cancer <- data.table::as.data.table(df_cancer)
  
  df_all <- df_feature[df_cancer, on = "patientid", nomatch = 0]
  message("Number of rows: ", nrow(df_all), " out of ", nrow(df_cancer), 
          " labelled examples")
  message("Benchmarking auc:")
  df_bench <- apply(df_all[, -c("patientid", "cancer")], 
                    2, 
                    function(col) Metrics::auc(df_all$cancer, col))
  
  # show(df_bench)
  df_bench <- sort(df_bench, decreasing = FALSE)
  df_plot <- data.frame(feature = names(df_bench), 
                        auc = df_bench, 
                        stringsAsFactors = FALSE, row.names = NULL)
  df_plot$feature <- factor(df_plot$feature, levels = df_plot$feature)
  g <- ggplot(df_plot, aes(x = feature, y = auc)) + 
    geom_bar(stat = "identity") + 
    geom_hline(yintercept = 0.5, color = "red") + 
    coord_flip() 
  print(g)
  return(df_plot)
}