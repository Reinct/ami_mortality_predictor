# ============================================================================
# export_shap.R — 从 workspace.RData 提取 SHAP 值，导出给 Streamlit
# ============================================================================
setwd("D:/paper/aotric_AMI/data")
load("workspace.RData")

library(jsonlite)

output_dir <- "D:/paper/aotric_AMI/data/streamlit_app"

# ── 显示名称映射 ──────────────────────────────────────────────────────
shap_disp <- c(
  "revasc"       = "Coronary Revascularization",
  "aspirin_like" = "Aspirin at Discharge",
  "alt"          = "Alanine Aminotransferase (ALT)",
  "DD"           = "D-Dimer",
  "Uric_acid"    = "Uric Acid",
  "cr"           = "Serum Creatinine",
  "AMIed"        = "Prior Myocardial Infarction",
  "age"          = "Age",
  "COPD"         = "COPD",
  "bnp"          = "NT-proBNP"
)

# ── 1. mean(|SHAP|) 特征重要性 ─────────────────────────────────────
mean_shap <- colMeans(abs(rf_shap$S))
mean_shap <- sort(mean_shap, decreasing = TRUE)

# 应用显示名称
shap_importance <- data.frame(
  feature    = names(mean_shap),
  importance = as.numeric(mean_shap),
  stringsAsFactors = FALSE
)
shap_importance$display_name <- sapply(shap_importance$feature, function(f) {
  if (f %in% names(shap_disp)) shap_disp[f] else f
})

cat("SHAP mean(|SHAP|) values:\n")
print(shap_importance, digits = 4)

# ── 2. 写入 JSON ──────────────────────────────────────────────────────
shap_export <- list(
  features       = shap_importance$feature,
  display_names  = shap_importance$display_name,
  importance     = shap_importance$importance
)

write_json(shap_export, file.path(output_dir, "shap_importance.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("[OK] shap_importance.json exported to %s\n", output_dir))

# ── 3. 可选: 导出完整 SHAP 矩阵用于 beeswarm ──────────────────────────
# rf_shap$S: 294 x 10 matrix
# 取前 1000 行（如果矩阵很大）
n_export <- min(nrow(rf_shap$S), 1000)
shap_matrix <- as.data.frame(rf_shap$S[1:n_export, , drop = FALSE])
shap_matrix$row_id <- seq_len(n_export)
write.csv(shap_matrix, file.path(output_dir, "shap_values.csv"), row.names = FALSE)
cat(sprintf("[OK] shap_values.csv (%d rows x %d cols)\n",
            nrow(shap_matrix), ncol(shap_matrix)))

cat("\n[DONE] SHAP export complete.\n")
