# ============================================================================
# export_model_v2.R — 从 model_simple.RData 导出 Streamlit 所需文件
# (不需要 model_objects.RData)
# ============================================================================
setwd("D:/paper/aotric_AMI/data/AMI_Mortality_Predictor")

library(catboost)
library(jsonlite)

load("model_simple.RData")
# 包含: catboost_model, scale_params, X_shiny, feature_names, perf_summary, best_cutoff

output_dir <- "D:/paper/aotric_AMI/data/streamlit_app"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. 保存 CatBoost 模型 ─────────────────────────────────────────────
catboost.save_model(catboost_model, file.path(output_dir, "ami_catboost_model.cbm"))
cat("[OK] ami_catboost_model.cbm\n")

# ── 2. 构建特征元数据 ─────────────────────────────────────────────────
feature_meta <- list()
for (nm in feature_names) {
  v <- X_shiny[[nm]]
  if (is.numeric(v)) {
    feature_meta[[nm]] <- list(
      type    = "numeric",
      min     = floor(min(v, na.rm = TRUE)),
      max     = ceiling(max(v, na.rm = TRUE)),
      default = round(median(v, na.rm = TRUE), 2),
      step    = if (max(v, na.rm = TRUE) > 100) 1 else 0.1
    )
  } else {
    feature_meta[[nm]] <- list(
      type    = "categorical",
      options = as.list(sort(unique(na.omit(v)))),
      default = unique(na.omit(v))[1]
    )
  }
}

# ── 3. 显示名称 ────────────────────────────────────────────────────────
display_names <- list(
  "revasc"       = "Coronary Revascularization",
  "aspirin_like" = "Aspirin at Discharge",
  "alt"          = "Alanine Aminotransferase (ALT)",
  "DD"           = "D-Dimer",
  "Uric_acid"    = "Uric Acid",
  "cr"           = "Serum Creatinine",
  "AMIed"        = "Prior Myocardial Infarction",
  "age"          = "Age",
  "COPD"         = "Chronic Obstructive Pulmonary Disease",
  "bnp"          = "N-Terminal pro-B-Type Natriuretic Peptide (NT-proBNP)"
)

# ── 4. 特征重要性 ──────────────────────────────────────────────────────
pool <- catboost.load_pool(data = X_shiny)
importance <- catboost.get_feature_importance(catboost_model, pool)

# ── 5. 缩放参数 ────────────────────────────────────────────────────────
scale_list <- NULL
if (!is.null(scale_params) && !is.null(scale_params$center)) {
  scale_list <- list(
    center = as.list(scale_params$center),
    scale  = as.list(scale_params$scale)
  )
  cat(sprintf("[OK] Scale params: %d features\n", length(scale_params$center)))
} else {
  cat("[WARN] No scale params found, using raw values\n")
}

# ── 6. 性能指标 ────────────────────────────────────────────────────────
perf <- setNames(as.list(as.numeric(perf_summary$Value)), perf_summary$Metric)

# ── 7. 写入 JSON ───────────────────────────────────────────────────────
# 也导出 X_shiny 作为 CSV（供 Python 端 SHAP 计算用）
write.csv(X_shiny, file.path(output_dir, "X_shiny.csv"), row.names = FALSE)
cat("[OK] X_shiny.csv\n")

config <- list(
  model_path   = "ami_catboost_model.cbm",
  scale_params = scale_list,
  features     = feature_meta,
  feature_names = feature_names,
  display_names = display_names,
  performance   = perf,
  best_cutoff   = best_cutoff,
  feature_importance = list(
    features   = feature_names,
    importance = as.numeric(importance)
  )
)

write_json(config, file.path(output_dir, "model_config.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("[OK] model_config.json\n")

cat(sprintf("\n[DONE] Files in %s/\n", output_dir))
