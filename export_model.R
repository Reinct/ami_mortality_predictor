# ============================================================================
# export_model.R — 从 mlr3 learner 中导出 CatBoost 模型 + 缩放参数
# 运行此脚本生成 streamlit_app 所需的所有文件
# ============================================================================
setwd("D:/paper/aotric_AMI/data/AMI_Mortality_Predictor")

library(mlr3)
library(mlr3verse)
library(mlr3extralearners)
library(mlr3pipelines)
library(catboost)
library(data.table)

load("model_objects.RData")

# ── 1. 从 GraphLearner 提取 CatBoost 原始模型 ─────────────────────────
extract_catboost_model <- function(learner_obj) {
  if (inherits(learner_obj, "GraphLearner")) {
    for (po_id in names(learner_obj$graph$pipeops)) {
      candidate <- learner_obj$graph$pipeops[[po_id]]
      if (inherits(candidate, "PipeOpLearner")) {
        inner <- candidate$learner
        if (inherits(inner, "AutoTuner")) {
          learner_obj <- inner
          break
        }
      }
    }
  }
  if (inherits(learner_obj, "AutoTuner")) {
    return(learner_obj$learner$model)
  } else if (inherits(learner_obj, "Learner")) {
    return(learner_obj$model)
  }
  stop("Cannot extract model")
}

cb_model <- extract_catboost_model(learner_shiny)

# ── 2. 保存 CatBoost 模型 ─────────────────────────────────────────────
output_dir <- "D:/paper/aotric_AMI/data/streamlit_app"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

catboost.save_model(cb_model, file.path(output_dir, "ami_catboost_model.cbm"))
cat("[OK] CatBoost model saved to ami_catboost_model.cbm\n")

# ── 3. 提取缩放参数 ──────────────────────────────────────────────────
extract_scale_params <- function(learner_obj) {
  # 从 graph state 找
  gs <- learner_obj$state$model
  if (is.list(gs)) {
    for (nm in names(gs)) {
      item <- gs[[nm]]
      if (is.list(item) && !is.null(item$center) && !is.null(item$scale)) {
        return(list(center = item$center, scale = item$scale))
      }
    }
  }
  # 从 PipeOp scale 找
  if (!is.null(learner_obj$graph$pipeops$scale)) {
    po_scale <- learner_obj$graph$pipeops$scale
    private_state <- po_scale$.__enclos_env__$private$.state
    if (!is.null(private_state) && !is.null(private_state$center)) {
      return(list(center = private_state$center, scale = private_state$scale))
    }
  }
  # 从训练数据计算
  if (!is.null(learner_obj$state$train_task)) {
    td <- learner_obj$state$train_task$data()
    num_cols <- names(td)[sapply(td, is.numeric) & names(td) != "live_result"]
    center <- sapply(td[, num_cols, drop = FALSE], mean, na.rm = TRUE)
    scale  <- sapply(td[, num_cols, drop = FALSE], sd, na.rm = TRUE)
    return(list(center = center, scale = scale))
  }
  return(NULL)
}

scale_params <- extract_scale_params(learner_shiny)
cat(sprintf("[OK] Scale params: %d features\n", length(scale_params$center)))

# ── 4. 保存特征元数据 ─────────────────────────────────────────────────
feature_names <- names(X_shiny)
feature_types <- sapply(X_shiny, function(x) if (is.numeric(x)) "numeric" else "categorical")

# 每个特征的取值范围 / 选项
feature_meta <- list()
for (nm in feature_names) {
  v <- X_shiny[[nm]]
  if (is.numeric(v)) {
    feature_meta[[nm]] <- list(
      type   = "numeric",
      min    = floor(min(v, na.rm = TRUE)),
      max    = ceiling(max(v, na.rm = TRUE)),
      default = round(median(v, na.rm = TRUE), 2),
      step   = if (max(v, na.rm = TRUE) > 100) 1 else 0.1,
      unit   = ""
    )
  } else {
    feature_meta[[nm]] <- list(
      type    = "categorical",
      options = as.list(sort(unique(na.omit(v)))),
      default = unique(na.omit(v))[1]
    )
  }
}

# 显示名称
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

# 性能指标
if (exists("perf_summary")) {
  perf <- as.list(perf_summary)
} else {
  perf <- list(
    AUC = 0.804,
    Sensitivity = 0.724,
    Specificity = 0.736,
    Precision = 0.354,
    F1 = 0.476
  )
}

best_cutoff_val <- if (exists("best_cutoff")) best_cutoff else 0.084

# 特征重要性（从 CatBoost 模型获取）
importance <- catboost.get_feature_importance(cb_model)
importance_df <- data.frame(
  feature = feature_names,
  importance = importance
)
importance_df <- importance_df[order(-importance_df$importance), ]

# ── 5. 写入 JSON ──────────────────────────────────────────────────────
library(jsonlite)

config <- list(
  model_path      = "ami_catboost_model.cbm",
  scale_params    = scale_params,
  features        = feature_meta,
  feature_names   = feature_names,
  display_names   = display_names,
  performance     = perf,
  best_cutoff     = best_cutoff_val,
  feature_importance = list(
    features   = importance_df$feature,
    importance = importance_df$importance
  )
)

write_json(config, file.path(output_dir, "model_config.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("[OK] model_config.json written\n"))

# ── 6. 复制模型文件到 streamlit_app ─────────────────────────────────────
cat(sprintf("\n[DONE] All files exported to %s\n", output_dir))
cat("  - ami_catboost_model.cbm\n")
cat("  - model_config.json\n")
cat("\nNow create the Python Streamlit app.\n")
