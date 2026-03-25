locals {
  dashboard_name = "ml-app-${var.env}"
}

resource "aws_cloudwatch_dashboard" "this" {
  dashboard_name = local.dashboard_name

  dashboard_body = jsonencode({
    widgets = [

      # ── Row 1: title ──────────────────────────────────────────────────────────
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 1
        properties = {
          markdown = "## ML App — ${var.env}  |  Inference Pipeline"
        }
      },

      # ── Row 2: inference timing breakdown ────────────────────────────────────
      {
        type   = "metric"
        x      = 0
        y      = 1
        width  = 8
        height = 6
        properties = {
          title  = "Total Inference Duration (ms)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          stat   = "p95"
          metrics = [
            ["MLApp/Inference", "InferenceDurationMs", "Environment", var.env, { label = "p95", stat = "p95" }],
            ["MLApp/Inference", "InferenceDurationMs", "Environment", var.env, { label = "p50", stat = "p50" }],
            ["MLApp/Inference", "InferenceDurationMs", "Environment", var.env, { label = "avg", stat = "Average" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 1
        width  = 8
        height = 6
        properties = {
          title  = "Model (ONNX) Duration (ms)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["MLApp/Inference", "ModelDurationMs", "Environment", var.env, { label = "p95", stat = "p95" }],
            ["MLApp/Inference", "ModelDurationMs", "Environment", var.env, { label = "p50", stat = "p50" }],
            ["MLApp/Inference", "ModelDurationMs", "Environment", var.env, { label = "avg", stat = "Average" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 1
        width  = 8
        height = 6
        properties = {
          title  = "Preprocessing Duration (ms)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["MLApp/Inference", "PreprocessingDurationMs", "Environment", var.env, { label = "p95", stat = "p95" }],
            ["MLApp/Inference", "PreprocessingDurationMs", "Environment", var.env, { label = "p50", stat = "p50" }],
            ["MLApp/Inference", "PreprocessingDurationMs", "Environment", var.env, { label = "avg", stat = "Average" }],
          ]
        }
      },

      # ── Row 3: model quality + input characteristics ──────────────────────────
      {
        type   = "metric"
        x      = 0
        y      = 7
        width  = 12
        height = 6
        properties = {
          title  = "Top-1 Confidence Score"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["MLApp/Inference", "TopScore", "Environment", var.env, { label = "avg", stat = "Average" }],
            ["MLApp/Inference", "TopScore", "Environment", var.env, { label = "min", stat = "Minimum" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 7
        width  = 12
        height = 6
        properties = {
          title  = "Input Image Size (pixels)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["MLApp/Inference", "ImagePixels", "Environment", var.env, { label = "avg", stat = "Average" }],
            ["MLApp/Inference", "ImagePixels", "Environment", var.env, { label = "max", stat = "Maximum" }],
          ]
        }
      },

      # ── Row 4: ALB traffic ────────────────────────────────────────────────────
      {
        type   = "metric"
        x      = 0
        y      = 13
        width  = 8
        height = 6
        properties = {
          title  = "ALB Request Count"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", var.alb_arn_suffix, { label = "Requests", stat = "Sum" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 13
        width  = 8
        height = 6
        properties = {
          title  = "ALB Target Response Time (s)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", var.alb_arn_suffix, { label = "p95", stat = "p95" }],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", var.alb_arn_suffix, { label = "p50", stat = "p50" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 13
        width  = 8
        height = 6
        properties = {
          title  = "ALB 5xx Errors"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", var.alb_arn_suffix, { label = "5xx", stat = "Sum" }],
            ["AWS/ApplicationELB", "HTTPCode_Target_4XX_Count", "LoadBalancer", var.alb_arn_suffix, { label = "4xx", stat = "Sum" }],
          ]
        }
      },

      # ── Row 5: ECS resource utilisation ──────────────────────────────────────
      {
        type   = "metric"
        x      = 0
        y      = 19
        width  = 12
        height = 6
        properties = {
          title  = "ECS CPU Utilisation (%)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", var.ecs_cluster_name, "ServiceName", var.ecs_service_name, { label = "CPU %", stat = "Average" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 19
        width  = 12
        height = 6
        properties = {
          title  = "ECS Memory Utilisation (%)"
          view   = "timeSeries"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ECS", "MemoryUtilization", "ClusterName", var.ecs_cluster_name, "ServiceName", var.ecs_service_name, { label = "Memory %", stat = "Average" }],
          ]
        }
      },
    ]
  })
}
