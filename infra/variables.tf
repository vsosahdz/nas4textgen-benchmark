# Inputs. Defaults come from infra/target.tfvars, which is committed on purpose: it
# carries no secret and it is what the subscription guard checks against.

variable "subscription_name" {
  description = <<-EOT
    Display name of the subscription this campaign is pinned to. Asserted before any
    resource is created. Pinned by name rather than id so that no subscription
    identifier is committed — the provider resolves the id from the CLI context.
  EOT
  type        = string
}

variable "location" {
  description = <<-EOT
    Azure region. Storage, workspace and both clusters share it; cross-region traffic
    costs egress and adds latency to every job.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.location))
    error_message = "Use the lowercase region short name, for example southcentralus."
  }
}

variable "gpu_sku" {
  description = <<-EOT
    GPU VM size. One T4 per node; concurrency comes from more nodes rather than packing
    several jobs per GPU, so that per-job wall-clock stays measurable.
  EOT
  type        = string
  default     = "Standard_NC4as_T4_v3"
}

variable "family_quota_vcpu" {
  description = <<-EOT
    Granted vCPU for the GPU family under **Microsoft.MachineLearningServices** in this
    region — not the Microsoft.Compute quota that `az vm list-usage` reports. Azure ML
    enforces its own counter for the same family and the two are granted independently;
    read the enforced one with scripts/azure_ml_quota.sh.

    Cluster maximums are checked against this at plan time, so an oversized request
    fails with arithmetic rather than as ClusterMinNodesExceedCoreQuota at creation.
  EOT
  type        = number
}

variable "sweep_max_nodes" {
  description = <<-EOT
    Maximum spot nodes for the sweep. Five nodes run the 188 GPU-hour sweep in roughly
    thirty-eight hours. Constrained by the family quota shared with the measurement
    cluster, not by the regional low-priority cap.
  EOT
  type        = number
  default     = 5

  validation {
    condition     = var.sweep_max_nodes >= 1
    error_message = "At least one node."
  }
}

variable "measurement_max_nodes" {
  description = <<-EOT
    Maximum dedicated nodes for the canary and the pilot. These measure per-job
    wall-clock and peak memory, so they stay off spot: a preemption would contaminate
    the numbers that set the cost projection and the packing factor.
  EOT
  type        = number
  default     = 2
}

variable "budget_amount_usd" {
  description = <<-EOT
    Monthly budget for the resource group. The projected campaign is roughly 28 USD on
    spot; this is set well above so that a runaway configuration is caught by an alert
    rather than an invoice.
  EOT
  type        = number
  default     = 200
}

variable "budget_start_date" {
  description = <<-EOT
    First day of the budget period, as RFC3339. Azure requires the first of a month and
    rejects a start date in the past, so this is an explicit input rather than a
    computed default — a computed one would break every plan run after its month ends.
  EOT
  type        = string

  validation {
    condition     = can(regex("^\\d{4}-\\d{2}-01T00:00:00Z$", var.budget_start_date))
    error_message = "Must be the first of a month in RFC3339, for example 2026-09-01T00:00:00Z."
  }
}

variable "budget_contact_emails" {
  description = "Who is alerted at 50%, 90% actual, and 100% forecast of the budget."
  type        = list(string)

  validation {
    condition     = length(var.budget_contact_emails) > 0
    error_message = "At least one address, or the budget alerts reach nobody."
  }
}

variable "log_daily_quota_gb" {
  description = <<-EOT
    Daily ingestion cap for Log Analytics. Uncapped log ingestion is one of the costs
    that surprises people on a GPU project; it is not the GPU that runs away.
  EOT
  type        = number
  default     = 1
}

variable "owner_tag" {
  description = "Value of the `owner` tag, so cost is attributable and orphans identifiable."
  type        = string
}

variable "sweep_idle_timeout" {
  description = <<-EOT
    How long a sweep node may sit idle before deallocating, as an ISO-8601 duration.
    While a wave is queued the cluster is never idle, so this only governs the tail
    after the last job finishes — and that tail is pure waste. Short.
  EOT
  type        = string
  default     = "PT5M"

  validation {
    condition     = can(regex("^PT[0-9]+M$", var.sweep_idle_timeout))
    error_message = "An ISO-8601 duration in whole minutes, for example PT5M."
  }
}

variable "measurement_idle_timeout" {
  description = <<-EOT
    Idle window for the dedicated cluster. Shorter than the sweep's, because this
    cluster runs few jobs and is dedicated, so an idle node here bills at full rate.
  EOT
  type        = string
  default     = "PT2M"

  validation {
    condition     = can(regex("^PT[0-9]+M$", var.measurement_idle_timeout))
    error_message = "An ISO-8601 duration in whole minutes, for example PT2M."
  }
}

variable "job_timeout_minutes" {
  description = <<-EOT
    Per-job wall-clock limit. A hung job keeps its node marked busy, so the cluster
    never scales down and bills indefinitely — the failure mode that idle timeouts do
    not catch. Derived from the duration bound the canary measures, not guessed, and
    carried into every submitted job.
  EOT
  type        = number
  default     = 120
}
