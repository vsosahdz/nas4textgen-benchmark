# The durable Azure platform for the sweep.
#
# Terraform owns the platform; it does NOT own the experiments. Individual training runs
# are submitted through the workspace SDK, because managing a thousand short-lived jobs
# as Terraform resources would churn the state file for no benefit.
#
# The subscription is pinned by display name and asserted before anything is created.
# No subscription identifier appears in this repository: the provider takes it from the
# CLI context at apply time.

terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
  # State is local and must be backed up. A remote backend is chicken-and-egg here —
  # the state would need the storage account that this configuration creates — and a
  # single-operator project does not need the coordination a remote backend buys.
  # Documented in docs/replication.md rather than left implicit.
}

provider "azurerm" {
  features {
    key_vault {
      # A purged vault cannot be recreated under the same name for the retention
      # period, which would block a rebuild after teardown.
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

data "azurerm_client_config" "current" {}

data "azurerm_subscription" "current" {}

# ── The guard, in Terraform as well as in the shell ─────────────────────────────
# A plan run against the wrong subscription fails here instead of creating resources
# that bill a budget which did not authorise them.
resource "null_resource" "subscription_guard" {
  lifecycle {
    precondition {
      condition     = data.azurerm_subscription.current.display_name == var.subscription_name
      error_message = <<-EOT
        Active subscription does not match the pinned target.
          active:   ${data.azurerm_subscription.current.display_name}
          expected: ${var.subscription_name}
        Fix with: az account set --subscription "${var.subscription_name}"
      EOT
    }
  }
}

# Both clusters are validated against the family quota at creation, so their maximums
# must sum to within it. Caught here at plan time rather than as a failed cluster.
resource "null_resource" "quota_guard" {
  lifecycle {
    precondition {
      condition     = (var.sweep_max_nodes + var.measurement_max_nodes) * 4 <= var.family_quota_vcpu
      error_message = <<-EOT
        Cluster maximums exceed the granted family quota.
          sweep       ${var.sweep_max_nodes} nodes = ${var.sweep_max_nodes * 4} vCPU
          measurement ${var.measurement_max_nodes} nodes = ${var.measurement_max_nodes * 4} vCPU
          total       ${(var.sweep_max_nodes + var.measurement_max_nodes) * 4} vCPU
          granted     ${var.family_quota_vcpu} vCPU

        Both clusters draw on the same per-family quota regardless of priority. Lower a
        maximum, or raise the quota and update family_quota_vcpu.
      EOT
    }
  }
}

# Storage account names must be globally unique, lowercase, and at most 24 characters,
# so they cannot be derived from the project name alone.
resource "random_string" "suffix" {
  length  = 6
  lower   = true
  numeric = true
  special = false
  upper   = false
}

locals {
  prefix = "nas4tg"

  # Every resource carries these, so cost can be attributed and orphans identified.
  # An untagged resource that survives a failed teardown is invisible in a cost report.
  tags = {
    project    = "nas4textgen"
    purpose    = "multi-objective NAS benchmark for text generation"
    managed_by = "terraform"
    repository = "github.com/vsosahdz/NAS4TextGenerationApproach"
    owner      = var.owner_tag
  }
}

resource "azurerm_resource_group" "main" {
  name     = "${local.prefix}-rg"
  location = var.location
  tags     = local.tags
}

# ── Workspace dependencies ─────────────────────────────────────────────────────
# An Azure ML workspace requires a storage account, a key vault and Application
# Insights. Application Insights in turn requires a Log Analytics workspace.

resource "azurerm_storage_account" "main" {
  name                     = "${local.prefix}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS" # Locally redundant: the corpora are re-downloadable
  # and the benchmark table is published elsewhere with a persistent identifier.

  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  tags                            = local.tags
}

# The data of record. Corpora are retrieved once into this container and every
# downstream stage reads only from here, so the pipeline runs with no public network.
resource "azurerm_storage_container" "cache" {
  name                  = "corpus-cache"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "artifacts"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_key_vault" "main" {
  name                       = "${local.prefix}-kv-${random_string.suffix.result}"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false # so that teardown genuinely leaves nothing
  tags                       = local.tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${local.prefix}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30

  # A daily cap, because uncapped log ingestion is one of the costs that surprises
  # people on a GPU project — it is not the GPU that runs away.
  daily_quota_gb = var.log_daily_quota_gb
  tags           = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "${local.prefix}-insights"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "other"

  # Application Insights carries its own ingestion cap, defaulting to 100 GB per day.
  # Capping Log Analytics alone leaves this one open, and telemetry ingestion is the
  # cost that actually runs away on a project like this — not the GPU.
  daily_data_cap_in_gb                 = var.log_daily_quota_gb
  daily_data_cap_notifications_enabled = true

  # Matches the workspace, so the two do not disagree about how long anything is kept.
  retention_in_days = 30

  tags = local.tags
}

resource "azurerm_container_registry" "main" {
  name                = "${local.prefix}acr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false # the workspace authenticates with its managed identity
  tags                = local.tags
}

# ── The workspace ──────────────────────────────────────────────────────────────

resource "azurerm_machine_learning_workspace" "main" {
  name                    = "${local.prefix}-ws"
  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id
  container_registry_id   = azurerm_container_registry.main.id

  # The workspace's own identity is what pulls images and writes artifacts, so no
  # credential needs to be issued, stored, or rotated.
  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# ── Cost guard ─────────────────────────────────────────────────────────────────
# The projected campaign is roughly 28 USD on spot. The budget is set well above that
# and alerts early, so that a runaway configuration is caught by a notification rather
# than by an invoice.
resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "${local.prefix}-budget"
  resource_group_id = azurerm_resource_group.main.id
  amount            = var.budget_amount_usd
  time_grain        = "Monthly"

  time_period {
    start_date = var.budget_start_date
  }

  dynamic "notification" {
    for_each = [50, 90, 100]
    content {
      enabled        = true
      threshold      = notification.value
      operator       = "GreaterThanOrEqualTo"
      threshold_type = notification.value == 100 ? "Forecasted" : "Actual"
      contact_emails = var.budget_contact_emails
    }
  }
}
