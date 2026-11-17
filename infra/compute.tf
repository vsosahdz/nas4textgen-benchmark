# Compute clusters.
#
# Two of them, because the campaign has two kinds of work with incompatible needs.
#
# The sweep is a thousand independent twenty-minute jobs, so a preemption costs at most
# one job and spot is close to free. The canary and the pilot, by contrast, exist to
# *measure* per-job wall-clock and peak memory — running those on spot would let
# preemption contaminate the very numbers they are meant to establish, which then feed
# the cost projection and the packing factor for everything after.
#
# Both scale to zero. An idle cluster with a non-zero minimum is the single largest
# avoidable cost on a project like this, and it is the mistake that does not announce
# itself.

resource "azurerm_machine_learning_compute_cluster" "sweep" {
  name                          = "sweep-spot"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  location                      = azurerm_resource_group.main.location
  vm_size                       = var.gpu_sku
  vm_priority                   = "LowPriority"

  scale_settings {
    # Zero is not a default here, it is the guarantee: no node exists unless a job
    # needs one. An idle cluster with a non-zero minimum is the largest avoidable
    # cost on a project like this, and it does not announce itself.
    min_node_count = 0

    # Set to zero to park the platform between waves: allocation becomes impossible
    # while every resource stays intact. See scripts/campaign.sh park.
    max_node_count = var.sweep_max_nodes

    # While a wave is queued the cluster is never idle, so this window only governs
    # the tail after the last job. Short, because that tail is pure waste.
    scale_down_nodes_after_idle_duration = var.sweep_idle_timeout
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(local.tags, {
    role     = "sweep"
    priority = "spot"
  })
}

resource "azurerm_machine_learning_compute_cluster" "measurement" {
  # Ordering is load-bearing, not cosmetic. Both clusters draw on the same family
  # quota, so this one can only be created once the sweep's maximum has been lowered
  # to leave room. Without the dependency Terraform is free to create this first,
  # against a sweep still reserving the whole allowance, and it fails exactly as it
  # did the first time.
  depends_on = [azurerm_machine_learning_compute_cluster.sweep]

  name                          = "measure-dedicated"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  location                      = azurerm_resource_group.main.location
  vm_size                       = var.gpu_sku
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = var.measurement_max_nodes
    scale_down_nodes_after_idle_duration = var.measurement_idle_timeout
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(local.tags, {
    role     = "canary-and-pilot"
    priority = "dedicated"
  })
}

# ── Quota: Azure ML has its own, and it is not the Compute one ──────────────────
#
# Two counters, same VM family, granted independently:
#
#   Microsoft.Compute                  "Standard NCASv3_T4 Family vCPUs"    24
#   Microsoft.MachineLearningServices  "Standard NCasT4_v3 family"           0
#   regional low-priority allowance                                        350
#
# Azure ML managed compute enforces the second. The first request granted the first,
# which is why a single-node dedicated cluster still failed with
# ClusterMinNodesExceedCoreQuota against "a family vCPU quota of 0".
#
# Low-priority compute is charged against the regional low-priority allowance, which is
# why sweep-spot provisioned while measure-dedicated did not. The two clusters are not
# competing for one grant; they draw on different counters.
#
# Read the enforced quota with scripts/azure_ml_quota.sh — `az vm list-usage` reports the
# wrong provider. Request increases under Quotas > Machine Learning, or from the
# workspace's Usage + quotas page; the Compute blade raises the counter that is not
# checked.

output "quota_requirements" {
  description = "vCPU needed at the configured SKU, four per node, for the configured limits."
  value = {
    sku                   = var.gpu_sku
    region                = var.location
    spot_vcpu_needed      = var.sweep_max_nodes * 4
    dedicated_vcpu_needed = var.measurement_max_nodes * 4
    # The Machine Learning provider, not Compute. Raising the Compute counter does not
    # raise the one Azure ML enforces — that mistake cost two failed cluster creations.
    request_dedicated_in_portal = "Workspace > Usage + quotas > ${var.location} > Standard NCASv3_T4 Family Cluster Dedicated"
    verify_with                 = "./scripts/azure_check_quota.sh"
  }
}

output "workspace" {
  description = "Where jobs are submitted. The SDK reads this; Terraform does not submit."
  value = {
    name            = azurerm_machine_learning_workspace.main.name
    resource_group  = azurerm_resource_group.main.name
    location        = azurerm_resource_group.main.location
    sweep_cluster   = azurerm_machine_learning_compute_cluster.sweep.name
    measure_cluster = azurerm_machine_learning_compute_cluster.measurement.name
  }
}

output "storage" {
  description = "The data of record. Downstream stages read only from the cache container."
  value = {
    account            = azurerm_storage_account.main.name
    cache_container    = azurerm_storage_container.cache.name
    artifact_container = azurerm_storage_container.artifacts.name
  }
}

output "container_registry" {
  description = "Where the pinned training image lives; its digest goes in every manifest."
  value       = azurerm_container_registry.main.login_server
}
