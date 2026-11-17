# Alarms for the case where a node outlives its work.
#
# Nodes already shut themselves down three independent ways: both clusters have a minimum
# of zero, so no node exists unless a job needs one; an idle window releases a node that
# finished its work; and every job carries a wall-clock limit, which covers the case an
# idle window cannot — a hung job holds its node *busy*, so the cluster never sees it as
# idle and never scales it down.
#
# These alerts do not add a fourth mechanism. They watch the three, and fire when one of
# them has failed. That distinction matters for what to expect from them: an alert cannot
# save money, it can only shorten how long money is lost before someone notices. The
# saving comes from the automation; the alert is what makes the automation's failure
# visible instead of arriving on next month's invoice.
#
# Each threshold is set from a duration the platform already declares, so raising an idle
# window or a job timeout cannot silently leave an alert firing on every normal wave.

resource "azurerm_monitor_action_group" "compute" {
  name                = "${local.prefix}-compute"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "nas4tgcost"

  dynamic "email_receiver" {
    for_each = var.budget_contact_emails
    content {
      name                    = "owner-${email_receiver.key}"
      email_address           = email_receiver.value
      use_common_alert_schema = true
    }
  }

  tags = local.tags
}

locals {
  workspace_id = azurerm_machine_learning_workspace.main.id

  # ISO-8601 windows to minutes, so a threshold can be expressed against the same variable
  # that configures the behaviour it is watching. PT5M -> 5.
  sweep_idle_minutes       = tonumber(replace(replace(var.sweep_idle_timeout, "PT", ""), "M", ""))
  measurement_idle_minutes = tonumber(replace(replace(var.measurement_idle_timeout, "PT", ""), "M", ""))
}

# A node that finished its work and did not go away.
#
# Idle means allocated, billing, and running nothing. The cluster should release it after
# its idle window, so anything still idle well past that window means the scale-down did
# not happen. The evaluation window is deliberately several times the configured idle
# timeout: firing at exactly the timeout would alert on every normal wave, and an alert
# that fires normally is one that gets muted.
resource "azurerm_monitor_metric_alert" "sweep_idle_nodes" {
  name                = "${local.prefix}-sweep-idle-nodes"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [local.workspace_id]
  description = join(" ", [
    "Sweep cluster has nodes allocated and running nothing, well past its idle window.",
    "Scale-down did not happen: these nodes are billing for no work.",
    "Check for a stuck job, then run ./scripts/campaign.sh park to force the cluster to zero.",
  ])

  # Five minutes of granularity over a window several times the idle timeout, so a single
  # noisy sample cannot trip it and a genuine failure cannot hide.
  frequency   = "PT5M"
  window_size = "PT${max(15, local.sweep_idle_minutes * 3)}M"
  severity    = 2

  criteria {
    metric_namespace = "Microsoft.MachineLearningServices/workspaces"
    metric_name      = "Idle Nodes"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 0

    dimension {
      name     = "ClusterName"
      operator = "Include"
      values   = [azurerm_machine_learning_compute_cluster.sweep.name]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.compute.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "measurement_idle_nodes" {
  name                = "${local.prefix}-measure-idle-nodes"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [local.workspace_id]
  description = join(" ", [
    "Measurement cluster has idle nodes past its window. This cluster is dedicated, not",
    "spot, so an idle node here costs several times what one on the sweep cluster does.",
  ])

  frequency   = "PT5M"
  window_size = "PT${max(15, local.measurement_idle_minutes * 3)}M"
  severity    = 1 # dedicated pricing: worth waking someone for

  criteria {
    metric_namespace = "Microsoft.MachineLearningServices/workspaces"
    metric_name      = "Idle Nodes"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 0

    dimension {
      name     = "ClusterName"
      operator = "Include"
      values   = [azurerm_machine_learning_compute_cluster.measurement.name]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.compute.id
  }

  tags = local.tags
}

# A node that is busy for longer than any job should take.
#
# The failure an idle alert cannot see. A hung job keeps its node marked active, so the
# cluster considers it working and the idle window never starts. The per-job timeout is
# supposed to end this, and this alert is what fires when the timeout did not.
#
# Six hours: the slowest architecture under the ceiling is projected at about three, and
# its limit carries head-room on top. Anything active beyond this has escaped both.
resource "azurerm_monitor_metric_alert" "nodes_busy_too_long" {
  name                = "${local.prefix}-nodes-busy-too-long"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [local.workspace_id]
  description = join(" ", [
    "A node has been active for longer than any single job should run. The per-job",
    "wall-clock limit should have released it, so the limit did not fire.",
    "Find the job with ./scripts/campaign.sh status and cancel it; the node follows.",
  ])

  frequency   = "PT15M"
  window_size = "PT6H"
  severity    = 1

  criteria {
    metric_namespace = "Microsoft.MachineLearningServices/workspaces"
    metric_name      = "Active Nodes"
    aggregation      = "Minimum" # the whole six hours, not a spike
    operator         = "GreaterThan"
    threshold        = 0
  }

  action {
    action_group_id = azurerm_monitor_action_group.compute.id
  }

  tags = local.tags
}

# Anything allocated at all, over a window no wave should span.
#
# The backstop for the case the others miss: a campaign that was started and forgotten.
# Twenty-four hours of continuously non-zero allocation is not a wave, it is an oversight,
# whatever the nodes are nominally doing.
resource "azurerm_monitor_metric_alert" "allocation_never_returned_to_zero" {
  name                = "${local.prefix}-allocation-not-zero-24h"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [local.workspace_id]
  description = join(" ", [
    "Nodes have been allocated continuously for a day without the cluster ever returning",
    "to zero. Whatever they are doing, no single wave should look like this.",
    "Verify against the campaign plan; ./scripts/campaign.sh teardown ends it.",
  ])

  frequency   = "PT1H"
  window_size = "P1D" # Azure accepts P1D, not PT24H, for a day
  severity    = 2

  criteria {
    metric_namespace = "Microsoft.MachineLearningServices/workspaces"
    metric_name      = "Total Nodes"
    aggregation      = "Minimum"
    operator         = "GreaterThan"
    threshold        = 0
  }

  action {
    action_group_id = azurerm_monitor_action_group.compute.id
  }

  tags = local.tags
}

output "cost_alarms" {
  description = "What watches for a node outliving its work, and what each one catches."
  value = {
    guarantees = join(" ", [
      "Nodes shut down three ways already: minimum of zero, idle window, per-job",
      "wall-clock limit. These alerts do not add a fourth — they fire when one of the",
      "three has failed, which shortens how long money is lost rather than preventing it.",
    ])
    alerts = {
      "${local.prefix}-sweep-idle-nodes"        = "allocated and running nothing, past the idle window"
      "${local.prefix}-measure-idle-nodes"      = "same, on dedicated pricing, so severity 1"
      "${local.prefix}-nodes-busy-too-long"     = "active past any job's limit — a hung job the idle window cannot see"
      "${local.prefix}-allocation-not-zero-24h" = "never returned to zero in a day: a campaign left running"
    }
    notified    = var.budget_contact_emails
    budget      = "${var.budget_amount_usd} USD monthly, notifying at 50, 90 and forecast 100 per cent"
    verify_with = "./scripts/azure_check_alarms.sh"
  }
}
