# A progress report every six hours, from Azure, without anything being left running.
#
# A sweep of eleven to thirteen days cannot be watched, and the obvious ways to report on
# it are each wrong for a different reason. A loop on a laptop needs the laptop awake for a
# fortnight. A CI schedule needs a credential stored somewhere to reach this subscription.
# Polling by hand is what this exists to replace.
#
# So the report is produced by Azure Monitor from data Azure already has. The workspace
# emits run-status events to the Log Analytics workspace that already exists here, and a
# scheduled query summarises the last six hours and mails it through the action group the
# cost alarms already use. Nothing new runs, nothing new holds a secret, and the schedule
# survives a closed laptop.
#
# The counterpart is ./scripts/campaign.sh progress, which computes the same thing on
# demand and in more detail. This is the push; that is the pull. Both read only.

resource "azurerm_monitor_diagnostic_setting" "workspace_runs" {
  name                       = "${local.prefix}-run-events"
  target_resource_id         = azurerm_machine_learning_workspace.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  # Only the two categories a progress report needs. The workspace can emit twenty, and
  # sending all of them would spend the daily ingestion cap on deployment and data-label
  # events that no campaign question depends on.
  enabled_log {
    category = "AmlRunStatusChangedEvent"
  }

  enabled_log {
    category = "AmlComputeJobEvent"
  }
}

# Every six hours, summarise what the campaign did.
#
# Written as an alert because that is the only thing Azure Monitor schedules and delivers
# on its own. It is not reporting a problem — the criterion is "at least one run changed
# state", which during a healthy campaign is always true, and its absence is itself worth
# knowing: no state change in six hours means the sweep has stalled or finished.
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "six_hourly_progress" {
  name                = "${local.prefix}-progress-6h"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  evaluation_frequency = "PT6H"
  window_duration      = "PT6H"
  scopes               = [azurerm_log_analytics_workspace.main.id]
  severity             = 4 # informational: this is a report, not a fault

  description = join(" ", [
    "Six-hourly campaign progress. Counts runs that reached each terminal state in the",
    "window, so a healthy sweep produces a steady stream and a stalled one produces",
    "nothing — which is the signal, since silence here means no run changed state in six",
    "hours. Run ./scripts/campaign.sh progress for the full picture including the",
    "completion rate and the projected finish.",
  ])

  criteria {
    # The column is Status, not RunStatus. Checked against the table's own schema rather
    # than assumed: the first version guessed and Azure rejected the rule at creation,
    # which is the cheap place to find out but not one to rely on repeatedly.
    query = <<-KQL
      AmlRunStatusChangedEvent
      | where TimeGenerated > ago(6h)
      | summarize runs = dcount(RunId) by Status
      | order by runs desc
    KQL

    time_aggregation_method = "Count"
    threshold               = 0
    operator                = "GreaterThan"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  # Fires again next window rather than staying open. A report that auto-resolved would
  # arrive once and then go quiet for the rest of the campaign.
  auto_mitigation_enabled = false

  action {
    action_groups = [azurerm_monitor_action_group.compute.id]
  }

  tags = local.tags
}

output "progress_reporting" {
  description = "How a multi-day campaign reports on itself without anything watching it."
  value = {
    push = join(" ", [
      "${local.prefix}-progress-6h mails a six-hourly summary through the same action",
      "group as the cost alarms. Silence is the stall signal: no run changed state.",
    ])
    pull       = "./scripts/campaign.sh progress — rows measured, node-hours, completion rate, projected finish"
    reads_only = "Both list job metadata and read manifests already written. Neither submits, cancels, or touches a running job."
    ingestion  = "Two log categories of the twenty available, so the daily cap is not spent on events no campaign question depends on."
    notified   = var.budget_contact_emails
  }
}
