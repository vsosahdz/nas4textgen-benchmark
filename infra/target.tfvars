# The single source of truth for where this campaign runs.
#
# Pinned by display name, not by id: the name is not sensitive, it is readable, and
# Terraform resolves it to an id from the CLI context at apply time. No subscription
# guid is ever committed.
#
# Every script and the Terraform apply assert the active subscription matches this
# name before doing anything. Creating resources against the wrong subscription would
# bill the wrong budget, so it is guarded rather than trusted.

# Set this to the display name of the subscription you are running in. The campaign
# this archive reports ran in one of the authors' institutional subscriptions, whose
# name is not published: it identifies internal infrastructure and reproducing the
# campaign requires your own.
subscription_name = "SET-YOUR-SUBSCRIPTION-DISPLAY-NAME"

# Storage, workspace and compute cluster must all live here — cross-region traffic
# costs egress and adds latency to every job. Chosen for latency from Mexico among the
# regions where Standard_NC4as_T4_v3 is unrestricted; eastus2 is the fallback.
location = "southcentralus"

# The GPU SKU the sweep runs on. One T4 per node; concurrency comes from more nodes
# rather than packing, so that per-job wall-clock stays measurable.
gpu_sku = "Standard_NC4as_T4_v3"

# Cluster sizing. Both clusters draw on the SAME per-family quota regardless of
# priority — established the hard way: sweep-spot at max 6 reserved all 24 vCPU, and
# measure-dedicated then failed with ClusterMinNodesExceedCoreQuota while the family
# quota still read 0 used. AzureML validates each cluster's maximum against the family
# allowance at creation, not against running cores.
#
# So the maximums must SUM to within the granted family quota:
#   (5 + 1) nodes x 4 vCPU = 24 = exactly what is granted.
#
# One measurement node rather than two is an improvement, not a concession: that
# cluster exists to measure per-job wall-clock and peak memory, so serial execution on
# a single node removes any doubt about contention in the numbers that set the cost
# projection and the packing factor.
family_quota_vcpu     = 24
sweep_max_nodes       = 5
measurement_max_nodes = 1

# Cost guard. The projected campaign is roughly 28 USD on spot.
budget_amount_usd     = 200
budget_start_date     = "2026-08-01T00:00:00Z" # current month — a future date leaves the campaign uncovered
budget_contact_emails = ["vsosa@tec.mx"]
owner_tag             = "vsosa@tec.mx"

# Lifecycle. No node exists without work: both clusters sit at zero, and these windows
# govern only the tail after the last job of a wave. A per-job timeout catches the case
# an idle window cannot — a hung job holds its node busy forever.
sweep_idle_timeout       = "PT5M"
measurement_idle_timeout = "PT2M"
job_timeout_minutes      = 120
