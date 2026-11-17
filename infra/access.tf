# Data-plane access. Declared here because it is not implied by anything above.
#
# Owning a subscription does not grant the right to read a blob. Azure separates the
# control plane — create, configure and delete a storage account — from the data plane —
# read and write the bytes inside it, and Owner covers only the first. The failure is
# unhelpful: an authenticated request that returns AuthorizationPermissionMismatch, from a
# principal that can delete the entire account.
#
# What this file grants: the operator who uploads the cache and downloads the results, and
# the two clusters whose nodes read the corpus and write outputs. What it deliberately does
# not grant is noted below. None of these principals receives a stored key or a SAS token;
# the storage account issues no credential that could be committed, leaked, or left to
# expire in the middle of a sweep.

# The operator: uploads the corpus cache and downloads results before teardown. Their
# principal is already read by data.azurerm_client_config.current in main.tf.
#
# Contributor rather than Reader: the same person must be able to write the cache up and
# read results back down, and splitting that across two identities would mean the campaign
# scripts could not run as one user.
resource "azurerm_role_assignment" "operator_blob" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id

  # A user principal, not a service principal, when Terraform is run from a developer
  # login. Stating it avoids the provider guessing wrong and producing a role assignment
  # that exists but grants nothing.
  principal_type = "User"
}

# Not declared here, on purpose: the workspace's own blob role, and AcrPull for both
# clusters. Azure ML grants those to its identities as part of creating the workspace and
# the clusters, so declaring them made every apply fail with RoleAssignmentExists against
# an assignment Terraform had not made and did not own. Two owners for one resource is a
# permanent conflict, not a one-time collision.
#
# They still have to hold, so they are asserted rather than declared —
# ./scripts/azure_check_access.sh fails if any of them is missing. Verifying something
# another system owns is the honest arrangement; claiming to create it is not.

# The clusters. A node mounts the corpus read-only and writes its outputs, so the identity
# that runs the container needs the data-plane role directly — the workspace's role does
# not travel to it.
resource "azurerm_role_assignment" "sweep_cluster_blob" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.sweep.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}

resource "azurerm_role_assignment" "measurement_cluster_blob" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.measurement.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}

# Not here either: the two Azure ML datastores that let jobs address these containers.
# The azurerm provider requires an account key or a SAS token on a blob datastore and has
# no keyless option, so declaring them would put a storage key into Terraform state and
# into the datastore — replacing an identity that can be revoked with a secret that has to
# be stored and rotated. They are created keyless through the SDK instead, by
# `nas4textgen.cli datastores`, and the role assignments above are what make that work.


output "data_plane_access" {
  description = "Who may read and write corpus and artifact bytes, and how to verify it."
  value = {
    note = join(" ", [
      "Owner on the subscription does not grant blob data access.",
      "These assignments are what make the keyless path work; without them an upload",
      "fails with AuthorizationPermissionMismatch from a principal that could delete",
      "the whole account.",
    ])
    role          = "Storage Blob Data Contributor"
    propagation   = "Role assignments take up to five minutes to take effect."
    verify_with   = "./scripts/azure_check_access.sh"
    declared_here = "operator, sweep cluster, measurement cluster — blob data"
    asserted_only = join(" ", [
      "The workspace's blob role and AcrPull for both clusters are created by Azure ML",
      "itself. They are checked by the script, never declared, because two owners for",
      "one assignment is a permanent conflict rather than a one-time collision.",
    ])
  }
}
