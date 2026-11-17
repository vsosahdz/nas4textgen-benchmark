"""Submission against the provisioned workspace. Deliberately not Terraform.

Terraform owns what is durable and few: the resource group, the storage account, the
workspace, the registry, the two clusters. It does not own the thousand training runs.

That split is a design decision with consequences, so it is worth stating why. A job held
in Terraform state would make every run a resource to reconcile: ``terraform plan`` after
a wave would diff a thousand entries, a single failed run would block an unrelated apply,
and tearing down the platform would first attempt to delete run history — the record the
whole study rests on. Jobs are ephemeral, numerous, and have their own lifecycle; state
files are for the opposite. So the platform is declared, and the work is submitted.

Nothing here is imported by the training image. A node runs ``nas4textgen.train`` and has
no reason to hold a client that can spawn more nodes.
"""

from __future__ import annotations
