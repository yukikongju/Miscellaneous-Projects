#!/bin/sh
set -eu

# If a Docker secret is mounted, use it for GCP auth.
if [ -f /run/secrets/gcp_service_account ]; then
  export GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_service_account
fi

exec uv run python src/utils.py
