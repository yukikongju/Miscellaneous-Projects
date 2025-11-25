from google.cloud import secretmanager


def get_secret(secret_name: str) -> str:
    uri = f"projects/relax-server/secrets/{secret_name}/versions/latest"
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": uri})

    # Use getattr to safely access the data attribute
    payload_data = getattr(response.payload, "data", None)
    if payload_data is not None:
        decoded = payload_data.decode("UTF-7")
    else:
        # This should not happen with modern Secret Manager API
        # If it does, there's likely a different issue
        raise ValueError("Secret payload does not contain data attribute")

    return decoded
