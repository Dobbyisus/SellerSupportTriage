"""Shared OpenSearch connection.

Talks to an Amazon OpenSearch Service domain with SigV4 request signing, or to
a local container when OPENSEARCH_LOCAL=1 (useful for rehearsing the whole
index/upload/search cycle before the AWS domain exists).
"""

from opensearchpy import OpenSearch, RequestsHttpConnection

import config


def connect(timeout: int = 60) -> OpenSearch:
    if config.USE_LOCAL:
        return OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=timeout,
        )

    if not config.ENDPOINT:
        raise SystemExit(
            "OPENSEARCH_ENDPOINT is not set.\n"
            "  PowerShell:  $env:OPENSEARCH_ENDPOINT = 'search-xxx.us-east-1.es.amazonaws.com'\n"
            "  or set OPENSEARCH_LOCAL=1 to use a local container instead."
        )

    import boto3
    from requests_aws4auth import AWS4Auth

    creds = boto3.Session(region_name=config.REGION).get_credentials()
    if creds is None:
        raise SystemExit("No AWS credentials found. Check ~/.aws/credentials.")
    frozen = creds.get_frozen_credentials()
    auth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        config.REGION,
        "es",
        session_token=frozen.token,
    )

    return OpenSearch(
        hosts=[{"host": config.ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        http_compress=True,
        timeout=timeout,
        max_retries=3,
        retry_on_timeout=True,
    )


def ping(client: OpenSearch) -> dict:
    """Return basic cluster info, with a readable error if unreachable."""
    from opensearchpy.exceptions import OpenSearchException

    try:
        info = client.info()
    except OpenSearchException as exc:
        raise SystemExit("Cannot reach OpenSearch: %s" % exc)

    version = info.get("version", {})
    return {
        "distribution": version.get("distribution", "elasticsearch"),
        "number": version.get("number", "?"),
        "cluster": info.get("cluster_name", "?"),
    }


def require_hybrid_support(client: OpenSearch) -> None:
    """The normalization processor needs OpenSearch 2.10 or newer."""
    info = ping(client)
    if info["distribution"] != "opensearch":
        raise SystemExit(
            "Connected to %s, not OpenSearch. Hybrid search needs OpenSearch 2.10+."
            % info["distribution"]
        )
    try:
        major, minor = (int(p) for p in info["number"].split(".")[:2])
    except ValueError:
        return
    if (major, minor) < (2, 10):
        raise SystemExit(
            "OpenSearch %s is too old — the normalization processor that makes\n"
            "hybrid search possible landed in 2.10. Recreate the domain on a\n"
            "newer version." % info["number"]
        )
