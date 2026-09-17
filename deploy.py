"""Redeploy without the aws CLI. `python deploy.py push` then `python deploy.py update`."""

import base64
import subprocess
import sys
import time

import boto3

REGION = "us-east-1"
ACCOUNT = "450484000186"
REPO = "seller-triage"
FUNCTION = "seller-triage"
LOCAL_TAG = "seller-triage:latest"
REGISTRY = "%s.dkr.ecr.%s.amazonaws.com" % (ACCOUNT, REGION)
IMAGE = "%s/%s:latest" % (REGISTRY, REPO)


def sh(*args, **kw):
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, check=True, text=True, **kw)


def push():
    tok = boto3.client("ecr", region_name=REGION).get_authorization_token()["authorizationData"][0]
    user, pw = base64.b64decode(tok["authorizationToken"]).decode().split(":", 1)
    subprocess.run(["docker", "login", "--username", user, "--password-stdin", REGISTRY],
                   input=pw, text=True, check=True)
    sh("docker", "tag", LOCAL_TAG, IMAGE)
    sh("docker", "push", IMAGE)
    digest = subprocess.run(["docker", "inspect", "--format", "{{index .RepoDigests 0}}", IMAGE],
                            capture_output=True, text=True, check=True).stdout.strip()
    print("pushed", digest)


def update(env_updates: dict | None = None):
    lam = boto3.client("lambda", region_name=REGION)
    if env_updates:
        cfg = lam.get_function_configuration(FunctionName=FUNCTION)
        env = dict(cfg.get("Environment", {}).get("Variables", {}))
        env.update(env_updates)
        lam.update_function_configuration(FunctionName=FUNCTION, Environment={"Variables": env})
        _wait(lam)
        print("env now:", env)
    lam.update_function_code(FunctionName=FUNCTION, ImageUri=IMAGE)
    _wait(lam)
    print("function updated to", IMAGE)


def _wait(lam):
    for _ in range(90):
        c = lam.get_function_configuration(FunctionName=FUNCTION)
        st = c.get("LastUpdateStatus")
        if st == "Successful":
            return
        if st == "Failed":
            raise SystemExit("update failed: %s" % c.get("LastUpdateStatusReason"))
        time.sleep(2)
    raise SystemExit("timed out waiting for the function update")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    if cmd == "push":
        push()
    elif cmd == "update":
        kv = dict(a.split("=", 1) for a in sys.argv[2:])
        update(kv or None)
    else:
        raise SystemExit("push | update [KEY=VALUE ...]")
