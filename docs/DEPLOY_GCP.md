# Deploy to GCP Free-Tier VM

This guide walks through creating a fresh GCP VM and running the Predictive
Maintenance Platform on it. Total time: ~10 minutes.

The Docker image already contains pre-trained models, so no file copying
or training is needed.

---

## Prerequisites

- A Google account (Gmail)
- A GCP project (free tier is fine)

---

## Step 1: Create a GCP Project (skip if you already have one)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Sign in with your Gmail
3. Click the project dropdown (top bar) > **New Project**
4. Project Name: `predictive-maintenance` (or anything you like)
5. Parent resource: **No organization**
6. Click **Create**

---

## Step 2: Create the VM

Go to **Compute Engine** > **VM Instances** > **Create Instance**.

Use these settings:

| Setting | Value |
|---------|-------|
| **Name** | `pm-demo` |
| **Region** | `us-central1` (Iowa) -- free-tier eligible |
| **Zone** | `us-central1-a` |
| **Series** | E2 |
| **Machine type** | `e2-micro` (free tier) |

**Boot disk** -- click **Change**:

| Setting | Value |
|---------|-------|
| Operating system | Ubuntu |
| Version | Ubuntu 22.04 LTS |
| Boot disk type | Standard persistent disk |
| Size | 30 GB (free tier allows up to 30GB) |

**Firewall**:
- [x] Allow HTTP traffic
- [x] Allow HTTPS traffic

Click **Create**. Wait for the green checkmark.

---

## Step 3: Open firewall for port 8000

The API runs on port 8000. You need a firewall rule to allow it.

**Option A: Via the GCP Console**

1. Go to **VPC Network** > **Firewall** > **Create Firewall Rule**
2. Name: `pm-allow-8000`
3. Targets: **Specified target tags** > `http-server`
4. Source IP ranges: `0.0.0.0/0`
5. Protocols and ports: **TCP** > `8000`
6. Click **Create**

**Option B: Via gcloud CLI (from your Mac)**

```bash
gcloud compute firewall-rules create pm-allow-8000 \
  --allow tcp:8000 \
  --target-tags=http-server \
  --description="Allow port 8000 for PM demo" \
  --project <YOUR_PROJECT_ID>
```

> Replace `<YOUR_PROJECT_ID>` with your actual project ID
> (e.g. `predictive-maintenance-492012`).

---

## Step 4: SSH into the VM

In the GCP Console, go to **Compute Engine** > **VM Instances**.
Click the **SSH** button next to your VM. A browser terminal opens.

All remaining commands run in this SSH terminal.

---

## Step 5: Install Docker

```bash
# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io
```

What this does:
- `apt-get update` -- refreshes the list of available packages
- `apt-get install -y docker.io` -- installs Docker (the `-y` flag auto-confirms)

Verify it's installed:

```bash
sudo docker --version
# Expected: Docker version 24.x.x (or similar)
```

---

## Step 6: Add swap space (important for e2-micro)

The e2-micro has only 1GB RAM. Adding swap prevents out-of-memory crashes.

```bash
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

What this does:
- `fallocate -l 1G /swapfile` -- creates a 1GB file on disk
- `chmod 600 /swapfile` -- restricts permissions (security)
- `mkswap /swapfile` -- formats the file as swap space
- `swapon /swapfile` -- activates the swap

Verify:

```bash
free -m
# You should see Swap: ~1023 total
```

Make swap permanent (survives reboot):

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Step 7: Pull and run the API

```bash
# Pull the image (contains pre-trained models)
sudo docker pull sherozshaikh/predictive-maintenance-api:1.1.0
```

What this does:
- Downloads the Docker image from Docker Hub
- The `:1.1.0` tag includes pre-trained models baked into the image

```bash
# Run the container
sudo docker run -d \
  --name pm-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e PYTHONPATH=/app \
  sherozshaikh/predictive-maintenance-api:1.1.0
```

What each flag does:
- `-d` -- run in background (detached mode)
- `--name pm-api` -- name the container `pm-api` so you can reference it later
- `--restart unless-stopped` -- auto-restart if it crashes or VM reboots
- `-p 8000:8000` -- map port 8000 on the VM to port 8000 in the container
- `-e PYTHONPATH=/app` -- set environment variable needed by the app

Wait 30 seconds for the model to load, then verify:

```bash
sleep 30 && curl -s http://localhost:8000/v1/health
# Expected: {"status":"healthy","models_loaded":true}
```

---

## Step 8: Open in your browser

Find your VM's external IP:

```bash
curl -s ifconfig.me
```

Open in your browser:

```
http://<YOUR_VM_IP>:8000
```

You should see the Predictive Maintenance dashboard. Click **Run Prediction**
to test.

Other URLs:

| URL | What |
|-----|------|
| `http://<IP>:8000` | Dashboard |
| `http://<IP>:8000/docs` | Swagger API docs |
| `http://<IP>:8000/v1/metrics` | Prometheus metrics |
| `http://<IP>:8000/v1/health` | Health check |
| `http://<IP>:8000/v1/alerts` | Alert history |

---

## Stopping and Starting the VM

**Stop** (to save costs / when not demoing):

1. GCP Console > Compute Engine > VM Instances
2. Click on `pm-demo`
3. Click **Stop** at the top

**Start** (when you want to demo again):

1. Click **Start** on the same page
2. Wait ~1 minute for the VM + container to boot
3. The external IP may change -- check the console for the new one

> **Tip**: To get a fixed IP, go to **VPC Network** > **IP Addresses** >
> **Reserve a Static Address** and attach it to `pm-demo`. Free while the
> VM is running.

---

## Useful Docker Commands (run in SSH terminal)

```bash
# Check container status
sudo docker ps

# View container logs
sudo docker logs pm-api

# Follow logs in real-time
sudo docker logs -f pm-api

# Restart the container
sudo docker restart pm-api

# Stop the container
sudo docker stop pm-api

# Remove and re-create (e.g. after pulling a new image version)
sudo docker rm -f pm-api
sudo docker pull sherozshaikh/predictive-maintenance-api:1.1.0
sudo docker run -d --name pm-api --restart unless-stopped -p 8000:8000 -e PYTHONPATH=/app sherozshaikh/predictive-maintenance-api:1.1.0

# Check RAM usage
free -m

# Check container resource usage
sudo docker stats --no-stream
```

---

## Cleanup (delete everything)

To delete the VM and all resources:

```bash
# From your Mac terminal (or do it in the GCP Console)
gcloud compute instances delete pm-demo --zone us-central1-a --project <YOUR_PROJECT_ID>
gcloud compute firewall-rules delete pm-allow-8000 --project <YOUR_PROJECT_ID>
```

Or in the GCP Console: Compute Engine > VM Instances > select `pm-demo` > **Delete**.
