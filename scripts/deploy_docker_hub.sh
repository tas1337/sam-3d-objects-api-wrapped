#!/bin/bash
set -e

echo "=========================================="
echo "Docker Image Build & Push (Fully Automated)"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Create a DigitalOcean droplet"
echo "  2. SSH in and watch the build live"
echo "  3. Download checkpoints, build Docker, push to Hub"
echo "  4. AUTO-DELETE the droplet when done"
echo ""
echo "Total cost: ~\$0.18-0.36 (auto-deletes!)"
echo "Total time: ~1-2 hours"
echo ""

# Check dependencies
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl is required."
    exit 1
fi

if ! command -v ssh &> /dev/null; then
    echo "ERROR: ssh is required."
    exit 1
fi

# Find SSH key
SSH_KEY_PATH=""
for keyfile in ~/.ssh/id_ed25519 ~/.ssh/id_rsa ~/.ssh/sampod; do
    if [ -f "$keyfile" ] && [ -f "$keyfile.pub" ]; then
        SSH_KEY_PATH="$keyfile"
        break
    fi
done

if [ -z "$SSH_KEY_PATH" ]; then
    read -p "SSH private key path (without .pub): " SSH_KEY_PATH
else
    echo "Found SSH key: $SSH_KEY_PATH"
    read -p "Use this key? [Y/n]: " USE_KEY
    if [ "$USE_KEY" = "n" ] || [ "$USE_KEY" = "N" ]; then
        read -p "SSH private key path (without .pub): " SSH_KEY_PATH
    fi
fi

# Expand ~ manually for Windows compatibility
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "ERROR: SSH private key not found at $SSH_KEY_PATH"
    exit 1
fi

if [ ! -f "$SSH_KEY_PATH.pub" ]; then
    echo "ERROR: SSH public key not found at $SSH_KEY_PATH.pub"
    exit 1
fi

SSH_PUBLIC_KEY=$(cat "$SSH_KEY_PATH.pub")
SSH_PRIVATE_KEY="$SSH_KEY_PATH"
echo ""

# Collect credentials (all tokens hidden for privacy)
read -p "Docker Hub Username: " DOCKER_USERNAME
read -sp "Docker Hub Access Token: " DOCKER_TOKEN
echo ""
read -sp "GitHub Personal Access Token: " GITHUB_TOKEN
echo ""
read -sp "HuggingFace Token: " HF_TOKEN
echo ""
read -sp "DigitalOcean API Token: " DO_TOKEN
echo ""

# Ask which image(s) to build
echo ""
echo "Which image(s) to build?"
echo "  1) Pod only (api_server.py)"
echo "  2) Serverless only (handler.py)"
echo "  3) Both"
read -p "Choice [1/2/3]: " BUILD_CHOICE

# Droplet config
DROPLET_NAME="docker-builder-$(date +%s)"
DROPLET_SIZE="s-4vcpu-8gb"
DROPLET_REGION="nyc3"
DROPLET_IMAGE="ubuntu-24-04-x64"

echo ""
echo "Creating droplet: $DROPLET_NAME"
echo "Size: $DROPLET_SIZE (4 vCPUs, 8GB RAM)"
echo ""

# Create cloud-init script that runs on boot
USER_DATA=$(cat << CLOUD_INIT_EOF
#!/bin/bash
exec > /var/log/docker-build.log 2>&1
set -ex

echo "========== Starting Docker Build =========="
date

cd /root

# Step 1: Install system dependencies FIRST
echo "Installing system dependencies..."
apt update
apt install -y python3-pip git curl

# Step 2: Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com | sh

# Step 3: Install HuggingFace CLI (pip is now available)
echo "Installing HuggingFace CLI..."
pip install --break-system-packages 'huggingface-hub[cli]<1.0'

# Step 4: Clone repo
echo "Cloning repository..."
git clone https://${GITHUB_TOKEN}@github.com/tas1337/sam-3d-objects.git
cd sam-3d-objects

# Step 5: Download checkpoints
echo "Downloading HuggingFace checkpoints (~13GB)..."
export HF_TOKEN=${HF_TOKEN}
huggingface-cli login --token \$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects

mv checkpoints/hf-download/checkpoints/* checkpoints/
rm -rf checkpoints/hf-download

echo "Checkpoints downloaded:"
ls -la checkpoints/

# Step 6: Login to Docker Hub
echo "Logging into Docker Hub..."
echo '${DOCKER_TOKEN}' | docker login -u ${DOCKER_USERNAME} --password-stdin

# Step 7: Build and push Docker image(s)
BUILD_CHOICE="${BUILD_CHOICE}"

if [ "\$BUILD_CHOICE" = "1" ] || [ "\$BUILD_CHOICE" = "3" ]; then
    echo ""
    echo "========== Building POD image =========="
    docker build -t sam3d-objects:latest .
    docker tag sam3d-objects:latest ${DOCKER_USERNAME}/sam3d-objects:latest
    docker push ${DOCKER_USERNAME}/sam3d-objects:latest
    echo "Pod image pushed: ${DOCKER_USERNAME}/sam3d-objects:latest"
fi

if [ "\$BUILD_CHOICE" = "2" ] || [ "\$BUILD_CHOICE" = "3" ]; then
    echo ""
    echo "========== Building SERVERLESS image =========="
    docker build -f Dockerfile.serverless -t sam3d-objects-serverless:latest .
    docker tag sam3d-objects-serverless:latest ${DOCKER_USERNAME}/sam3d-objects-serverless:latest
    docker push ${DOCKER_USERNAME}/sam3d-objects-serverless:latest
    echo "Serverless image pushed: ${DOCKER_USERNAME}/sam3d-objects-serverless:latest"
fi

echo ""
echo "========== BUILD COMPLETE =========="
date
echo "SUCCESS" > /root/build_status.txt

# Step 8: Auto-delete droplet after successful build
echo "Build complete! Auto-deleting droplet..."
DROPLET_ID=\$(curl -s http://169.254.169.254/metadata/v1/id)
curl -X DELETE -H "Authorization: Bearer ${DO_TOKEN}" "https://api.digitalocean.com/v2/droplets/\$DROPLET_ID"
echo "Droplet deleted!"
CLOUD_INIT_EOF
)

# First, add SSH key to DigitalOcean account
echo "Adding SSH key to DigitalOcean..."
SSH_KEY_NAME="docker-builder-key-$(date +%s)"
SSH_KEY_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    -d "{
        \"name\": \"$SSH_KEY_NAME\",
        \"public_key\": \"$SSH_PUBLIC_KEY\"
    }" \
    "https://api.digitalocean.com/v2/account/keys")

SSH_KEY_ID=$(echo "$SSH_KEY_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$SSH_KEY_ID" ]; then
    # Key might already exist, try to get fingerprint
    SSH_KEY_FINGERPRINT=$(ssh-keygen -lf "$SSH_PRIVATE_KEY.pub" -E md5 2>/dev/null | awk '{print $2}' | sed 's/MD5://')
    if [ -z "$SSH_KEY_FINGERPRINT" ]; then
        echo "Warning: Could not add SSH key to DO. Continuing anyway..."
        SSH_KEYS_PARAM=""
    else
        SSH_KEYS_PARAM="\"ssh_keys\": [\"$SSH_KEY_FINGERPRINT\"],"
    fi
else
    echo "SSH key added: $SSH_KEY_ID"
    SSH_KEYS_PARAM="\"ssh_keys\": [$SSH_KEY_ID],"
fi

# Create droplet with cloud-init and SSH key
echo "Creating droplet with build script..."
CREATE_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    -d "{
        \"name\": \"$DROPLET_NAME\",
        \"size\": \"$DROPLET_SIZE\",
        \"region\": \"$DROPLET_REGION\",
        \"image\": \"$DROPLET_IMAGE\",
        $SSH_KEYS_PARAM
        \"user_data\": \"$(echo "$USER_DATA" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')\"
    }" \
    "https://api.digitalocean.com/v2/droplets")

DROPLET_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$DROPLET_ID" ]; then
    echo "Failed to create droplet!"
    echo "$CREATE_RESPONSE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Droplet created: $DROPLET_ID"
echo "=========================================="
echo ""

# Wait for IP
echo "Waiting for IP address..."
DROPLET_IP=""
for i in {1..30}; do
    DROPLET_INFO=$(curl -s -H "Authorization: Bearer $DO_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID")
    
    DROPLET_IP=$(echo "$DROPLET_INFO" | grep -o '"ip_address":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ ! -z "$DROPLET_IP" ] && [ "$DROPLET_IP" != "null" ]; then
        break
    fi
    
    echo "  Waiting for IP... ($i/30)"
    sleep 5
done

if [ -z "$DROPLET_IP" ]; then
    echo "ERROR: Could not get droplet IP"
    exit 1
fi

echo ""
echo "Droplet IP: $DROPLET_IP"
echo ""

# Wait for SSH to be ready
echo "Waiting for SSH to be ready..."
SSH_READY=false
for i in {1..30}; do
    if ssh -i "$SSH_PRIVATE_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes root@$DROPLET_IP "echo SSH ready" 2>/dev/null; then
        echo "SSH is ready!"
        SSH_READY=true
        break
    fi
    echo "  Waiting for SSH... ($i/30)"
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo ""
    echo "ERROR: Could not connect via SSH after 5 minutes."
    echo "The build is still running in background!"
    echo ""
    echo "Try manually: ssh -i $SSH_PRIVATE_KEY root@$DROPLET_IP"
    echo "Then run: tail -f /var/log/docker-build.log"
    echo ""
    echo "Or check Docker Hub in 1-2 hours."
    exit 1
fi

echo ""
echo "=========================================="
echo "Connecting to watch build progress..."
echo "=========================================="
echo ""
echo "Press Ctrl+C to disconnect (build continues in background)"
echo "Droplet auto-deletes when build completes!"
echo ""
echo "Check Docker Hub when done:"
if [ "$BUILD_CHOICE" = "1" ] || [ "$BUILD_CHOICE" = "3" ]; then
    echo "  https://hub.docker.com/r/$DOCKER_USERNAME/sam3d-objects"
fi
if [ "$BUILD_CHOICE" = "2" ] || [ "$BUILD_CHOICE" = "3" ]; then
    echo "  https://hub.docker.com/r/$DOCKER_USERNAME/sam3d-objects-serverless"
fi
echo ""
echo "=========================================="
echo ""

# SSH in and tail the log (wait for it to exist)
ssh -i "$SSH_PRIVATE_KEY" -o StrictHostKeyChecking=no root@$DROPLET_IP "
echo 'Waiting for build to start...'
while [ ! -f /var/log/docker-build.log ]; do
    sleep 2
done
echo 'Build started! Tailing log...'
echo ''
tail -f /var/log/docker-build.log
"
