#!/bin/bash
#
# Training Commands executed remotely are in `run_remotely.sh`
#
# Requirements:
#   - Local: `pip install --upgrade --user awscli`
#   - Local: `aws configure`
#   - Remote: `sudo locale-gen de_DE.UTF-8`
#   - Remote: `sudo timedatectl set-timezone Europe/Berlin`
#   - Remote: `source activate tensorflow_p36`

source ./config.cfg


# Start EC2 Instance
aws ec2 start-instances --instance-ids $EC2_INSTANCE_ID
aws ec2 wait instance-running --instance-ids $EC2_INSTANCE_ID
EC2_INSTANCE_IP="$(aws ec2 describe-instances --instance-id $EC2_INSTANCE_ID --query 'Reservations[].Instances[].PublicIpAddress[]' --output text)"
EC2_INSTANCE_URL=ubuntu@$EC2_INSTANCE_IP
[[ -z "$EC2_INSTANCE_IP" ]] && { echo "Couldn't determine address of EC2-instance." ; exit 1; }
echo "EC2 Instance IP: $EC2_INSTANCE_URL"
sleep 15


# Copy most recent execution commands over to the remote 
# (this eliminates the need to do commit/push each time `run_remotely.sh` is changed)
RUN_REMOTELY=run_remotely.sh 
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $RUN_REMOTELY $EC2_INSTANCE_URL:$RUN_REMOTELY


# Execute Commands on Remote (and keep them running there independently with `nohup`)
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $EC2_INSTANCE_URL REPO_URL=$REPO_URL RUN_REMOTELY=$RUN_REMOTELY 'bash -ls' <<'ENDSSH'
  source activate tensorflow_p36
  rm -rf data_tmp
  mv repo/data/ data_tmp/
  rm -rf repo
  git clone $REPO_URL repo
  mv data_tmp/ repo/data/
  cd repo
  pip install --upgrade pip
  pip install -r requirements.txt
  pip list
  mkdir -p results
  mv ../$RUN_REMOTELY $RUN_REMOTELY
  nohup ./$RUN_REMOTELY >./results/$(date "+%Y.%m.%d-%H.%M.%S").log 2>&1 &
ENDSSH


# Remove IP from ~/.ssh/known_hosts
# ssh-keygen -R $EC2_INSTANCE_IP
