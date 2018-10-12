#!/bin/bash

source ./config.cfg


aws ec2 wait instance-running --instance-ids $EC2_INSTANCE_ID
EC2_INSTANCE_IP="$(aws ec2 describe-instances --instance-id $EC2_INSTANCE_ID --query 'Reservations[].Instances[].PublicIpAddress[]' --output text)"
EC2_INSTANCE_URL=ubuntu@$EC2_INSTANCE_IP
[[ -z "$EC2_INSTANCE_IP" ]] && { echo "Couldn't determine address of EC2-instance." ; exit 1; }
echo "EC2 Instance IP: $EC2_INSTANCE_URL"


# Copy log file(s) from remote to local dir (as a backup)
LOCAL_LOG_DIR=./results/logs_remote/
REMOTE_LOG_DIR=repo/results/  

mkdir -p $LOCAL_LOG_DIR
rsync -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" -av -f "- */" -f "+ *" $EC2_INSTANCE_URL:$REMOTE_LOG_DIR $LOCAL_LOG_DIR

# Show live version of remote log file
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $EC2_INSTANCE_URL REMOTE_LOG_DIR=$REMOTE_LOG_DIR 'bash -ls' <<'ENDSSH'
    cd $REMOTE_LOG_DIR
    echo Showing content of `ls -Art | grep -v / | tail -n 1`..
    tail -fn +1 `ls -Art | grep -v / | tail -n 1`
ENDSSH


