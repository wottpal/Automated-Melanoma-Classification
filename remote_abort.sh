#!/bin/bash

source ./config.cfg


# Stop Instance
aws ec2 stop-instances --instance-ids $EC2_INSTANCE_ID
aws ec2 wait instance-stopped --instance-ids $EC2_INSTANCE_ID
