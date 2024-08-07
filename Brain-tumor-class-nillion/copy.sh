#!/bin/bash

# Define paths
SOURCE_ENV_FILE="$HOME/.config/nillion/nillion-devnet.env"
TARGET_ENV_FILE="quickstart/cra-nillion/.env"

# Create target directory if it does not exist
mkdir -p quickstart/cra-nillion

# Copy .env.example to .env
# cp quickstart/cra-nillion/.env.example $TARGET_ENV_FILE

# Extract and set environment variables from the source file
REACT_APP_NILLION_CLUSTER_ID=$(grep 'NILLION_CLUSTER_ID' $SOURCE_ENV_FILE | cut -d '=' -f2)
REACT_APP_NILLION_BOOTNODE_WEBSOCKET=$(grep 'NILLION_BOOTNODE_WEBSOCKET' $SOURCE_ENV_FILE | cut -d '=' -f2)
REACT_APP_NILLION_NILCHAIN_JSON_RPC=$(grep 'NILLION_NILCHAIN_JSON_RPC' $SOURCE_ENV_FILE | cut -d '=' -f2)
REACT_APP_NILLION_NILCHAIN_PRIVATE_KEY=$(grep 'NILLION_NILCHAIN_PRIVATE_KEY_0' $SOURCE_ENV_FILE | cut -d '=' -f2)
REACT_APP_API_BASE_PATH="/nilchain-proxy"

# Append variables to the target .env file
echo "REACT_APP_NILLION_CLUSTER_ID=${REACT_APP_NILLION_CLUSTER_ID}" >> $TARGET_ENV_FILE
echo "REACT_APP_NILLION_BOOTNODE_WEBSOCKET=${REACT_APP_NILLION_BOOTNODE_WEBSOCKET}" >> $TARGET_ENV_FILE
echo "REACT_APP_NILLION_NILCHAIN_JSON_RPC=${REACT_APP_NILLION_NILCHAIN_JSON_RPC}" >> $TARGET_ENV_FILE
echo "REACT_APP_NILLION_NILCHAIN_PRIVATE_KEY=${REACT_APP_NILLION_NILCHAIN_PRIVATE_KEY}" >> $TARGET_ENV_FILE
echo "REACT_APP_API_BASE_PATH=${REACT_APP_API_BASE_PATH}" >> $TARGET_ENV_FILE

echo "Environment variables have been successfully copied to ${TARGET_ENV_FILE}"
