#!/bin/bash
# ============================================================
# AWS Infrastructure Setup for Financial RAG Chatbot
# ============================================================
# Prerequisites: AWS CLI configured, Docker installed

set -euo pipefail

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="financial-rag-chatbot"
ECR_REPO="${APP_NAME}"

echo "============================================"
echo "  Financial RAG Chatbot - AWS Deployment"
echo "============================================"

# 1. Create ECR Repository
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "ECR repository already exists"

# Get ECR login
ECR_URI=$(aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --query 'repositories[0].repositoryUri' \
    --output text)

echo "ECR Repository: ${ECR_URI}"

# 2. Build and push Docker image
echo "Building Docker image..."
docker build -t "${APP_NAME}:latest" .

echo "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

echo "Pushing image to ECR..."
docker tag "${APP_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

# 3. Create S3 bucket for data
S3_BUCKET="${APP_NAME}-data-$(aws sts get-caller-identity --query Account --output text)"
echo "Creating S3 bucket: ${S3_BUCKET}"
aws s3 mb "s3://${S3_BUCKET}" --region "${AWS_REGION}" 2>/dev/null || echo "S3 bucket already exists"

# 4. Upload data to S3
echo "Uploading data to S3..."
aws s3 sync data/ "s3://${S3_BUCKET}/data/" --exclude "*.pyc"

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo "  ECR Image: ${ECR_URI}:latest"
echo "  S3 Bucket: ${S3_BUCKET}"
echo ""
echo "  Next steps:"
echo "  1. Create an ECS cluster or EC2 instance"
echo "  2. Pull the image from ECR"
echo "  3. Run: docker run -p 8501:8501 ${ECR_URI}:latest"
echo "============================================"
