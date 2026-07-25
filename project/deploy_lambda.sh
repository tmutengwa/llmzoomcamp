#!/bin/bash

# Exit on first error
set -e

# Configuration variables (Modify these as needed)
REGION="ap-southeast-2"
REPO_NAME="fastapi-qa-assistant"
FUNCTION_NAME="fastapi-qa-assistant"
ROLE_NAME="fastapi-assistant-lambda-role"

# Retrieve AWS Account ID
echo "Retrieving AWS Account ID..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_URL}/${REPO_NAME}:latest"

echo "Using Account: ${ACCOUNT_ID} in Region: ${REGION}"

# 1. Create ECR Repository (if not exists)
echo "Checking/Creating ECR Repository..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}

# 2. Authenticate Docker to ECR
echo "Authenticating Docker with ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URL}

# 3. Build Docker image for Lambda
echo "Building Docker image from Dockerfile.lambda..."
docker build -t ${REPO_NAME} -f Dockerfile.lambda .

# 4. Tag and Push Image to ECR
echo "Tagging and pushing image to ECR..."
docker tag ${REPO_NAME}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}

# 5. Create IAM Role for Lambda execution (if not exists)
echo "Checking/Creating IAM Role..."
ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query Role.Arn --output text 2>/dev/null) || true

if [ -z "$ROLE_ARN" ]; then
    echo "Creating IAM Role ${ROLE_NAME}..."
    TRUST_POLICY='{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'
    ROLE_ARN=$(aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document "$TRUST_POLICY" --query Role.Arn --output text)
    
    echo "Attaching basic execution policy to role..."
    aws iam attach-role-policy --role-name ${ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    # Wait for role propagation
    sleep 10
fi

# 6. Create or Update Lambda Function
echo "Checking if Lambda function exists..."
FUNCTION_EXISTS=$(aws lambda get-function --function-name ${FUNCTION_NAME} 2>/dev/null) || true

if [ -z "$FUNCTION_EXISTS" ]; then
    echo "Creating Lambda function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${IMAGE_URI} \
        --role ${ROLE_ARN} \
        --timeout 120 \
        --memory-size 2048 \
        --region ${REGION}
else
    echo "Updating Lambda function code..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${IMAGE_URI} \
        --region ${REGION}

    echo "Waiting for function update to complete..."
    aws lambda wait function-updated \
        --function-name ${FUNCTION_NAME} \
        --region ${REGION}

    echo "Updating Lambda function configuration..."
    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout 120 \
        --memory-size 2048 \
        --region ${REGION}
fi

# 7. Create Function URL for Public HTTP Access (if not exists)
echo "Configuring Lambda Function URL (Public)..."
FUNCTION_URL=$(aws lambda get-function-url-config --function-name ${FUNCTION_NAME} --query FunctionUrl --output text 2>/dev/null) || true

if [ -z "$FUNCTION_URL" ]; then
    echo "Creating Function URL configuration..."
    FUNCTION_URL=$(aws lambda create-function-url-config \
        --function-name ${FUNCTION_NAME} \
        --auth-type NONE \
        --query FunctionUrl \
        --output text)
    
    echo "Adding public invocation permissions..."
    aws lambda add-permission \
        --function-name ${FUNCTION_NAME} \
        --action lambda:InvokeFunctionUrl \
        --statement-id FunctionURLAllowPublicAccess \
        --principal "*" \
        --function-url-auth-type NONE
fi

echo "============================================="
echo "Deployment successful!"
echo "AWS Lambda Function URL: ${FUNCTION_URL}"
echo "---------------------------------------------"
echo "To run the Streamlit app locally with zero memory footprint, configure the environment variable:"
echo "export LAMBDA_URL=\"${FUNCTION_URL}\""
echo "streamlit run app_frontend.py"
echo "============================================="
