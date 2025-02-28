# Use AWS Lambda's official Python base image
FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /var/task

# Copy application files
COPY app.py . 

# Copy Model and Preprocessing Files into the container
COPY churn_model.pkl . 
COPY encoder.pkl . 
COPY scaler.pkl . 

# Install required dependencies
RUN pip install boto3 joblib numpy pandas scikit-learn

# Set the Lambda function handler
CMD ["app.lambda_handler"]
