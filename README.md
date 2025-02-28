# AI-Driven-Customer-Churn-Prediction
Developed a scalable, cloud-native churn prediction system using Machine Learning, deployed via AWS Lambda with Docker and exposed through API Gateway for real-time predictions.
Key Features
- Machine Learning Model – Trained using RandomForestClassifier, optimized for accuracy
- Cloud Deployment – Containerized with Docker, deployed on AWS Lambda
- Scalable API – Built using AWS API Gateway for external access
- Efficient Data Pipeline – Processed and encoded customer data with NumPy, Pandas, and Scikit-learn
- Logging & Monitoring – Integrated AWS CloudWatch for real-time debugging and analytics
Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Joblib)
- AWS Lambda, API Gateway, CloudWatch
- Docker & AWS ECR for containerized deployment
- Flask for REST API (inside Docker container)
- Boto3 for AWS S3 & IAM automation
Impact
- Reduced churn risk identification time with real-time predictions
- Serverless and cost-efficient architecture, minimizing cloud resource consumption
- Robust API integration, allowing seamless business integration
