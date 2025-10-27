# CSAT-Prediction-with-ANN


Overview
This project focuses on predicting Customer Satisfaction (CSAT) scores using Deep Learning Artificial Neural Networks (ANN). In the context of e-commerce, understanding customer satisfaction through their interactions and feedback is crucial for enhancing service quality, customer retention, and overall business growth. By leveraging advanced neural network models, we aim to accurately forecast CSAT scores based on a myriad of interaction-related features, providing actionable insights for service improvement.

Project Background
Customer satisfaction in the e-commerce sector is a pivotal metric that influences loyalty, repeat business, and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction, which can be time-consuming and may not always capture the full spectrum of customer experiences. With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time, offering a granular view of service performance and identifying areas for immediate improvement.

Dataset Overview
The dataset encompasses customer satisfaction scores over a one-month period on an e-commerce platform named "Shopzilla." It consists of the following features:

Unique id: Unique identifier for each record (integer).
Channel name: Name of the customer service channel (object/string).
Category: Category of the interaction (object/string).
Sub-category: Sub-category of the interaction (object/string).
Customer Remarks: Feedback provided by the customer (object/string).
Order id: Identifier for the order associated with the interaction (integer).
Order date time: Date and time of the order (datetime).
Issue reported at: Timestamp when the issue was reported (datetime).
Issue responded: Timestamp when the issue was responded to (datetime).
Survey response date: Date of the customer survey response (datetime).
Customer city: City of the customer (object/string).
Product category: Category of the product (object/string).
Item price: Price of the item (float).
Connected handling time: Time taken to handle the interaction (float).
Agent name: Name of the customer service agent (object/string).
Supervisor: Name of the supervisor (object/string).
Manager: Name of the manager (object/string).
Tenure Bucket: Bucket categorizing agent tenure (object/string).
Agent Shift: Shift timing of the agent (object/string).
CSAT Score: Customer Satisfaction (CSAT) score (integer, 1-5).
Project Goal
The primary goal of this project is to develop a deep learning model that can accurately predict the CSAT scores based on customer interactions and feedback. By doing so, we aim to provide e-commerce businesses with a powerful tool to monitor and enhance customer satisfaction in real-time, thereby improving service quality and fostering customer loyalty.

Specific Objectives
Data Preparation: Clean and preprocess the dataset to ensure it is suitable for training a deep learning model.
Feature Engineering: Identify and engineer features from the dataset that are most predictive of CSAT scores.
Model Development: Design and train a deep learning ANN model to predict CSAT scores.
Evaluation: Assess the model's performance using appropriate metrics and validate its predictive accuracy.
Insight Generation: Analyze the model's predictions to identify trends, patterns, and areas for service improvement.
Local Deployment: Deploy the model in your local system and setting to provide ongoing predictions and insights into customer satisfaction.
Features
Data Preprocessing: Handles missing values, categorical encoding, feature scaling, and resampling for class imbalance.
ANN Model: Built with TensorFlow/Keras, including layers for deep learning prediction.
Web App: Flask-based local deployment with an interactive HTML dashboard for predictions.
Evaluation Metrics: Includes accuracy, precision, recall, F1-score, and confusion matrix.
Real-Time Predictions: Input customer data via the web interface and get instant CSAT group predictions ('great', 'low', 'median').
