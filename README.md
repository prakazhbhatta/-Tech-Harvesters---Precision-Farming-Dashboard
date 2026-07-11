# TechHarvesters
# 🌾 Tech Harvesters - Precision Farming Dashboard

An interactive, real-time dashboard built for precision agriculture using ESP32, IoT sensors, AI-based disease detection, and GIS-based field analysis. This system enables smarter decisions in irrigation, disease prevention, and crop yield forecasting.

## 🚀 Features

- 📍 *Field Overview Panel*  
  View total area
, crop type, growth stage, and pesticide usage.

- 📡 *IoT Node Status*  
  Live connectivity status for Irrigation Node, Drone Node, and Sensor Node.

- 🌱 *Real-Time Sensor Data*  
  Monitors:
  - Soil Moisture
  - Temperature
  - Humidity
  - Light Intensity(no module)

- 🤖 *AI-Based Predictive Analytics*  
  Disease risk prediction for:
  - 🥔 Potato (e.g., Late Blight,late Blight,healthy,)
  - 🍅 Tomato (e.g., Early Blight,late Blight,healthy)  
  Plus:
  - Water requirement suggestions
  - Yield estimations

- 🧠 *AI Model Training Insights*  
  Displays training and validation accuracy/loss of ML models for disease detection.

- 🗺 *GIS-Based Field Analysis*  
  Interactive map showing:
  - Soil variability
  - Crop zones
  - Segmentation by health zones
  -recommendation of crops

- 💬 *Chatbot + AI Image Detector*  
  - AI-powered chatbot for plant disease Q&A  
  - Upload leaf images for disease diagnosis using AI

## 🛠 Built With

- *HTML/CSS + TailwindCSS* – Responsive UI Design  
- *JavaScript* – Dynamic sensor updates & interaction  
- *Firebase* – Real-time database (for sensor data)  
- *Leaflet.js* – GIS-based mapping  
- *AI Models* – Image-based disease detection using ML  

                

# Project description, features, and setup guide

#How to use it:
run in cmd 

python backend.py
test_modules.py
generate csv.py or train_leaf_diseases_pytorch.py
#browse into html directory
run login.html
# run it using local server or firebase domain
create a firebase domain and connect the api's


![WhatsApp Image 2025-06-28 at 11 17 11_ad724e16](https://github.com/user-attachments/assets/23468b89-e51b-4f43-8f1a-40c2480cf1ba)
