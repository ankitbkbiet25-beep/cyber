# Web Traffic Anomaly Detection: WAF-Rule Suspicion Predictor

## Project Overview
This project develops a robust, multi-model unsupervised learning system designed to detect anomalous behavior in web traffic logs (CloudWatch/WAF context).

The core goal is to shift security defense from reactive rule-setting to proactive behavioral analysis, enabling real-time flagging of potential DDoS attacks, intrusion attempts, or sophisticated botnets.

This solution was developed as part of a comprehensive Data Science Internship portfolio.

---

## Core Problem Challenge (Unsupervised Learning Mandate)
The central challenge in cybersecurity data is that the most dangerous threats (Zero-Day Attacks) are, by definition, unlabeled and unseen.

- Dataset Challenge: The raw network logs provide only traffic flow features (`bytes_in`, `duration`, `time`) without a reliable 'Attack/Normal' label.
- Solution Mandate: We must use Unsupervised Learning to model the statistical "normality" of traffic and employ a Majority Vote Ensemble for high stability and low false-positive rates.

---

## 🛠️ Technology Stack

Category       | Tools & Libraries                  		 | Function
---------------|-------------------------------------------------|---------
Modeling Core  | scikit-learn, Keras/TensorFlow    		 | Primary ML/DL frameworks for model construction
Ensemble Models| Isolation Forest, One-Class SVM, Neural Network | Models forming the final reliable detection ensemble
Preprocessing  | PowerTransformer, Custom Python Modules 	 | Stabilizing skewed traffic data and creating custom detection features
Deployment     | Streamlit, joblib, HDF5            		 | Interactive dashboard deployment and pipeline serialization

---

## Workflow and Methodology

The project followed a rigorous, iterative approach (fully documented in `notebooks/cyber.ipynb`):

### Phase 1: Data Preparation & Exploration (5-Model Testing)
1. Feature Engineering: Converted the raw `Time` string to `Time_in_seconds` and created key metrics like `Traffic_Ratio` and `Total_per_Second`.
2. Model Exploration: Rigorously tested five initial unsupervised models (ISO, OC-SVM, LOF, DBSCAN, PCA). Models deemed too slow or unstable for real-time prediction (LOF, DBSCAN, PCA) were rejected from the final solution.

### Phase 2: Ensemble Construction
1. Core Backbone: Selected Isolation Forest and One-Class SVM as the stable, complementary backbone models.
2. Deep Learning Integration: Added a shallow Neural Network to capture complex, non-linear anomaly patterns.
3. Persistence: The Power Transformer and all three final models were saved as:
   - power_transformer.pkl
   - iso_model.pkl
   - svm_model.pkl
   - anomaly_nn_model.h5

### Phase 3: Real-Time Deployment and Voting Logic
1. Ensemble Logic: The `main.py` Streamlit app implements a Majority Rule Ensemble:
   Anomaly is flagged if 2 out of 3 models vote 'Anomaly'.
2. Deployment: The full pipeline is loaded into a Streamlit dashboard for real-time traffic analysis and transparent voting results.

---

## Running the Application

1. Install Dependencies:
   Ensure `streamlit`, `scikit-learn`, `tensorflow`, etc., are installed in your environment.

2. Run the Dashboard:
   streamlit run main/main.py

---

## Deployment Snapshot

The deployed application provides full transparency into the detection process:

- Output: Clear status - "Anomaly Detected" or "Normal Traffic."
- Transparency: Individual votes from the Isolation Forest, One-Class SVM, and Neural Net are displayed alongside the final Majority Vote decision, allowing security analysts to understand the model's rationale.
