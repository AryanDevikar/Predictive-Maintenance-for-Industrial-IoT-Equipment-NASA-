Abstract: 
Forecasting Equipment Failures in Industrial IoT Environments 
This study addresses the critical challenge of unplanned equipment failures in industrial IoT (or IIoT) 
environments, which often result in significant downtime and increased maintenance costs. Using the NASA 
Commercial Modular Aero-Propulsion System Simulation (aka C-MAPSS) dataset, models were developed to 
predict the Remaining Useful Life (RUL) of aircraft engines based on sensor data collected under various 
operating conditions. The project pipeline includes data preprocessing, feature engineering, and model 
training with cross-validation. Five machine learning regression models were compared: Linear Regression, 
Ridge Regression, Random Forest, xgBoost, and LightGBM and their performance was evaluated using Root 
Mean Square Error (RMSE), Mean Absolute Error (MAE), and a custom PHM08 scoring function. This project 
aims to contribute to the development of effective predictive maintenance strategies for industrial equipment, 
which can substantially reduce operational costs and improve system reliability. 
Introduction: 
In today's industrial operations, unexpected equipment failures can lead to substantial economic losses 
through production downtime, emergency maintenance costs, and potential safety hazards. To address this, 
NASA released the C-MAPSS dataset which provides run-to-failure data for turbofan engines operating under 
various conditions. 
Predictive maintenance offers several advantages over traditional maintenance strategies: 
• Reduction in unplanned downtime  
• Optimization of maintenance scheduling  
• Extension of equipment lifespan  
• Decrease in overall maintenance costs  
• Improvement in operational efficiency and safety 
Dataset: 
To address the problem of unexpected equipment failures, NASA released the C-MAPSS dataset which provides 
run-to-failure data for turbofan engines operating under various conditions. 
Each engine starts with different degrees of initial wear and manufacturing variation, making the dataset 
realistic and challenging. The engines operate normally at the start of each time series and develop a fault at 
some point, which then grows in magnitude until system failure. The dataset contains the following 
components: 
• Unit number: Unique identifier for each engine 
• Time cycles: Number of operational cycles completed 
• Operational settings: Three operational settings that influence engine performance 
• Sensor measurements: 21 sensor readings (e.g., temperature, pressure, fan speed) for each cycle 
The dataset is structured as: 
• Training data: Contains complete run-to-failure sensor data for multiple engines (train_FD001) 
• Test data: Contains sensor data up to a certain point before failure (test_FD001) 
• Ground truth RUL values for the test data (RUL_FD001) 
Figure 1 illustrates the distribution of engine lifetimes in the training dataset. 
Methodology: 
Data Preprocessing 
Initial exploration of the dataset revealed several challenges that had to be addressed: 
• Feature selection: Not all sensors showed variation with increasing cycles, indicating they provided little 
information about engine degradation. After correlation analysis, seven sensor channels (s_1, s_5, s_6, 
s_10, s_16, s_18, s_19) were identified as constant or near-constant and were removed from further 
analysis. 
• RUL calculation: For the training data, RUL values were calculated by subtracting the current cycle from 
the maximum cycle for each engine:  
• RUL = max_time_cycle - current_time_cycle 
• Data normalization: To ensure model performance, all features were scaled using MinMaxScaler to 
bring values into a consistent range. 
• RUL clipping: Analysis of RUL distribution showed very high values for early cycles of long-lasting 
engines. Since degradation patterns are typically more evident closer to failure, RUL values were 
clipped at 195 cycles to improve model performance. 
Feature Engineering 
To enhance the predictive power of the models, several feature engineering techniques were applied: 
• Moving Averages: To reduce the noise in sensor readings and highlight degradation trends, 10-cycle 
rolling means were calculated for each sensor. This reduced the impact of measurement noise while 
preserving the underlying degradation patterns. 
• Derived Features: The code includes calculation of various statistics and derived features based on the 
sensor readings. 
• Risk Categorization: For classification models, RUL values were transformed into three risk categories: 
RISK ZONE: RUL ≤ 68 cycles 
MODERATED RISK: 69 < RUL ≤ 137 cycles 
NO RISK: RUL > 137 cycles 
These thresholds were determined by analyzing the distribution of RUL values in the training data to create 
approximately balanced classes. 
Figure 2 shows the distribution after feature engineering. 
Model Development 
Both regression and classification approaches were implemented to address different aspects of the predictive 
maintenance problem: 
Regression 
The following regression models were implemented to predict the exact RUL value: 
• Linear Regression: A custom implementation of linear regression was used as a baseline model. 
• Ridge Regression: Linear regression with L2 regularization to prevent overfitting. Hyperparameter 
tuning was performed to determine the optimal alpha value. 
• Random Forest Regressor: An ensemble of decision trees with hyperparameter optimization for the 
number of estimators and maximum depth. 
• XGBoost Regressor: A gradient boosting framework optimized for computational speed and model 
performance. 
• LightGBM Regressor: A gradient boosting framework that uses a leaf-wise tree growth strategy. 
Classification 
For the classification task, the models were trained to predict the risk category of each engine: 
• Random Forest Classifier: Ensemble of decision trees for classification 
• XGBoost Classifier: Gradient boosting algorithm adapted for multi-class classification. 
• Naive Bayes: Probabilistic classifier based on Bayes' theorem. 
• K-Nearest Neighbors: Instance-based learning algorithm. 
Evaluation Metrics 
For Regression Models: 
• Root Mean Square Error (RMSE): Measures the square root of the average squared difference 
between predicted and actual RUL values 
• R² Score: Indicates the proportion of variance in the dependent variable predictable from the 
independent variables 
For Classification Models: 
• Accuracy: The proportion of correct predictions 
• Confusion Matrix: Visualizes true versus predicted classes 
• Classification Report: Provides precision, recall, and F1-score for each class 
Cross-Validation 
To ensure robust model evaluation, k-fold cross-validation was implemented with k=4. This technique helps to 
assess how the model would perform on unseen data and reduces the risk of overfitting.
