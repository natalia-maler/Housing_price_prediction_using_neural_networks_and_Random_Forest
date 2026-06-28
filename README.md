### Housing price prediction using neural_networks and Random Forest

The goal of this project was to build regression models predicting median house prices based on demographic and geographic features from the California Housing dataset.

The study compares:
- a simple neural network,
- an improved neural network with regularization techniques,
- a Random Forest model with hyperparameter tuning.

### Dataset Description

The dataset contains aggregated information about housing districts in California, including:
- median income of residents,
- median house age,
- total number of rooms,
- total number of bedrooms,
- population,
- number of households,
- geographic location: latitude and longitude,
- proximity to the ocean,
- target variable: median house value.

The dataset is a classic example of tabular structured data, where each row represents a geographical region rather than an individual house.

### Data Cleaning and Preprocessing
Before training the models, several preprocessing steps were applied:
- removal of records with capped target value (median_house_value = 500001), as these values are artificially truncated by the dataset creators and could bias model learning,
- handling of missing values, ensuring dataset completeness before training,
- conversion of categorical variable ocean_proximity using One-Hot Encoding, allowing machine learning models to process categorical information,
- conversion of boolean columns into integer format (0/1), ensuring compatibility with neural network inputs,
- feature scaling using StandardScaler, which normalizes features to zero mean and unit variance.
- the target variable - median house value was also scaled to improve neural network training stability.

Scaling and preprocessing were crucial because:
- neural networks are sensitive to feature scale,
- unscaled features can slow down convergence,
- large-value features may dominate learning,
- standardized inputs improve gradient stability and training speed.

### Model 1 - Simple Neural Network
Architecture:
- Dense(64, ReLU)
- Dense(32, ReLU)
- Dense(1)

Optimizer: Adam

Results:
- MSE: 0.2539
- MAE: 0.3516
- R^2: 0.7507

The baseline neural network achieved reasonable performance, explaining around 75% of variance in housing prices.

### Model 2 - Improved Neural Network
To improve performance and generalization, the following techniques were introduced:
- Batch Normalization,
- Dropout regularization,
- L2 regularization (weight decay),
- Early Stopping,
- learning rate tuning,

Architecture:
- Dense(128)
- BatchNormalization
- Dropout(0.3)
- Dense(64)
- BatchNormalization
- Dropout(0.2)
- Dense(32)
- BatchNormalization
- Dense(1)

Results:
- MSE: 0.2349
- MAE: 0.3175
- R^2: 0.7859

The improved model significantly reduced error and improved generalization ability.

### Model 3 - Random Forest with Hyperparameter tuning

A Random Forest Regressor was trained and optimized using RandomizedSearchCV.

Tuned parameters included: number of trees, maximum depth, minimum samples split, minimum samples per leaf, number of features considered per split.

Results:
- MSE: 0.2129
- MAE: 0.3147
- R^2: 0.7910

The Random Forest model achieves the best performance overall, slightly outperforming neural networks in all metrics.

This project demonstrates that:
- model performance improves with architectural and optimization enhancements,
- neural networks can reach strong performance on tabular data with proper tuning,
- however, Random Forest remains the strongest model for this dataset,
- achieving an R^2 of approximately 0.79, explaining 79% of variance in housing prices.
