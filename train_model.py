### train_model.py
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
df = pd.read_csv('house_prices.csv')
TARGET = 'Price (in rupees)'  # ganti sesuai nama kolom target
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# 2. Split train–test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Option 2: If it makes sense, fill with a value (like 0 or the mode)
y_train = y_train.fillna(y_train.mode()[0])

# 3. Identify feature types
num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# 4. Preprocess numeric: median impute + scale
num_imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train_num = scaler.fit_transform(num_imputer.fit_transform(X_train[num_cols]))
X_test_num  = scaler.transform(          num_imputer.transform(X_test[num_cols]))

# 5. Preprocess categorical: mode impute + ordinal encoding
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
X_test_cat  = cat_imputer.transform(X_test[cat_cols])
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_cat = ord_enc.fit_transform(X_train_cat)
X_test_cat  = ord_enc.transform(X_test_cat)

# 6. Combine features
X_train_enc = np.hstack([X_train_num, X_train_cat])
X_test_enc  = np.hstack([X_test_num,  X_test_cat])

# 7. Train Random Forest
tree = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
tree.fit(X_train_enc, y_train)

# 8. Save artifacts
joblib.dump({
    'model': tree,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'num_imputer': num_imputer,
    'scaler': scaler,
    'cat_imputer': cat_imputer,
    'ord_enc': ord_enc
}, 'house_price_rf.pkl')