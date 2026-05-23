import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

class ApneaClassifier:
    """
    Advanced Random Forest Manager utilizing LOSO Cross-Validation, 
    Feature Pruning, and Hyperparameter Grid Search.
    """
    def __init__(self, model_save_path="trained_rf_model.pkl"):
        self.model_save_path = model_save_path
        self.model = None
        self.top_features = None # Stores the names of the features kept after pruning

    def _prune_features(self, X, y, prune_count=5):
        """
        Trains a quick baseline model to rank mathematical features by importance,
        then drops the weakest 'prune_count' features to reduce noise.
        """
        print(f"Analyzing feature importance to prune the weakest {prune_count} dimensions...")
        baseline_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        baseline_rf.fit(X, y)
        
        # Map importances to feature names
        importances = pd.Series(baseline_rf.feature_importances_, index=X.columns)
        
        # Identify the lowest performing features
        weakest_features = importances.nsmallest(prune_count).index.tolist()
        print(f"Dropped Features: {weakest_features}")
        
        # Keep the rest
        self.top_features = importances.nlargest(len(X.columns) - prune_count).index.tolist()
        return X[self.top_features]

    def train_model(self, csv_path="data/mit_bih_processed_features.csv"):
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}.")
            return False

        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # 1. Separate Features, Labels, and Subjects (for LOSO)
        X_raw = df.drop(columns=['Label', 'Subject'])
        y = df['Label']
        groups = df['Subject']
        
        # 2. Feature Pruning
        X_pruned = self._prune_features(X_raw, y, prune_count=5)
        
        # 3. Setup Leave-One-Subject-Out CV
        logo = LeaveOneGroupOut()
        
        # 4. Define the Grid Search Parameter Space
        param_grid = {
            'n_estimators': [50, 100, 200],       # Number of trees
            'max_depth': [None, 10, 20],          # Maximum depth of each tree
            'min_samples_split': [2, 5, 10]       # Min samples needed to split a node
        }
        
        print("\nInitiating GridSearchCV with Leave-One-Subject-Out CV...")
        print("This will train dozens of models. Please wait...")
        
        rf = RandomForestClassifier(random_state=42)
        
        # We use accuracy as the primary tuning metric, but it will calculate across all subjects
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=logo,
            scoring='accuracy',
            n_jobs=-1, # Uses all available CPU cores to speed up training
            return_train_score=True
        )
        
        # Execute the massive grid search
        grid_search.fit(X_pruned, y, groups=groups)
        
        # 5. Export the Comparison Sheet
        results_df = pd.DataFrame(grid_search.cv_results_)
        # Clean up the output sheet for better readability
        columns_to_keep = ['param_max_depth', 'param_min_samples_split', 'param_n_estimators', 'mean_test_score', 'std_test_score', 'rank_test_score']
        results_df = results_df[columns_to_keep].sort_values(by='rank_test_score')
        
        sheet_path = "data/parameter_comparison_sheet.csv"
        results_df.to_csv(sheet_path, index=False)
        print(f"\n✅ Parameter Comparison Sheet saved to: {sheet_path}")
        
        # 6. Extract and Evaluate the Absolute Best Model
        self.model = grid_search.best_estimator_
        
        # Calculate final clinical metrics on the whole dataset using the best model
        y_pred = self.model.predict(X_pruned)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0    
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0    
        accuracy = accuracy_score(y, y_pred)               
        
        print("\n" + "="*50)
        print(" BEST MODEL FOUND & DEPLOYED ")
        print("="*50)
        print(f" Best Parameters   : {grid_search.best_params_}")
        print(f" Final Accuracy    : {accuracy * 100:.2f}%")
        print(f" Final Sensitivity : {sensitivity * 100:.2f}%")
        print(f" Final Specificity : {specificity * 100:.2f}%")
        print("="*50)
        
        # Save both the model AND the list of kept features so live inference knows what to drop
        joblib.dump({'model': self.model, 'top_features': self.top_features}, self.model_save_path)
        return True

    def load_model(self):
        if os.path.exists(self.model_save_path):
            saved_data = joblib.load(self.model_save_path)
            # Support backward compatibility if an old pure-model pkl is loaded
            if isinstance(saved_data, dict):
                self.model = saved_data['model']
                self.top_features = saved_data['top_features']
            else:
                self.model = saved_data
                self.top_features = None # Will crash if feature counts mismatch, requires retraining
            print(f"Model loaded from {self.model_save_path}")
            return True
        return False

    def predict_live_window(self, prv_dict, mse_list):
        if self.model is None:
            return None

        # Build the full 23-feature row
        row = {**prv_dict}
        for j, entropy_val in enumerate(mse_list):
            row[f'd_{j+1}'] = entropy_val
            
        live_df = pd.DataFrame([row])
        
        # Automatically prune the weakest features from the live stream
        if self.top_features:
            live_df = live_df[self.top_features]
        
        return self.model.predict(live_df)[0]