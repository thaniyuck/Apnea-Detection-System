import os
import warnings
import joblib
import numpy as np
import pandas as pd
from collections import deque
import optuna  # NEW: Optuna for Bayesian Optimization

# Suppress the expected threshold sweep warnings & optuna terminal spam
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)

# =====================================================
# IMBALANCED-LEARN FOR SMOTE
# =====================================================
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class ApneaClassifier:
    """
    Advanced Random Forest Manager utilizing:
    - Optuna Bayesian Hyperparameter Tuning
    - SMOTE Data Synthesis (Zero Leakage CV)
    - Feature Pruning (Top 12 Features)
    - Temporal Derivative (SpO2 Regression Slope)
    - LOSO Cross Validation
    - Threshold Optimization (Balanced Accuracy)
    - ROC-AUC Analysis
    - Probability Distribution Analysis
    - Feature Importance Analysis
    - Confidence Zone Analysis
    - Temporal Prediction Smoothing
    - 3-Minute Rolling Context
    """

    def __init__(self, model_save_path="trained_rf_model.pkl"):
        self.model_save_path = model_save_path
        self.model = None
        self.top_features = None
        self.optimal_threshold = 0.5
        self.max_features_to_keep = 12  # FEATURE PRUNING LIMIT

        # =====================================================
        # TEMPORAL SMOOTHING & ROLLING CONTEXT BUFFERS
        # =====================================================
        self.prediction_buffer = deque(maxlen=5)
        self.spo2_history = deque(maxlen=3)
        self.pr_history = deque(maxlen=3)

    # =========================================================
    # MAIN TRAINING FUNCTION
    # =========================================================
    def train_model(self, csv_path="data/apnea_ecg_processed_features.csv"):
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}")
            return False

        print("Loading dataset...")
        df = pd.read_csv(csv_path)

        # =====================================================
        # DATA PREPROCESSING: TEMPORAL DERIVATIVE (SpO2 SLOPE)
        # =====================================================
        print("\nCalculating Temporal Derivatives (SpO2 Slope)...")
        df = df.sort_values(by=['Subject'])
        
        def calc_slope(vals):
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals))
            return np.polyfit(x, vals, 1)[0] # Linear regression slope

        # Inject SpO2 Slope calculation dynamically
        if 'SpO2 (%)' in df.columns:
            df['SpO2_Slope_3m'] = df.groupby('Subject')['SpO2 (%)'].transform(
                lambda x: x.rolling(3, min_periods=1).apply(calc_slope, raw=True)
            )

        # =====================================================
        # DATASET OVERVIEW
        # =====================================================
        print("\n" + "="*60)
        print(" DATASET OVERVIEW ")
        print("="*60)
        print(f"Total Samples : {len(df)}")

        class_counts = df['Label'].value_counts()
        print("\nClass Balance:")
        print(class_counts)
        print("\nClass Percentage:")
        print((class_counts / len(df)) * 100)
        print("="*60)

        # =====================================================
        # PREPARE DATA
        # =====================================================
        X_raw = df.drop(columns=['Label', 'Subject'])
        y = df['Label']
        groups = df['Subject']

        # =====================================================
        # SUBJECT DISTRIBUTION
        # =====================================================
        print("\nSubject Distribution:")
        subject_distribution = (
            df.groupby("Subject")["Label"]
            .value_counts()
            .unstack(fill_value=0)
        )
        print(subject_distribution)
        subject_distribution.to_csv("data/subject_distribution.csv")

        # =====================================================
        # FEATURE PRUNING (REMOVING NOISE)
        # =====================================================
        print(f"\nEvaluating {len(X_raw.columns)} raw features for pruning...")
        
        prune_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        prune_rf.fit(X_raw, y)
        importances = pd.Series(prune_rf.feature_importances_, index=X_raw.columns)
        
        # Keep only the top features
        self.top_features = importances.nlargest(self.max_features_to_keep).index.tolist()
        X_pruned = X_raw[self.top_features]

        print(f"\nPruned down to Top {self.max_features_to_keep} Features:")
        print(self.top_features)

        # =====================================================
        # LOSO
        # =====================================================
        logo = LeaveOneGroupOut()

        # =====================================================
        # OPTUNA BAYESIAN OPTIMIZATION WITH SMOTE PIPELINE
        # =====================================================
        print("\nRunning Optuna Bayesian Optimization (Searching 20 intelligent trials)...")

        def objective(trial):
            # Optuna dynamically suggests parameters within these ranges
            rf_n_estimators = trial.suggest_int('n_estimators', 50, 300)
            rf_max_depth = trial.suggest_categorical('max_depth', [None, 10, 15, 20, 25])
            rf_min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

            smote = SMOTE(random_state=42)
            rf = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            
            pipeline = ImbPipeline([
                ('smote', smote),
                ('rf', rf)
            ])

            # Evaluate using LOSO Cross Validation
            scores = cross_val_score(
                pipeline, X_pruned, y, 
                groups=groups, cv=logo, scoring='balanced_accuracy', n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        print(f"\nBest Optuna Parameters Found: {study.best_params}")

        # =====================================================
        # SAVE OPTUNA TRIAL RESULTS
        # =====================================================
        results_df = study.trials_dataframe()
        results_df.to_csv("data/parameter_comparison_sheet.csv", index=False)
        print("Parameter comparison saved.")

        # =====================================================
        # BEST MODEL
        # =====================================================
        print("\nGenerating LOSO predictions...")

        # Build the final master pipeline using Optuna's best parameters
        best_smote = SMOTE(random_state=42)
        best_rf = RandomForestClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_split=study.best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        self.model = ImbPipeline([
            ('smote', best_smote),
            ('rf', best_rf)
        ])

        y_prob_cv = cross_val_predict(
            self.model,
            X_pruned,
            y,
            groups=groups,
            cv=logo,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]

        # =====================================================
        # PROBABILITY ANALYSIS
        # =====================================================
        print("\n" + "="*60)
        print(" PROBABILITY ANALYSIS ")
        print("="*60)
        print("\nOverall Distribution:")
        print(pd.Series(y_prob_cv).describe())
        
        print("\nNormal Distribution:")
        print(pd.Series(y_prob_cv[y == 0]).describe())
        
        print("\nApnea Distribution:")
        print(pd.Series(y_prob_cv[y == 1]).describe())

        # =====================================================
        # ROC AUC
        # =====================================================
        roc_auc = roc_auc_score(y, y_prob_cv)

        print("\n" + "="*60)
        print(" ROC-AUC ")
        print("="*60)
        print(f"ROC-AUC : {roc_auc:.4f}")

        if roc_auc < 0.60:
            print("Weak separation")
        elif roc_auc < 0.70:
            print("Moderate separation")
        elif roc_auc < 0.80:
            print("Good separation")
        else:
            print("Excellent separation")
        print("="*60)

        # =====================================================
        # THRESHOLD OPTIMIZATION (BALANCED ACCURACY)
        # =====================================================
        print("\nOptimizing thresholds...")
        thresholds = np.arange(0.10, 0.91, 0.05)
        threshold_results = []
        
        best_balanced_acc = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_prob_cv >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            accuracy = accuracy_score(y, y_pred)
            balanced_acc = balanced_accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y, y_pred)

            # ================================================
            # CONFIDENCE ZONE COUNTS
            # ================================================
            low_confidence = np.sum((y_prob_cv >= 0.30) & (y_prob_cv < 0.50))
            high_risk = np.sum(y_prob_cv >= 0.50)

            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1_score': f1,
                'mcc': mcc,
                'low_confidence_windows': low_confidence,
                'high_risk_windows': high_risk
            })

            print(
                f"Threshold={threshold:.2f} | "
                f"ACC={accuracy:.4f} | "
                f"BAL_ACC={balanced_acc:.4f} | "
                f"SENS={sensitivity:.4f} | "
                f"SPEC={specificity:.4f}"
            )

            # ================================================
            # BEST BALANCED ACCURACY
            # ================================================
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_threshold = threshold

        # =====================================================
        # STORE THRESHOLD
        # =====================================================
        self.optimal_threshold = best_threshold

        # =====================================================
        # SAVE THRESHOLD RESULTS
        # =====================================================
        threshold_df = pd.DataFrame(threshold_results)
        threshold_df = threshold_df.sort_values(by='balanced_accuracy', ascending=False)
        threshold_df.to_csv("data/threshold_comparison_sheet.csv", index=False)
        print("\nThreshold comparison saved.")

        # =====================================================
        # TOP THRESHOLDS
        # =====================================================
        print("\n" + "="*60)
        print(" TOP THRESHOLDS (By Balanced Accuracy) ")
        print("="*60)
        print(
            threshold_df[
                ['threshold', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'mcc']
            ].head(5)
        )

        # =====================================================
        # FINAL EVALUATION
        # =====================================================
        y_final = (y_prob_cv >= self.optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_final).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y, y_final)
        balanced_acc = balanced_accuracy_score(y, y_final)
        precision = precision_score(y, y_final, zero_division=0)
        recall = recall_score(y, y_final, zero_division=0)
        f1 = f1_score(y, y_final, zero_division=0)
        mcc = matthews_corrcoef(y, y_final)

        # =====================================================
        # FEATURE IMPORTANCE (EXTRACTED FROM PIPELINE)
        # =====================================================
        print("\nCalculating feature importance...")
        self.model.fit(X_pruned, y)

        feature_importance_df = (
            pd.DataFrame({
                'feature': X_pruned.columns,
                'importance': self.model.named_steps['rf'].feature_importances_
            })
            .sort_values(by='importance', ascending=False)
        )

        feature_importance_df.to_csv("data/feature_importance.csv", index=False)
        print("\nTop Features Used in Model:")
        print(feature_importance_df.head(self.max_features_to_keep))

        # =====================================================
        # CONFIDENCE ZONES
        # =====================================================
        normal_zone = np.sum(y_prob_cv < 0.30)
        suspicious_zone = np.sum((y_prob_cv >= 0.30) & (y_prob_cv < 0.50))
        high_risk_zone = np.sum(y_prob_cv >= 0.50)

        # =====================================================
        # FINAL RESULTS
        # =====================================================
        print("\n" + "="*60)
        print(" BEST MODEL FOUND & DEPLOYED ")
        print("="*60)
        print(f"Optimal Threshold    : {self.optimal_threshold:.2f}")
        print(f"\nROC-AUC              : {roc_auc:.4f}")
        print(f"Accuracy             : {accuracy * 100:.2f}%")
        print(f"Balanced Accuracy    : {balanced_acc * 100:.2f}%")
        print(f"Sensitivity          : {sensitivity * 100:.2f}%")
        print(f"Specificity          : {specificity * 100:.2f}%")
        print(f"Precision            : {precision * 100:.2f}%")
        print(f"Recall               : {recall * 100:.2f}%")
        print(f"F1 Score             : {f1:.4f}")
        print(f"MCC                  : {mcc:.4f}")

        print("\nCONFIDENCE ZONES")
        print(f"Normal Zone (<0.30)       : {normal_zone}")
        print(f"Suspicious Zone (0.30-0.50): {suspicious_zone}")
        print(f"High Risk Zone (>0.50)    : {high_risk_zone}")
        print("="*60)

        # =====================================================
        # SAVE MODEL
        # =====================================================
        joblib.dump({
            'model': self.model,
            'top_features': self.top_features,
            'optimal_threshold': self.optimal_threshold
        }, self.model_save_path)

        print(f"\nModel saved to: {self.model_save_path}")
        return True

    # =========================================================
    # LOAD MODEL
    # =========================================================
    def load_model(self):
        if not os.path.exists(self.model_save_path):
            return False

        saved_data = joblib.load(self.model_save_path)

        if isinstance(saved_data, dict):
            self.model = saved_data['model']
            self.top_features = saved_data['top_features']
            self.optimal_threshold = saved_data.get('optimal_threshold', 0.5)
        else:
            self.model = saved_data
            self.top_features = None
            self.optimal_threshold = 0.5

        print(f"Model loaded from {self.model_save_path}")
        print(f"Loaded Threshold: {self.optimal_threshold}")
        return True

    # =========================================================
    # LIVE PREDICTION
    # =========================================================
    def predict_live_window(self, prv_dict, mse_list):
        if self.model is None:
            return None

        # =====================================================
        # BUILD FEATURE ROW & ROLLING CONTEXT
        # =====================================================
        row = {**prv_dict}
        
        # Unpack MSE List
        for j, entropy_val in enumerate(mse_list):
            row[f'd_{j+1}'] = entropy_val

        # Update Rolling Histories (defaults applied if missing from dictionary)
        current_spo2 = prv_dict.get('SpO2 (%)', 97.0)
        current_pr = prv_dict.get('PR (BPM)', 60.0)
        
        self.spo2_history.append(current_spo2)
        self.pr_history.append(current_pr)

        # Calculate live SpO2 Delta
        if len(self.spo2_history) > 1:
            row['SpO2_Delta'] = self.spo2_history[-1] - self.spo2_history[-2]
        else:
            row['SpO2_Delta'] = 0.0

        # Calculate live 3-min SpO2 Min
        row['SpO2_Rolling_Min_3m'] = min(self.spo2_history)

        # Calculate live 3-min Heart Rate Standard Deviation
        if len(self.pr_history) > 1:
            row['PR_Rolling_Std_3m'] = np.std(self.pr_history, ddof=1)
        else:
            row['PR_Rolling_Std_3m'] = 0.0

        # =====================================================
        # CALCULATE LIVE TEMPORAL DERIVATIVE (SLOPE)
        # =====================================================
        if len(self.spo2_history) > 1:
            y_vals = list(self.spo2_history)
            x_vals = np.arange(len(y_vals))
            row['SpO2_Slope_3m'] = np.polyfit(x_vals, y_vals, 1)[0]
        else:
            row['SpO2_Slope_3m'] = 0.0

        live_df = pd.DataFrame([row])

        # =====================================================
        # APPLY FEATURE ORDER & FILTERING (PRUNING)
        # =====================================================
        if self.top_features:
            for feature in self.top_features:
                if feature not in live_df.columns:
                    live_df[feature] = 0.0
            live_df = live_df[self.top_features]

        # =====================================================
        # GET PROBABILITY
        # =====================================================
        prob = self.model.predict_proba(live_df)[0][1]

        # =====================================================
        # CONFIDENCE ZONES
        # =====================================================
        if prob < 0.30:
            confidence_label = "NORMAL"
        elif prob < 0.50:
            confidence_label = "SUSPICIOUS"
        else:
            confidence_label = "HIGH RISK"

        # =====================================================
        # RAW PREDICTION
        # =====================================================
        raw_prediction = 1 if prob >= self.optimal_threshold else 0

        # =====================================================
        # TEMPORAL SMOOTHING
        # =====================================================
        self.prediction_buffer.append(raw_prediction)
        smoothed_prediction = int(round(np.mean(self.prediction_buffer)))

        # =====================================================
        # DEBUG OUTPUT
        # =====================================================
        print("\n" + "="*50)
        print(f"Apnea Probability : {prob:.4f}")
        print(f"Confidence Zone   : {confidence_label}")
        print(f"Threshold Used    : {self.optimal_threshold:.2f}")
        print(f"Raw Prediction    : {raw_prediction}")
        print(f"Smoothed Prediction: {smoothed_prediction}")
        print(f"Prediction Buffer : {list(self.prediction_buffer)}")
        print("="*50)

        # =====================================================
        # FINAL OUTPUT
        # =====================================================
        return smoothed_prediction