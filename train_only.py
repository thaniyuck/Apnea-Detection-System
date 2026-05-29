from src.ml_model import ApneaClassifier

def main():
    print("Starting Forced Retraining Sequence...")
    
    # Initialize the classifier
    classifier = ApneaClassifier(model_save_path="trained_rf_model.pkl")
    
    # FORCING the new apnea-ecg dataset
    dataset_path = "data/apnea_ecg_processed_features.csv"
    
    # Trigger the training
    success = classifier.train_model(csv_path=dataset_path)
    
    if success:
        print("Retraining complete. New .pkl file saved!")
    else:
        print("Retraining failed. Check dataset path.")

if __name__ == "__main__":
    main()