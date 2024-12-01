from classes.classifier import Classifier

if __name__ == "__main__":
    # Create Model Object
    operation = 1
    train_test_operation = 0
    
    # Train
    if operation == 0:
        model = Classifier(pretrained=False,data_dir="data")
        model.train(15) # Epoch Amount
        
        if train_test_operation == 0:
            print(model.predict("test_image"))
    # Test
    else:
        model = Classifier(True,"your_model_path")
        

    
    
    
    