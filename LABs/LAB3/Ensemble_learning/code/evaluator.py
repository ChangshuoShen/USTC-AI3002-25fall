from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_test, y_test):
    try:
        pred = model.predict(X_test)
        
        # 检查预测结果
        if pred is None:
            print("Error: model.predict returned None")
            return 0, 0
            
        print(f"Prediction shape: {pred.shape}")
        print(f"Prediction type: {type(pred)}")
        print(f"y_test shape: {y_test.shape}")
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        return acc, f1
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0, 0

    return acc, f1
