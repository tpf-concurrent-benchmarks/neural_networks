from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    df = pd.read_csv("housing.csv")
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    X_train = pd.DataFrame(X_train, columns=X_train.columns)
    X_train["median_house_value"] = y_train
    
    X_test = pd.DataFrame(X_test, columns=X_test.columns)
    X_test["median_house_value"] = y_test
    
    X_val = pd.DataFrame(X_val, columns=X_val.columns)
    X_val["median_house_value"] = y_val

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)
    # X_train.to_csv("train.csv", index=False)
    # X_test.to_csv("test.csv", index=False)
    # X_val.to_csv("validation.csv", index=False)

if __name__ == "__main__":
    main()