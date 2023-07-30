from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from matplotlib import pyplot as plt 
from prefect import flow, task
from sklearn import metrics
import pandas as pd 
import joblib
import mlflow 
import os





@task(retries=2,name="process_data")
def process(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df.drop(columns=['customer_id','country','gender'],axis =1)
    
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Print the shapes of the arrays
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)
    
    return X_train, X_test, y_train, y_test




@task(retries=2,name="getting_metrics")
def model_metrics(actual, pred):
    
    if not os.path.exists('plot'):
        os.makedirs('plot')
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, threshold1 = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False positive rate', size=14)
    plt.ylabel('True positve Rate', size=14)
    plt.legend(loc='lower right')
    
    plt.savefig('plot/roc_curve.png')
    
    plt.close()
    
    return accuracy, f1, auc




@flow(name = 'mlflow_logs')
def mlflow_logs(model, X, y, name):
    
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('experiment_2')
    
    
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        mlflow.set_tag('run_id', run_id)
        
        pred = model.predict(X)
        
        accuracy, f1, auc = model_metrics(y, pred)
        
        mlflow.log_params(model.best_params_)
        mlflow.log_metric('Mean cv score', model.best_score_)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('f1-score', f1)
        mlflow.log_metric('AUC', auc)
        
        
        
        mlflow.log_artifact("plot/roc_curve.png")
        mlflow.sklearn.log_model(model, name)
        
       
        
        mlflow.end_run()





@flow(name = 'main_flow_entry point',description= 'this flow excutes 2 other functions',flow_run_name ='Churn1')
def main():
    # Example file path
    file_path = r'/home/mussie/Music/home projects/final proj/Bank-Churn-end-to-end/src/data/train_data.csv'

    # Process the data
    X_train, X_test, y_train, y_test = process(file_path)

    # Train DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=1)
    dt_param_grid = {
        'max_depth': [3, 5, 7, 9, 11, 13],
        'criterion': ['gini', 'entropy']
    }
    dt_gs = GridSearchCV(
        estimator=dt,
        param_grid=dt_param_grid,
        cv=5,
        n_jobs=1,
        scoring='accuracy',
        verbose=0
    )
    dt_model = dt_gs.fit(X_train, y_train)
    mlflow_logs(dt_model, X_test, y_test, 'DecisionTreeClassifierfinal')

    # Train RandomForestClassifier
    rf = RandomForestClassifier(random_state=1)
    rf_param_grid = {
        'n_estimators': [400, 700],
        'max_depth': [15, 20, 25],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [50, 100]
    }
    rf_gs = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=5,
        n_jobs=1,
        scoring='accuracy',
        verbose=0
    )
    rf_model = rf_gs.fit(X_train, y_train)
    mlflow_logs(rf_model, X_test, y_test, 'RandomForestClassifierfinal')




if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    