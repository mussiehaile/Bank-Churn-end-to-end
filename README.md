Bank-Churn-end-to-end

==============================

This repository contains the code and resources for a machine learning project focused on predicting customer churn at a bank.
The goal of this project is to develop a predictive model that can identify customers who are likely to leave the bank, allowing the bank to take proactive measures to retain those customers.


------------
## Dataset

The dataset used for this project is stored in the  `src/data` directory. It consists of a collection of customer records, each with various attributes such as age, gender, account balance, credit score, etc., along with a binary label indicating whether the customer has churned or not.
   

--------

## Dependencies

To run the code in this repository, you need to have the following dependencies installed:

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook (optional, for running the provided notebooks)
evidently, etc ...

You can install the required dependencies using pip or conda.  you can run the following command:

`pip install -r requirements.txt`

---
## Code

The code for this project is organized using Cookicutter:

so we have 
1. `train_model.py` for training a model and experiment tracking using MLFlow and an orchestration tool using Prefect.
2. `deploy_model.py` file for deploying the model using Fastapi.
3. `prdict_model.py` for testing the deployed Fastapi server.
4.  `evidently_metrics_calculation.py` for calculating model drift,data drift, .. in the model prediction results to check the health of our model.

## Usage

To run this project, you can follow these steps:

1. Clone or download this repository to your local machine.
2. Install the required dependencies as mentioned in the Dependencies section.
3. Open and run the train_model.py file to train the model and experiment with it. Since we have set up MLflow and Prefect, feel free to experiment.
4. Once you have trained the model, deploy it by running the deploy_model.py file, and you can access it using the specified port. Then, run the predict_model.py file to test the model's performance.
5. Running the evidently_metrics_calculation.py file will retrieve the prediction response from the database, and you can visualize it using Grafana.

## Contributing

Contributions to this project are welcome. If you find any issues or have any suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). You are free to modify and use the code for both personal and commercial purposes.




