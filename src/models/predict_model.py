import requests

url = "http://localhost:80/predict"

data = {
    "credit_score": 700,
    "age": 35,
    "tenure": 5,
    "balance": 10000.0,
    "products_number": 2,
    "credit_card": 1,
    "active_member": 1,
    "estimated_salary": 50000.0,
}

response = requests.post(url, json=data)
predictions = response.json()

print(predictions)
