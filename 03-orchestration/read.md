To run the python script it is necessary to:

1. Activate the enviroment:

`conda activate env-name`

2. Initialize mlflow at the port designed at set uri. To do so at local host port 5000:

`mlflow server -h 0.0.0.0 -p 5000`

3. Open a new terminal, initialize the enviroment and run the script:

`python duration-prediction.py --year=(desired year) --month=(desired_month)`