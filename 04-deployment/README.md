## Deploying a model as a web-service

* Creating a virtual environment with pipenv, i. e `pipenv install scikit-learn==1.0.2 numpy==1.22.4 flask --python=3.9`
* Creating a script for prediction
* Putting the script into a Flask app
* Packaging the app to Docker

```bash
docker build -t ride-duration-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```