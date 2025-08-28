import bentoml
import numpy as np

target_names = ['setosa', 'versicolor', 'virginica']

my_image = bentoml.images.Image(python_version="3.12") \
            .python_packages("mlflow", "scikit-learn")


@bentoml.service(
    image=my_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    # Declare the model as a class variable
    bento_model = bentoml.models.BentoModel("iris:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> list[str]:
        preds = self.model.predict(input_data)
        return [target_names[i] for i in preds]


