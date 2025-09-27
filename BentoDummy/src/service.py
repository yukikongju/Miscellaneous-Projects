import bentoml
import numpy as np
import numpy.typing as npt
from typing import Annotated, List
from bentoml.validators import Shape, DType

target_names = ["setosa", "versicolor", "virginica"]

my_image = bentoml.images.Image(python_version="3.12").python_packages(
    "mlflow", "scikit-learn"
)


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
    def predict(
        self,
        input_data: Annotated[
            npt.NDArray[np.float64], Shape((-1, 4)), DType("float64")
        ] = bentoml.Field(default=[[0.1, 0.4, 0.2, 1.0]]),
    ) -> List[str]:
        preds = self.model.predict(input_data)
        return [target_names[i] for i in preds]
