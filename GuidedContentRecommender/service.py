import bentoml

#  from bentoml.io import JSON
from typing import List

bento_image = bentoml.images.PythonImage(python_version="3.12").python_packages(
    "mlflow", "gensim"
)


@bentoml.service(
    image=bento_image,
    resources={"cpu": "2"},
    # traffic={"timeout": 10}
)
class GuidedContentEmbedding:
    def __init__(self):
        self.model = bentoml.mlflow.load_model("embedding:latest")

    @bentoml.api(batchable=True)
    def vectorize(self, input_data: List[str]) -> List[str]:
        # return self.model._model_impl.python_model.model.infer_vector(input_data)
        return self.model.predict({"action": "vectorize", "data": input_data})

    @bentoml.api(batchable=True)
    def get_similar(self, input_data: List[str], top_n: int) -> List[str]:
        #  vec = self.vectorize(input_data)
        # return self.model._model_impl.python_model.model.dv.most_similar(vec, topn=top_n)
        return self.model.predict(
            {"action": "most_similar", "data": input_data, "top_n": top_n}
        )
