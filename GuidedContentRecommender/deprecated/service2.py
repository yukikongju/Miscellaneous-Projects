import bentoml

#  from bentoml.io import JSON
from typing import List, Tuple
from pydantic import BaseModel

bento_image = bentoml.images.Image(python_version="3.12").python_packages(
    "mlflow", "gensim"
)


class GuidedContentModel(BaseModel):
    content: List[str]


class SimilarModel(BaseModel):
    content: List[str]
    top_n: int


@bentoml.service(image=bento_image, resources={"cpu": "2"}, traffic={"timeout": 10})
class GuidedContentEmbedding:
    def __init__(self):
        self.model = bentoml.mlflow.load_model("embedding:latest")

    @bentoml.api(batchable=True)
    def vectorize(self, input_data: GuidedContentModel) -> List[float]:
        # return self.model._model_impl.python_model.model.infer_vector(input_data)
        return self.model.predict({"action": "vectorize", "data": input_data.content})
        #  return input_data

    @bentoml.api(batchable=True)
    def get_similar(self, input_data: SimilarModel) -> List[Tuple[int, float]]:
        #  def get_similar(self, input_data: List[str], top_n: int) -> List[Tuple[int, float]]:
        #  vec = self.vectorize(input_data)
        # return self.model._model_impl.python_model.model.dv.most_similar(vec, topn=top_n)
        #  return input_data
        return self.model.predict(
            {
                "action": "most_similar",
                "data": input_data.content,
                "top_n": input_data.top_n,
            }
        )

        #  return self.model.predict(
        #      {"action": "most_similar", "data": input_data, "top_n": top_n}
        #  )
