import bentoml

#  from bentoml.io import JSON
from typing import List, Tuple

#  from pydantic import BaseModel

#  embedding_runner = bentoml.mlflow.get("embedding:latest").to_runner()
#  svc = bentoml.Service("guided_content_embedding", runners=[embedding_runner])


bento_image = bentoml.images.Image(python_version="3.12").python_packages(
    "mlflow", "gensim"
)

#  class GuidedContentModel(BaseModel):
#      content: List[str]
#      top_n: int


@bentoml.service(image=bento_image, resources={"cpu": "2"}, traffic={"timeout": 10})
class GuidedContentEmbedding:
    bento_model = bentoml.models.BentoModel("embedding:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(route="/v1/vectorize")
    def vectorize(self, input_data: List[str]) -> List[float]:
        return self.model.predict({"action": "vectorize", "data": input_data})

    # note: 'batchable=True' only accept a single argument
    @bentoml.api(route="/v1/similar")
    def get_similar(self, input_data: List[str], top_n: int) -> List[Tuple[int, float]]:
        return self.model.predict(
            {"action": "most_similar", "data": input_data, "top_n": top_n}
        )
