import bentoml

content_sample = """
once upon a time
"""

content_split = ["once", "upon", "a", "time"]


client = bentoml.SyncHTTPClient("http://localhost:3000")
v1 = client.vectorize(content_sample)
s1 = client.get_similar(content_sample, top_n=10)

v2 = client.vectorize(content_split)
s2 = client.get_similar(content_split, top_n=10)
