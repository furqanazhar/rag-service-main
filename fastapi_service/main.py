from fastapi import FastAPI
from elasticsearch import Elasticsearch

app = FastAPI()
es = Elasticsearch("http://elasticsearch:9200")


@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}


@app.get("/patents/{company}")
async def get_patents(company: str):
    response = es.search(index="family_g1_v2",
                         query={"term":
                                    {"members.best_standardized_name.name.keyword": company}
                                }
                         )
    result = []
    for hit in response["hits"]["hits"]:
        result.append(
            {"family": hit["_source"]["family_id"],
             "members": hit["_source"]["members"],
             "embedding": hit["_source"]["embeddings_768_bgebase"]
             }
        )
    return result
