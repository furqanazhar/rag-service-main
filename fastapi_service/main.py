from fastapi import FastAPI
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import numpy as np

app = FastAPI()
es = Elasticsearch("http://elasticsearch:9200")


@app.get("/")
async def root():
    return {"message": "Welcome to RAG Service!"}


def search_patents_by_company(company: str):
    """Search for patents associated with the given company."""
    s = Search(using=es, index="family_g1_v2") \
        .query("term", **{"members.best_standardized_name.name.keyword": company})
    return s.execute()


def extract_embeddings(response):
    """Extract embeddings from the search response."""
    return [hit.embeddings_768_bgebase for hit in response.hits if hasattr(hit, 'embeddings_768_bgebase')]


def combine_embeddings(embeddings):
    """Combine embeddings using mean."""
    if embeddings:
        return np.mean(np.array(embeddings), axis=0).tolist()
    return None


def build_knn_query(combined_embedding, company):
    """Build the KNN query to find similar patents."""
    return Q(
        "bool",
        must=[
            Q("exists", field="members.best_standardized_name"),
            Q("knn", field="embeddings_768_bgebase", query_vector=combined_embedding, num_candidates=300)
        ],
        must_not=[
            Q("term", **{"members.best_standardized_name.name.keyword": company})
        ]
    )


def get_competitor_names(knn_response):
    """Extract competitor names from KNN query results."""
    return [
        hit.members[0]["best_standardized_name"][0]["name"]
        for hit in knn_response.hits
        if hit.members and "best_standardized_name" in hit.members[0]
    ]


def format_competitor_response(competitor_results, company):
    """Format the list of competitors into a natural language sentence."""
    competitor_results = list(set(filter(None, competitor_results)))
    if competitor_results:
        if len(competitor_results) == 1:
            return f"The competitor of {company} is {competitor_results[0]}."
        else:
            formatted_list = ", ".join(competitor_results[:-1]) + f", and {competitor_results[-1]}"
            return f"The competitors of {company} are {formatted_list}."
    return f"There are no competitors found for {company}."


@app.get("/competitors/{company}")
async def get_competitors(company: str):
    try:
        # Step 1: Collect company's patents
        response = search_patents_by_company(company)

        # Step 2: Extract embeddings from the patents response
        embeddings = extract_embeddings(response)

        # Step 3: Combine embeddings
        combined_embedding = combine_embeddings(embeddings)
        if not combined_embedding:
            return {"message": "No embeddings found for the given company."}

        # Step 4: Perform KNN query
        knn_query = build_knn_query(combined_embedding, company)
        s_knn = Search(using=es, index="family_g1_v2").query(knn_query)[:10]
        knn_response = s_knn.execute()

        # Step 5: Extract competitor names from KNN response
        competitor_results = get_competitor_names(knn_response)

        # Step 6: Format and return the response
        response_message = format_competitor_response(competitor_results, company)
        return {"message": response_message}

    except Exception as e:
        return {"message": f"An error occurred: {e}"}
