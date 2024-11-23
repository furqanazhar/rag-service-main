from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import numpy as np
import traceback

app = FastAPI()
es = Elasticsearch("http://elasticsearch:9200")


@app.get("/")
async def root():
    return {"detail": "Welcome to RAG Service!"}


def search_patents_by_company(company: str):
    """Search for patents associated with the given company."""
    s = Search(using=es, index="family_g1_v2") \
        .query("term", **{"members.best_standardized_name.name.keyword": company})
    return s.execute()


def extract_and_combine_embeddings(request):
    """Extract embeddings from the search response and compute their mean."""
    embeddings = [hit["embeddings_768_bgebase"] for hit in request if "embeddings_768_bgebase" in hit]
    return np.mean(embeddings, axis=0).tolist() if embeddings else None


def build_and_run_knn_query(combined_embedding, company):
    """Build the KNN query and run it to find similar patents."""
    knn_query = Q(
        "bool",
        must=[
            Q("exists", field="members.best_standardized_name"),
            Q("knn", field="embeddings_768_bgebase", query_vector=combined_embedding, num_candidates=300)
        ],
        must_not=[
            Q("term", **{"members.best_standardized_name.name.keyword": company})
        ]
    )
    s_knn = Search(using=es, index="family_g1_v2").query(knn_query)[:10]
    return s_knn.execute()


def get_competitor_names(knn_response):
    """Extract competitor names from KNN query results."""
    competitors = []
    for hit in knn_response:
        member = hit["members"][0]  # Process only the first member
        if "best_standardized_name" in member and len(member["best_standardized_name"]) > 0:
            competitors.append(member["best_standardized_name"][0]["name"])
    return competitors


def format_competitor_response(competitor_results, company):
    """Format the list of competitors into a natural language sentence."""
    # Remove duplicates while preserving order
    competitor_results = list(dict.fromkeys(filter(None, competitor_results)))
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

        # Step 2: Extract embeddings from the patents response & combine embeddings
        combined_embedding = extract_and_combine_embeddings(response.hits)
        if not combined_embedding:
            return {"detail": "No embeddings found for the given company."}

        # Step 4: Perform KNN query
        knn_response = build_and_run_knn_query(combined_embedding, company)

        # Step 5: Extract competitor names from KNN response
        competitor_results = get_competitor_names(knn_response.hits)

        # Step 6: Format and return the response
        response_message = format_competitor_response(competitor_results, company)
        return {"detail": response_message}

    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")
