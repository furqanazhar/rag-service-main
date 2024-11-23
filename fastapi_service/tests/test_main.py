import fastapi_service.main as main
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
import numpy as np

client = TestClient(main.app)


@pytest.mark.parametrize(
    "company_name, mock_hits, expected_family_id",
    [
        ("Test Company 1", [{"family_id": "1"}], "1"),
        ("Test Company 2", [{"family_id": "2"}], "2"),
        ("Test Company 3", [], None),  # Test case with no hits
    ],
)
@patch("elasticsearch_dsl.Search.execute")
def test_search_patents_by_company(mock_execute, company_name, mock_hits, expected_family_id):
    # Mock response setup
    mock_response = mock_execute.return_value
    mock_response.hits = mock_hits

    # Call the function
    response = main.search_patents_by_company(company_name)

    # Assert the result
    if mock_hits:
        assert response.hits[0]["family_id"] == expected_family_id
    else:
        assert not response.hits  # Expect no hits for empty mock_hits

@pytest.mark.parametrize(
    "company_name, mock_hits, expected_result",
    [
        # Case 1: Valid embeddings in the hits
        (   "Test Company 1",
            [
                {"family_id": "1", "embeddings_768_bgebase": np.array([1.0, 2.0, 3.0])},
                {"family_id": "2", "embeddings_768_bgebase": np.array([4.0, 5.0, 6.0])},
            ],
            [2.5, 3.5, 4.5],
        ),
        # Case 2: Some hits without embeddings
        (   "Test Company 2",
            [
                {"family_id": "3", "embeddings_768_bgebase": np.array([1.0, 2.0, 3.0])},
                {"family_id": "4"},
            ],
            [1.0, 2.0, 3.0],
        ),
        # Case 3: No hits with embeddings
        (   "Test Company 3",
            [
                {"family_id": "5"},
                {"family_id": "6"},
            ],
            None,
        ),
        # Case 4: Empty response hits
        (   "Test Company 4",
            [],
            None,
        ),
    ],
)
def test_extract_and_combine_embeddings(company_name, mock_hits, expected_result):
    result = main.extract_and_combine_embeddings(mock_hits)
    assert result == expected_result


@pytest.mark.parametrize(
    "mock_hits, expected_result",
    [
        # Test Case 1: Valid response with multiple competitors
        (
            [
                {"members": [{"best_standardized_name": [{"name": "Competitor A"}]}]},
                {"members": [{"best_standardized_name": [{"name": "Competitor B"}]}]}
            ],
            ["Competitor A", "Competitor B"]
        ),
        # Test Case 2: Response with some members missing `best_standardized_name`
        (
            [
                {"members": [{"best_standardized_name": [{"name": "Competitor C"}]}]},
                {"members": [{"best_standardized_name": []}]}
            ],
            ["Competitor C"]
        ),
        # Test Case 3: Empty hits list
        (
            [],
            []
        ),
        # Test Case 4: Members with no members in `best_standardized_name`
        (
            [
                {"members": [{"best_standardized_name": []}]},
                {"members": [{"best_standardized_name": []}]},
            ],
            []
        ),
    ])
def test_get_competitor_names(mock_hits, expected_result):
    competitors = main.get_competitor_names(mock_hits)
    assert competitors == expected_result


@pytest.mark.parametrize(
    "competitors, company, expected_result",
    [
    # Test Case 1: Single competitor
    (["Competitor A"], "Test Company", "The competitor of Test Company is Competitor A."),

    # Test Case 2: Multiple competitors
    (["Competitor A", "Competitor B", "Competitor C"], "Test Company",
     "The competitors of Test Company are Competitor A, Competitor B, and Competitor C."),

    # Test Case 3: No competitors
    ([], "Test Company", "There are no competitors found for Test Company."),

    # Test Case 4: Duplicate competitors in input
    (["Competitor A", "Competitor A", "Competitor B"], "Test Company",
     "The competitors of Test Company are Competitor A, and Competitor B."),

    # Test Case 5: Competitor list contains None or empty strings
    (["Competitor A", None, ""], "Test Company", "The competitor of Test Company is Competitor A."),

    # Test Case 6: Competitor list with all empty values
    ([None, "", ""], "Test Company", "There are no competitors found for Test Company."),
])
def test_format_competitor_response(competitors, company, expected_result):
    response = main.format_competitor_response(competitors, company)
    assert response == expected_result


@pytest.mark.parametrize(
    "combined_embedding, company, mock_hits, expected_result",
    [
        (
            [0.1, 0.2, 0.3],
            "Test Company",
            [{"family_id": "123"}, {"family_id": "456"}],
            [{"family_id": "123"}, {"family_id": "456"}],
        ),
        (
            [0.5, 0.6, 0.7],
            "Another Company",
            [],
            []
        )
    ]
)
@patch("elasticsearch_dsl.Search.execute")
def test_knn_query_execution(mock_search_class, combined_embedding, company, mock_hits, expected_result):
    mock_response = mock_search_class.return_value
    mock_response.hits = mock_hits

    # Call the function to test
    actual_result = main.build_and_run_knn_query(combined_embedding, company)

    # Assert the result
    if mock_hits:
        assert actual_result.hits == expected_result
    else:
        assert not actual_result.hits  # Expect no hits for empty mock_hits


@pytest.mark.parametrize(
    "company, mock_patent_hits, mock_combined_embedding, mock_knn_hits, expected_status, expected_message",
    [
        # Case 1: Successful retrieval of competitors
        (
                "Test Company",
                [{"embedding": [0.1, 0.2, 0.3]}],
                [0.1, 0.2, 0.3],
                [{"members": [{"best_standardized_name": [{"name": "Competitor A"}]}]},
                 {"members": [{"best_standardized_name": [{"name": "Competitor B"}]}]}],
                200,
                "The competitors of Test Company are Competitor A, and Competitor B.",
        ),
        # Case 2: Successful retrieval of single competitor
        (
                "Test Company",
                [{"embedding": [0.1, 0.2, 0.3]}],
                [0.1, 0.2, 0.3],
                [{"members": [{"best_standardized_name": [{"name": "Competitor A"}]}]}],
                200,
                "The competitor of Test Company is Competitor A.",
        ),
        # Case 3: No competitors found
        (
                "Test Company",
                [{"embedding": [0.1, 0.2, 0.3]}],
                [0.1, 0.2, 0.3],
                [],
                200,
                "There are no competitors found for Test Company.",
        ),
        # Case 3: No embeddings found
        (
                "Test Company",
                [],
                None,
                [],
                200,
                "No embeddings found for the given company.",
        ),
        # Case 4: Exception during processing
        (
                "Test Company",
                None,
                None,
                None,
                400,
                "An error occurred: Error in search_patents_by_company",
        ),
    ],
)
@patch("fastapi_service.main.search_patents_by_company")
@patch("fastapi_service.main.extract_and_combine_embeddings")
@patch("fastapi_service.main.build_and_run_knn_query")
@patch("fastapi_service.main.get_competitor_names")
@patch("fastapi_service.main.format_competitor_response")
def test_get_competitors(
        mock_format_response,
        mock_get_competitor_names,
        mock_knn_query,
        mock_extract_embeddings,
        mock_search_patents,
        company,
        mock_patent_hits,
        mock_combined_embedding,
        mock_knn_hits,
        expected_status,
        expected_message,
):
    if mock_patent_hits is not None:
        mock_search_patents.return_value.hits = mock_patent_hits
    else:
        mock_search_patents.side_effect = Exception("Error in search_patents_by_company")

    mock_extract_embeddings.return_value = mock_combined_embedding
    mock_knn_query.return_value.hits = mock_knn_hits

    mock_get_competitor_names.return_value = [
        member["best_standardized_name"][0]["name"]
        for hit in (mock_knn_hits or [])
        for member in hit.get("members", [])
    ]

    mock_format_response.return_value = expected_message
    response = client.get(f"/competitors/{company}")
    assert response.status_code == expected_status
    assert expected_message == response.json()["detail"]


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
