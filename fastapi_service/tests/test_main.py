import fastapi_service.main as main
from unittest.mock import patch, MagicMock
from elasticsearch_dsl import Q
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

    if expected_result is None:
        assert result is None, f"Expected None, but got {result} for {company_name}"
    else:
        np.testing.assert_array_almost_equal(result, expected_result, err_msg=f"Failed for {company_name}")


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


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
