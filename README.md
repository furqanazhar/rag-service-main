# Retrieval Augmented Generation (RAG) Case
At Focus we extract strategic business intelligence from patent data. Traditional method to gather such insights would require keyword search to find the relevant information. The problem with keyword search is that you miss out on important data when you miss a relevant keyword. Patents generally include many synonyms. This means that if you are missing a synonym of a word, you might miss important business intelligence.

Recent advances in Generative AI has driven a different type of searching: vector search. Vector search tries to find close data points in a high dimensional space. Closer points are considered to be better matches, whereas distance points are considered poor matches. The advantage of vector search is that you two texts can be written completely differently while their respective vectors are close to each other. In other words, vectors represent the underlying meaning/concept of the text. Embedding models have been created to transform texts into vectors. Better models produce better vectors representing the underlying concept of the text.

Vector search and Generative AI is a perfect match together. Recently RAG's have gotten a lot of attention. The idea of a RAG is that you combine a database with a Generative AI model. For example, you ask a question, the Generative AI model transforms this into a query sentence. The query is transformed into an embedding, and a vector search query is being performed against the database. The results from the database are then being used in the Generative AI model to answer your original question.

In this case you will be setting up a simple API service which implements some type of RAG. The service needs to do the following:
- Answer the question: Who are the competitors of company X?
- This can be achieved by implementing some important key components:
  - Collect some patents for company X and their respective vectors.
  - Find similar patents (with different owners than company X) to the collected patent vectors, and collect all organizations owning those patents.
  - Reinsert this list of owners into a Generative AI response to the user to answer the user's question.
- Stream the result back to the requester.
- Make sure the most important components of the service are covered by tests.

Furthermore:
- If you have any questions, do not hesitate to contact me via email or WhatsApp.
- If you make any assumptions, please note them down clearly.

Good luck!

## Resources
We have set up some necessities for implementing the service. The resources provided are:
- Elasticsearch and Kibana
- Embedding Model

You can bring up these resources with:
```shell
docker-compose up --build
```

**Elasticsearch**
The Elasticsearch service will be seeded with some sample data. This includes an index with patent families. Patent families are collections of patents representing the same IP claim, but in different countries. Practically this means that patents within the same family are near duplicates. Furthermore, there is much more data seeded than you would need for this case. You may use whatever data you need. One thing that is important to note is that the term `best_standardized_name` refers to the organization owning the patent. Furthermore, the field `embeddings_768_bgebase` represents the vector for that family.

You can check if the Elasticsearch cluster is running using:
```shell
curl http://localhost:9200/_cluster/health
```

**Embedding Model**
...
The embedding model used to generate the patent family embeddings is also added to this case. It is an open source model and implemented in PyTorch. 

You can test if it works using:
```shell
curl -H "Content-type:application/json" -d '{"instances":["Some sentence"]}' http://localhost:8601/embedding-model/predict
```

**Unit Tests**
This project includes unit tests to ensure the reliability and correctness of the code. Unit tests are written using the pytest framework.

You can run tests using below command
```shell
pytest fastapi_service/tests/
```