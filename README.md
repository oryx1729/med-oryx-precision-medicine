# MedOryx: Precision Medicine AI Agent

MedOryx is a Streamlit-based conversational AI agent designed to interact with the PrimeKG (Precision Medicine Knowledge Graph) hosted in a Neo4j database. It allows users to ask questions in natural language about diseases, drugs, genes, and other biomedical entities, and receive synthesized answers, including interactive graph visualizations. For more details, see the original publication: [Chandak et al., Nature Scientific Data (2023)](https://doi.org/10.1038/s41597-023-01960-3).


You will need to download the PrimeKG dataset and ingest it into your Neo4j instance as described in the setup steps below.

## Key Features

*   **Natural Language Queries:** Ask complex biomedical questions in plain English.
*   **Knowledge Graph Interaction:** Leverages the PrimeKG dataset stored in Neo4j.
*   **Haystack Agent:** Uses a Haystack AI agent with an Anthropic Claude model for query interpretation and Cypher generation.
*   **Interactive Visualizations:** Displays query results as interactive graphs using `streamlit-agraph`.
*   **Tool-Based Architecture:** Utilizes Neo4j MCP (Multi-Capability Proxy) tools for database interaction.

## Setup and Installation

1.  **Clone the repository (if applicable).**

2.  **Environment Variables:**
    Set the following environment variables.
    ```
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    NEO4J_URI="your_neo4j_uri" 
    NEO4J_USERNAME="your_neo4j_username"
    NEO4J_PASSWORD="your_neo4j_password"
    NEO4J_DATABASE="your_neo4j_database" 
    ```

2. **Install dependencies with uv:**
   ```
   uv pip install .
   ```

3. Download the PrimeKG dataset. <ADD CITATION>

4. Run the import script to ingest data to Neo4J.


## How to Run

1.  Run the Streamlit app with uv: `uv run streamlit run main.py`.
2.  Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).
