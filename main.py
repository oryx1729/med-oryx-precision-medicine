import streamlit as st
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack.dataclasses import ChatMessage
import os
import nest_asyncio
import logging
import json
import traceback
import pandas as pd
import re
from streamlit_agraph import agraph, Node, Edge, Config

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Haystack loggers to ERROR level to reduce noise
logging.getLogger('haystack').setLevel(logging.ERROR)
logging.getLogger('haystack_integrations').setLevel(logging.ERROR)

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# --- START HELPER FUNCTION FOR GRAPH RENDERING ---
def render_graph_from_message_content(content_text, is_expanded=True):
    """Parses graph JSON from message content and renders it using streamlit-agraph."""
    text_for_markdown = content_text
    graph_json_str = None
    graph_block_start_match = re.search(r"<graph_visualization>", content_text)
    graph_rendered = False

    if graph_block_start_match:
        text_for_markdown = content_text[:graph_block_start_match.start()].strip()
        potential_json_payload = content_text[graph_block_start_match.end():]
        graph_block_end_match = re.search(r"</graph_visualization>", potential_json_payload)
        
        if graph_block_end_match:
            graph_json_str = potential_json_payload[:graph_block_end_match.start()].strip()
        else:
            graph_json_str = potential_json_payload.strip() # Assume truncation
    
    if graph_json_str:
        try:
            graph_data = json.loads(graph_json_str)
            agraph_nodes_data = graph_data.get("nodes", [])
            agraph_edges_data = graph_data.get("edges", [])

            agraph_nodes = [
                Node(id=n.get("id"), label=n.get("label"), title=n.get("title"), 
                     group=n.get("group"), size=n.get("size", 15), 
                     shape=n.get("shape", "dot"), color=n.get("color")) 
                for n in agraph_nodes_data
            ]
            agraph_edges = [
                Edge(source=e.get("source"), target=e.get("target"), label=e.get("label")) 
                for e in agraph_edges_data
            ]
            
            legend_items = {}
            for node_data in agraph_nodes_data:
                group = node_data.get("group")
                color = node_data.get("color")
                if group and color and group not in legend_items:
                    legend_items[group] = color

            if agraph_nodes:
                agraph_config = Config(
                    width=1200, height=800, directed=True,
                    physics={
                        "enabled": True,
                        "barnesHut": {
                            "gravitationalConstant": -30000, "centralGravity": 0.1,
                            "springLength": 120, "springConstant": 0.05,
                            "damping": 0.09, "avoidOverlap": 0.5
                        },
                        "stabilization": {"iterations": 1500, "fit": True}
                    },
                    hierarchical=False,
                    edges={
                        "font": {"size": 10, "align": "middle"},
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                        "smooth": {"type": "continuous", "roundness": 0.2}
                    },
                    nodes={"font": {"size": 12, "strokeWidth": 2, "strokeColor": "white"}},
                    interaction={"hover": True, "tooltipDelay": 200, "dragNodes": True, "zoomView": True},
                    manipulation={"enabled": False}
                )
                with st.expander("🔍 View Interactive Graph", expanded=is_expanded):
                    agraph(nodes=agraph_nodes, edges=agraph_edges, config=agraph_config)
                    if legend_items:
                        st.markdown("**Node Legend:**")
                        legend_html_parts = [
                            f'''<div style="display: flex; align-items: center; margin-bottom: 5px;">
                                   <div style="width: 15px; height: 15px; background-color: {color}; margin-right: 8px; border: 1px solid #ccc;"></div>
                                   <span>{group.capitalize()}</span>
                               </div>''' for group, color in legend_items.items()
                        ]
                        st.markdown("\n".join(legend_html_parts), unsafe_allow_html=True)
                graph_rendered = True
            elif not text_for_markdown:
                text_for_markdown = "Graph data was found, but it contained no nodes to display."
            
            if not graph_rendered and not agraph_nodes and agraph_edges and not text_for_markdown:
                text_for_markdown = "Graph data was found, but it contained edges without corresponding nodes."

        except json.JSONDecodeError:
            logger.error(f"Failed to parse graph JSON (likely due to truncation or malformed content): {graph_json_str}")
            st.error("Failed to parse graph data. The data might be incomplete or malformed. Check console logs.")
            if not text_for_markdown: text_for_markdown = "Tried to render a graph, but data format was invalid/incomplete."
        except Exception as graph_ex:
            logger.error(f"Error rendering graph: {graph_ex}\n{traceback.format_exc()}")
            st.error(f"An error occurred displaying the graph: {str(graph_ex)}. Check console logs.")
            if not text_for_markdown: text_for_markdown = "An error occurred displaying the graph."
    
    return text_for_markdown, graph_rendered
# --- END HELPER FUNCTION FOR GRAPH RENDERING ---

# Set page config and initialize chat history
st.set_page_config(page_title="MedOryx", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for tool invocations
if "tool_invocations" not in st.session_state:
    st.session_state.tool_invocations = []

# Function to initialize the Haystack pipeline
@st.cache_resource
def initialize_pipeline():
    # Check for Anthropic API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        st.error("ANTHROPIC_API_KEY environment variable not set.")
        st.stop()

    # Check for Neo4j connection environment variables
    required_neo4j_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_neo4j_vars if not os.environ.get(var)]
    if missing_vars:
        st.error(f"The following Neo4j environment variables are not set: {', '.join(missing_vars)}. "
                 "Please set them (e.g., NEO4J_URI=\"neo4j://localhost:7687\", NEO4J_USERNAME=\"neo4j\", NEO4J_PASSWORD=\"password\"). "
                 "Optionally, also set NEO4J_DATABASE if you use a non-default database.")
        st.stop()
    
    # Initialize MCP tools for Neo4j
    current_env = os.environ.copy()
    # The mcp-neo4j-cypher server will use NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, 
    # and optionally NEO4J_DATABASE from its environment.
    server_info = StdioServerInfo(
        command="uv",  # Assuming 'uv' is your preferred runner and mcp-neo4j-cypher is installed as a package
        args=["run", "mcp-neo4j-cypher"],
        # If mcp-neo4j-cypher is a direct command in PATH (e.g., after pip install), you might use:
        # command="mcp-neo4j-cypher",
        # args=[], 
        env=current_env
    )
    mcp_toolset = MCPToolset(server_info) 
    
    # Specify the tool names we want to make available to the agent
    # We are selecting the two query tools and excluding the write tool.
    enabled_tool_names = ["read_neo4j_cypher", "get_neo4j_schema"]
    
    active_agent_tools = [
        tool_obj  # Select the tool object itself
        for tool_obj in mcp_toolset.tools  # Iterate assuming mcp_toolset.tools is a list of Tool objects
        if tool_obj.name in enabled_tool_names # Filter by the name attribute of each tool object
    ]

    # Initialize the agent and pipeline
    agent = Agent(
        chat_generator=AnthropicChatGenerator(model="claude-3-7-sonnet-latest", generation_kwargs={"max_tokens": 15_000}),
        system_prompt="""You are an expert AI assistant interfacing with PrimeKG, a comprehensive precision medicine knowledge graph hosted in a Neo4j database.
Your mission is to help users explore PrimeKG by translating their natural language questions about diseases, drugs, proteins, genes, biological
pathways, and phenotypes into precise Cypher queries. Use the appropriate Neo4j tools to execute these queries against the database. After
receiving the query results, synthesize the information into a clear and helpful answer for the user.

PrimeKG is a rich resource, integrating data from 20 sources, covering diseases and millions of relationships. It includes details on drug
indications, contraindications, off-label uses, protein perturbations, and more, across multiple biological scales.

# Database Structure

## Node Types
The database uses a single Neo4j label "Node" with a "node_type" property that categorizes nodes into the following types:
1. biological_process
2. gene/protein
3. disease
4. effect/phenotype
5. anatomy
6. molecular_function
7. drug
8. cellular_component
9. pathway
10. exposure

## Node Properties
Nodes have the following properties:
- node_index: INTEGER (indexed)
- node_name: STRING
- node_type: STRING 
- node_source: STRING
- node_id: STRING

## Relationship Types
All relationships in the database use the Neo4j type "RELATES_TO" with properties:
- relation: STRING (specifies the actual relationship type)
- display_relation: STRING

The top relationship types (in the "relation" property) include:
1. anatomy_protein_present: Indicates a protein is present in an anatomical structure
2. drug_drug: Relationships between drugs
3. protein_protein: Interactions between proteins
4. disease_phenotype_positive: Associations between diseases and phenotypes
5. bioprocess_protein: Proteins involved in biological processes
6. cellcomp_protein: Proteins located in specific cellular components
7. disease_protein: Associations between diseases and proteins
8. molfunc_protein: Proteins with specific molecular functions
9. drug_effect: Effects of drugs
10. bioprocess_bioprocess: Relationships between biological processes
11. pathway_protein: Proteins involved in pathways
12. disease_disease: Relationships between diseases
13. contraindication: Drug contraindications
14. drug_protein: Drug-protein interactions
15. anatomy_protein_absent: Proteins absent in specific anatomical structures
16. indication: Drug indications

# Query Guidelines
When creating Cypher queries:
- Use the single Neo4j label "Node" for all node types
- Filter nodes by their "node_type" property (e.g., `WHERE n.node_type = "disease"`)
- Use the "RELATES_TO" relationship type for all relationships
- Filter relationships using the "relation" property (e.g., `WHERE r.relation = "disease_protein"`)
- Use the "display_relation" property for human-readable relationship descriptions in visualizations
- When using LIMIT clauses, use a value of up to 100 to ensure comprehensive results (not small values like 5 or 20)
- Consider using ORDER BY clauses to prioritize the most relevant results when using LIMIT

Always strive to provide answers based *only* on the information retrieved from the PrimeKG database via your Cypher queries. If a query returns no
results or insufficient information, clearly state that.

When your Cypher queries return interconnected data (e.g., multiple nodes and their relationships) that would benefit from a visual representation,
you should include a graph visualization section in your response. This section must be clearly demarcated as follows:
<graph_visualization>
{
  "nodes": [
    {"id": "unique_node_id_1", "label": "Node Label 1", "title": "Tooltip for Node 1", "group": "node_type value", "color": "#RRGGBB"},
    {"id": "unique_node_id_2", "label": "Node Label 2", "title": "Tooltip for Node 2", "group": "node_type value", "color": "#RRGGBB"}
  ],
  "edges": [
    {"source": "unique_node_id_1", "target": "unique_node_id_2", "label": "Value from r.display_relation"}
  ]
}
</graph_visualization>

When your Cypher queries return interconnected data suitable for visualization (e.g., specific proteins and their pathways as requested), you should prepare a `<graph_visualization>` block.
1.  Your Cypher query for this data should aim to be comprehensive (e.g., using `LIMIT` up to 100, as per general query guidelines) to gather a good dataset.
2.  **Then, when constructing the JSON for the `<graph_visualization>` block, you should include a substantial number of the nodes and relationships *retrieved by that query*. For example, if your query for proteins and pathways returns 50 distinct entities and their connections, aim to include a significant portion of these (e.g., 20-30 entities or more if manageable) in the graph visualization, rather than just a few highlights. The goal is to provide a visually richer representation of the data directly related to the user\'s question.**
3.  Ensure all node `id`s are unique, and `group`, `color`, and edge `label`s are correctly assigned as per the schema.

The `id` for nodes must be unique. **Crucially, for every node, you MUST specify a `group` (e.g., 'disease', 'gene/protein', 'drug', 'pathway', etc.) based on the node's "node_type" property and a distinct `color` (in hex format, e.g., "#FF5733") for that group.** The `group` name MUST match the exact value of the "node_type" property as retrieved from the database. **For edges, the `label` MUST be derived from the query results: prioritize using the value of the relationship property named `display_relation` if it exists; otherwise, use the value of the `relation` property.** This ensures all node types are represented in the legend and that the visualization accurately reflects the graph data.
""",
        tools=active_agent_tools # Use the filtered list of tools
    )
    
    pipeline = Pipeline()
    pipeline.add_component("agent", agent)
    return pipeline, mcp_toolset # Return the original provider, agent has filtered tools

# Initialize pipeline once
pipeline, mcp_toolset = initialize_pipeline()

# Main chat interface
st.title("MedOryx Precision Medicine AI Agent")
st.info("""A conversational interface to the PrimeKG Knowledge Graph. Built with [Haystack Agent](https://github.com/deepset-ai/haystack), [Neo4j database](https://neo4j.com/), and [Neo4j MCP](https://github.com/neo4j-contrib/mcp-neo4j/blob/main/servers/mcp-neo4j-cypher/README.md).

Understanding complex biomedical data such as connections between diseases, genes, and drugs is challenging. Information is often spread out and querying databases requires knowledge of specific query languages like Cypher. This chat app allows users to ask questions in plain English and see results including interactive graphs and the actual database queries generated by the Agent.

### About the Data Source (PrimeKG)

The PrimeKG knowledge graph is ingested into a Neo4j database and serves as the core data source. It integrates data from 20 distinct biomedical resources to catalog 17,080 diseases and encompass over 4 million relationships. These relationships cover diverse areas such as protein-protein interactions, biological processes, and drug effects: indications, contraindications, and off-label uses.. For more details, see the original publication: [Chandak et al., Sci Data (2023)](https://doi.org/10.1038/s41597-023-01960-3).

### How it works

The Haystack Agent accesses and processes information from the Neo4j database through the following steps:

1.  **Question Interpretation:** When a user asks a question in natural language, the Haystack AI agent (utilizing an Anthropic Claude model) interprets the query.
2.  **Cypher Query Generation:** The Haystack Agent is provided with a prompt defining the high-level schema of the graph (e.g., node types, relationship types). Using this schema and the interpreted question, it generates Cypher queries. The agent may engage in multiple interactions with the database, refining queries based on intermediate results.
3.  **Database Interaction:** The generated Cypher queries are executed against the Neo4j database using the Neo4j MCP Server.
4.  **Result Presentation:** The Agent synthesizes the data retrieved from Neo4j into a plain text answer. In addition, the generated answer includes a graph describing nodes and relationships in JSON format that is then used by the `streamlit-agraph` library to render an interactive visual graph.


### Example Questions

- Are there shared molecular pathways between Type 2 diabetes and Alzheimer's disease?
- Which proteins are targeted by drugs for autoimmune thyroid conditions?
- What environmental factors are linked to lung cancer risk?
- Which diseases are commonly linked to multiple genes in this set: BRCA1, TP53, APOE, CFTR, and MECP2?
- Find diseases related to the gene APP.
- Show me proteins associated with Parkinson's disease and their biological pathways.
""")

# Display chat history with tool invocations
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # For assistant messages, parse for graph and display text part separately
            # Pass expanded=False for historical graphs to not auto-open all of them
            text_to_display, _ = render_graph_from_message_content(message["content"], is_expanded=False)
            if text_to_display: # Only display markdown if there's text remaining after graph extraction
                st.markdown(text_to_display)
            elif not _: # If render_graph_from_message_content returned no text and didn't render a graph
                st.markdown(message["content"]) # Fallback to raw content if absolutely nothing was processed
        else:
            st.markdown(message["content"]) # User messages are displayed as is
        
        invocations = st.session_state.tool_invocations[idx] if idx < len(st.session_state.tool_invocations) else None
        if message["role"] == "assistant" and invocations:
            with st.expander("🔧 View Cypher Queries", expanded=False):
                for tool_invocation in invocations:
                    tool_name = tool_invocation.get("name", "Unknown Tool")
                    tool_args = tool_invocation.get("args", {})
                    tool_result = tool_invocation.get("result", "No result")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"🔧 **{tool_name}**")
                    with col2:
                        st.markdown("**Input:**")
                        st.json(tool_args) # Input args (e.g., Cypher query) are usually JSON-friendly
                        st.markdown("**Output:**")
                        try:
                            # Handle MCP's TextContent wrapper if present
                            if isinstance(tool_result, str) and "meta=" in tool_result and "content=[TextContent" in tool_result:
                                text_match = re.search(r"text='(.*?)'", tool_result, re.DOTALL) # DOTALL for multiline, non-greedy
                                if text_match:
                                    extracted_text = text_match.group(1).replace('\\\\n', '\\n').replace('\\n', '\\n') # Handle escaped newlines
                                    try:
                                        # Attempt to parse the extracted text as JSON
                                        parsed_json_from_extracted = json.loads(extracted_text)
                                        st.json(parsed_json_from_extracted) # Pretty-prints and wraps
                                    except json.JSONDecodeError:
                                        # If not valid JSON, display as is (could be multi-line text)
                                        st.code(extracted_text, language=None)
                                else:
                                    st.code(tool_result, language=None) # Fallback if regex fails, show raw result
                            # Handle strings that might just have escaped newlines (common direct output)
                            elif isinstance(tool_result, str) and ("\\n" in tool_result or "\\n" in tool_result):
                                clean_text = tool_result.replace('\\n', '\n').replace('\n', '\n')
                                st.code(clean_text, language=None)
                            # Handle other string results (could be JSON or plain text)
                            elif isinstance(tool_result, str):
                                try:
                                    parsed_json = json.loads(tool_result)
                                    st.json(parsed_json) # If it's a valid JSON string
                                except json.JSONDecodeError:
                                    st.code(tool_result, language=None) # If not JSON, display as preformatted text
                                except Exception as json_ex: # Catch any other JSON processing error
                                    logger.error(f"Error displaying tool result as JSON: {json_ex}")
                                    st.code(str(tool_result), language=None) # Fallback to code display
                            # Handle non-string results (e.g., already a dict/list)
                            else:
                                st.json(tool_result)
                        except Exception as display_ex:
                            logger.error(f"Error displaying tool result: {display_ex}\n{traceback.format_exc()}")
                            st.text(str(tool_result)) # Fallback to plain text for any error
                    st.markdown("---")

# Process new messages
if prompt := st.chat_input("Ask PrimeKG about diseases, drugs, genes..."):
    # Display and store user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.tool_invocations.append(None)  # Add placeholder for user message tools (no invocations)
    
    # Process with agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Querying PrimeKG..."): # Updated spinner text
            try:
                # Convert chat history to ChatMessage objects
                # This list will be passed to the agent
                chat_messages_for_agent = []
                for msg in st.session_state.messages: # Use all messages up to and including the current user prompt
                    if msg["role"] == "user":
                        chat_messages_for_agent.append(ChatMessage.from_user(msg["content"]))
                    elif msg["role"] == "assistant":
                        # For assistant messages in history, include content. Tool calls are handled by Haystack internally.
                        chat_messages_for_agent.append(ChatMessage.from_assistant(msg["content"]))
                
                # Run agent pipeline with the constructed chat history
                result = pipeline.run({"agent": {"messages": chat_messages_for_agent}})
                logger.info(f"Agent run result: {result}")
                
                # Extract all messages from the agent's response (can include multiple tool calls and assistant replies)
                messages_from_run = result.get("agent", {}).get("messages", [])
                
                # Extract tool invocations from "tool" messages in the current run's output
                current_tool_invocations = []
                for msg_from_run in messages_from_run:
                    if msg_from_run.role == "tool" and hasattr(msg_from_run, 'tool_call_results'):
                        for tool_call_result in msg_from_run.tool_call_results:
                            tool_origin = tool_call_result.origin  # This is the original ToolCall (request)
                            if tool_origin: # Ensure origin exists
                                current_tool_invocations.append({
                                    "name": tool_origin.tool_name,
                                    "args": tool_origin.arguments, # This is usually a dict like {"query": "MATCH (n) RETURN n LIMIT 1"}
                                    "result": tool_call_result.result # The actual string/object result from the tool
                                })
                
                # Find the final assistant text response
                response_text = ""
                for msg_from_run in reversed(messages_from_run): # Look from the end
                    if msg_from_run.role == "assistant" and msg_from_run.text:
                        response_text = msg_from_run.text
                        break
                
                # Handle cases where there might be no direct text response but tool calls occurred
                if not response_text and current_tool_invocations:
                    response_text = "I've used some tools to process your request. See the invocations for details."
                elif not response_text and not current_tool_invocations:
                    response_text = "I could not generate a response. Please try rephrasing your query."

                # Store the full assistant response text (including graph block if present)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.tool_invocations.append(current_tool_invocations) # Store for history display

                # --- START GRAPH VISUALIZATION LOGIC ---
                # Use the refactored function for new messages as well
                # For new graphs, we want them expanded by default, so is_expanded=True
                response_text_for_markdown, graph_rendered_in_expander = render_graph_from_message_content(response_text, is_expanded=True)
                
                # Display the (potentially modified) text response
                if response_text_for_markdown:
                    st.markdown(response_text_for_markdown)
                # If the original response_text was empty, and there was no graph block even started, 
                # or if graph processing left response_text_for_markdown empty and no graph rendered.
                elif not response_text and not graph_rendered_in_expander and not re.search(r"<graph_visualization>", response_text): 
                     st.markdown("I could not generate a textual response and no graph was provided.")
                elif not response_text_for_markdown and not graph_rendered_in_expander and re.search(r"<graph_visualization>", response_text):
                    # This case handles when a graph was expected (opening tag found) and failed to render, 
                    # and there was no other text preceding the graph block.
                    # The st.error in render_graph_from_message_content would have already shown a message.
                    pass


                # Display current tool invocations immediately after the response
                if current_tool_invocations:
                    with st.expander("🔧 View Cypher Queries For This Response", expanded=False): # Expand by default for new results
                        for tool_invocation in current_tool_invocations:
                            tool_name = tool_invocation.get("name", "Unknown Tool")
                            tool_args = tool_invocation.get("args", {})
                            tool_result = tool_invocation.get("result", "No result")
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown(f"🔧 **{tool_name}**")
                            with col2:
                                st.markdown("**Input:**")
                                st.json(tool_args)
                                st.markdown("**Output:**")
                                # Replicated logic from historical display for consistency
                                try:
                                    if isinstance(tool_result, str) and "meta=" in tool_result and "content=[TextContent" in tool_result:
                                        text_match = re.search(r"text='(.*?)'", tool_result, re.DOTALL)
                                        if text_match:
                                            extracted_text = text_match.group(1).replace('\\\\n', '\\n').replace('\\n', '\\n')
                                            try:
                                                # Attempt to parse the extracted text as JSON
                                                parsed_json_from_extracted = json.loads(extracted_text)
                                                st.json(parsed_json_from_extracted) # Pretty-prints and wraps
                                            except json.JSONDecodeError:
                                                # If not valid JSON, display as is (could be multi-line text)
                                                st.code(extracted_text, language=None)
                                        else:
                                            st.code(tool_result, language=None)
                                    elif isinstance(tool_result, str) and ("\\n" in tool_result or "\\n" in tool_result):
                                        clean_text = tool_result.replace('\\n', '\n').replace('\n', '\n')
                                        st.code(clean_text, language=None)
                                    elif isinstance(tool_result, str):
                                        try:
                                            parsed_json = json.loads(tool_result)
                                            st.json(parsed_json)
                                        except json.JSONDecodeError:
                                            st.code(tool_result, language=None)
                                        except Exception as json_ex:
                                            logger.error(f"Error displaying tool result as JSON: {json_ex}")
                                            st.code(str(tool_result), language=None)
                                    else:
                                        st.json(tool_result)
                                except Exception as display_ex:
                                    logger.error(f"Error displaying tool result: {display_ex}\n{traceback.format_exc()}")
                                    st.text(str(tool_result))
                            st.markdown("---")
                            
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc()) # Log full traceback for debugging
                st.error(error_msg)
                # Append error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.tool_invocations.append([])  # Add empty tools for error message 