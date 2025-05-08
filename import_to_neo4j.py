from py2neo import Graph, Node, Relationship
import csv
from tqdm import tqdm

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# CSV file paths
NODES_CSV = "dataverse_files/nodes.csv"
EDGES_CSV = "dataverse_files/edges.csv"

# Constants
BATCH_SIZE = 10000  # Adjust based on your system's memory

# Connect to Neo4j
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def import_nodes():
    print("Importing nodes...")
    nodes = []
    
    with open(NODES_CSV, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in tqdm(reader):
            nodes.append({
                "node_index": int(row['node_index']),
                "node_id": row['node_id'],
                "node_type": row['node_type'],
                "node_name": row['node_name'],
                "node_source": row['node_source']
            })
            
            if len(nodes) >= BATCH_SIZE:
                create_nodes_batch(nodes)
                nodes = []
    
    if nodes:  # Create any remaining nodes
        create_nodes_batch(nodes)
    
    print("Nodes import completed.")

def create_nodes_batch(nodes):
    query = """
    UNWIND $nodes AS node
    CREATE (n:Node)
    SET n = node
    """
    graph.run(query, nodes=nodes)

def import_edges():
    print("Importing edges...")
    edges = []
    
    with open(EDGES_CSV, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in tqdm(reader):
            edges.append({
                "start_index": int(row['x_index']),
                "end_index": int(row['y_index']),
                "relation": row['relation'],
                "display_relation": row['display_relation']
            })
            
            if len(edges) >= BATCH_SIZE:
                create_edges_batch(edges)
                edges = []
    
    if edges:  # Create any remaining edges
        create_edges_batch(edges)
    
    print("Edges import completed.")

def create_edges_batch(edges):
    query = """
    UNWIND $edges AS edge
    MATCH (start:Node {node_index: edge.start_index})
    MATCH (end:Node {node_index: edge.end_index})
    CREATE (start)-[r:RELATES_TO {relation: edge.relation, display_relation: edge.display_relation}]->(end)
    """
    graph.run(query, edges=edges)

def create_indexes():
    print("Creating indexes...")
    # Create index on node_index for all node types
    graph.run("CREATE INDEX node_index_idx IF NOT EXISTS FOR (n:Node) ON (n.node_index)")
    print("Indexes created.")

if __name__ == "__main__":
    create_indexes()
    import_nodes()
    import_edges()
    print("Import process completed.")
