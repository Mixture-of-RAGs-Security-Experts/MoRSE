import os
import pandas as pd
import time
import statistics
from py2neo import Graph
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class MalwareAnalyzer:
    def __init__(self):
        self.model_name = "BAAI/bge-large-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs, encode_kwargs=self.encode_kwargs
        )
        self.retriever = None
        self.df = self.get_malware_df()
        self.graph = self.get_graph()

    def get_graph(self):
        """Connect to the Neo4j database."""
        username = ""
        password = ""
        database_uri = "bolt://localhost:7687"  # Ensure to use the correct URI

        # Connect to the Neo4j database
        graph = Graph(database_uri, auth=(username, password))
        return graph

    def execute_cypher_query(self, query, parameters=None):
        """Execute a Cypher query and return the results."""
        try:
            with self.graph.session() as session:
                result = session.run(query, parameters)
                return result.data()
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None

    def get_malware_data(self):
        """Retrieve malware data from the database."""
        query = """
        MATCH (m:Malware_name)
        RETURN m.Malware_name, m.description_malware
        """
        return self.execute_cypher_query(query)

    def get_techniques_for_malware(self, malware_name):
        """Retrieve techniques used by a specific malware."""
        query = """
        MATCH (m1:Malware_name {Malware_name: $malware_name})-[:USES]->(t:Technique_name)
        RETURN t.Technique_name
        """
        parameters = {"malware_name": malware_name}
        return self.execute_cypher_query(query, parameters)

    def mitigations_for_malware_techniques(self, malware_name):
        """Retrieve mitigations for techniques used by a specific malware."""
        query = """
        MATCH (m:Malware_name {Malware_name: $malware_name})-[:USES]->(t:Technique_name)
        MATCH (t)-[:MITIGATED]->(mit:Mitigation_name)
        RETURN DISTINCT mit.Mitigation_name, mit.description_mitigation, t.Technique_name
        """
        parameters = {"malware_name": malware_name}
        return self.execute_cypher_query(query, parameters)

    def create_malware_dataframe(self, data):
        """Create a DataFrame from malware data."""
        descriptions = []
        df_data = []

        for malware in data:
            malware_name = malware['m.Malware_name']
            malware_description = malware['m.description_malware']
            descriptions.append(malware_description)
            technique_result = self.get_techniques_for_malware(malware_name)
            techniques = ', '.join(technique['t.Technique_name'] for technique in technique_result)
            df_data.append({
                'Malware Name': malware_name,
                'Malware Description': malware_description,
                'Techniques Used': techniques
            })

        df = pd.DataFrame(df_data)
        return df

    def save_dataframe_to_csv(self, df, filename):
        """Save the DataFrame to a CSV file."""
        df.to_csv(filename, index=False)

    def get_malware_df(self):
        """Load the malware DataFrame from a CSV file."""
        df = pd.read_csv('malware_data.csv')
        return df

    def build_faiss_index(self, descriptions):
        """Build and save a FAISS index from malware descriptions."""
        db = FAISS.from_texts(descriptions, self.hf)
        db.save_local("Malware_Descriptions")

    def load_faiss_index(self):
        """Load the FAISS index from the local storage."""
        return FAISS.load_local("Malware_Descriptions", self.hf)

    def create_retriever(self, db):
        """Create a retriever from the FAISS index."""
        return db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 10})

    def run(self, query):
        """Run the malware analysis for a given query."""
        faiss_db = self.load_faiss_index()
        self.retriever = self.create_retriever(faiss_db)
        context = ''
        for doc in self.retriever.get_relevant_documents(query):
            malware_description = doc.page_content
            row = self.df.loc[self.df['Malware Description'] == malware_description]
            
            if not row.empty:
                row_dict = row.to_dict(orient='records')[0]
                name = row_dict['Malware Name']
                desc = row_dict['Malware Description']
                techniques = row_dict['Techniques Used']

                # Handle the case where techniques is NaN
                if pd.notna(techniques):
                    context += '\n' + desc + f' Techniques used by {name}: ' + techniques
                else:
                    context += '\n' + desc
            else:
                print("[-] No Malware found ...")
        return context

# if __name__ == "__main__":
#     malware_analyzer = MalwareAnalyzer()
#     with open("mitre_questions.txt", "r") as f:
#         lines = f.readlines()
#     times = []
#     for query in lines:
#         start_time = time.time()
#         context = malware_analyzer.run(query)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         times.append(elapsed_time)

#     media = statistics.mean(times)
#     std = statistics.stdev(times)
#     print(f"Average time elapsed: {media}")
#     print(f"Standard Deviation: {std}")
