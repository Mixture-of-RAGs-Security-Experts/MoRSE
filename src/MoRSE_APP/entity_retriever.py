import logging
import time
import statistics
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

class EntityRetrievalSystem:
    def __init__(self):
        self.setup_logger()
        self.retriever = self.set_up()

    def setup_logger(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename="Entity_Retrieval_Log.txt",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logging_logger = logging.getLogger("EntityRetrievalLogger")
        self.entity_logger = logging.getLogger("EntityRetrievalLogger")

    def set_up(self):
        """Initialize the retriever with embeddings and set up the retriever."""
        try:
            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            self.entity_logger.info("Entities embeddings loaded successfully.")

            db = FAISS.load_local(
                "/home/marcos/GPU_EXPERT/PARALLEL_EXPERIMENT/Entities_Database_Complete",
                embeddings,
            )
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 5},
            )
            self.entity_logger.info("Retriever set up successfully.")
            return retriever
        except Exception as e:
            self.entity_logger.error(f"An error occurred during setup: {e}")
            raise  # Rethrow the exception after logging

    def count_words(self, text):
        """Count the number of words in the given text."""
        words = text.split()
        return len(words)

    def retrieve_entities(self, query):
        """Retrieve relevant documents based on the query."""
        try:
            context = ""
            for doc in self.retriever.get_relevant_documents(query):
                total_word = self.count_words(doc.page_content)
                if total_word >= 10:
                    context += doc.page_content + "\n\n\n"
            self.entity_logger.info(f"Documents retrieved successfully for query: {query}")
            return context
        except Exception as e:
            print(e)
            self.entity_logger.error(f"An error occurred during retrieval for query {query}: {e}")
            self.logging_logger.error(f"An error occurred during retrieval for query {query}: {e}")
            raise  # Rethrow the exception after logging

def get_questions():
    """Read questions from a file and return them as a list."""
    file_name = 'entity_questions.txt'
    with open(file_name, 'r') as f:
        questions = f.readlines()
    return questions

# # Main execution block
# if __name__ == "__main__":
#     total_scores = []
#     total_contexts = []
#     times = []
#     try:
#         entity_retrieval_system = EntityRetrievalSystem()
#         questions = get_questions()
#         for query in questions:
#             print(f"Processing Question: {query}")
#             start_time = time.time()
#             context = entity_retrieval_system.retrieve_entities(query)
#             elapsed_time = time.time() - start_time
#             times.append(elapsed_time)
#             total_contexts.append(context)
#             print(f"Time ENTITIES Database: {elapsed_time} seconds")
#     except Exception as e:
#         print(e)

#     FAILS = sum(1 for con in total_contexts if con == '')
#     media = statistics.mean(times)
#     std = statistics.stdev(times)

#     print(f"TOTAL FAILS: {FAILS}")
#     print(f"Average time elapsed: {media}")
#     print(f"Standard Deviation: {std}")
