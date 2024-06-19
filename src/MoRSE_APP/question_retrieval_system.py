import os
import re
import warnings
import logging
import time
from tqdm import tqdm
import pandas as pd
from langchain.document_transformers import LongContextReorder
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import FAISS

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging configuration
log_file = "Question_Retrieval_Log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class QuestionRetrievalSystem:
    def __init__(self):
        self.bm25_retrievers = []
        self.vector_stores = []
        self.df = pd.DataFrame()
        self.compression_retriever_reordered = None

        # Initialize HuggingFaceBgeEmbeddings
        self.hf_bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en"
        )

        # Load and set up retrievers
        self.setting_filters()
        self.initialize_dataframe()

    def find_questions_file(self, directory):
        """Search for 'questions.txt' file in the given directory."""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "questions.txt":
                    return os.path.join(root, file)
        return None

    def initialize_dataframe(self):
        """Initialize the dataframe with questions and contexts."""
        try:
            data_original, inquiries_original = self.build_original_df()
            data_metasploit, inquiries_metasploit = self.build_metasploit_df()
            data_references, inquiries_references = self.build_ref_question_df()
            data_books, inquiries_books = self.build_book_question_df()

            # Concatenate the data and inquiries
            self.df = pd.DataFrame(
                data_original + data_metasploit + data_references + data_books
            )
            inquiries = (
                inquiries_original
                + inquiries_metasploit
                + inquiries_references
                + inquiries_books
            )
            self.df["Question"] = inquiries

            self.compression_retriever_reordered = (
                self.initialize_compression_retriever()
            )
        except Exception as e:
            logging.error(f"Error initializing dataframe: {str(e)}")

    def initialize_compression_retriever(self):
        """Initialize the compression retriever with document compressors."""
        try:
            lotr = MergerRetriever(retrievers=self.bm25_retrievers)
            reordering = LongContextReorder()
            relevant_filter = EmbeddingsFilter(
                embeddings=self.get_paper_embeddings(), similarity_threshold=0.7
            )
            pipeline = DocumentCompressorPipeline(
                transformers=[relevant_filter, reordering]
            )
            compression_retriever_reordered = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=lotr,
                search_kwargs={"top_k": 5},
            )
            return compression_retriever_reordered
        except Exception as e:
            logging.error(f"Error initializing compression retriever: {str(e)}")
            return None

    def build_df_from_path(self, base_path, source_prefix=""):
        """Build a dataframe from the given path by processing the 'questions.txt' files."""
        books = os.listdir(base_path)
        data = []
        inquiries = []

        for book in tqdm(books):
            try:
                book_path = os.path.join(base_path, book)
                questions_file = self.find_questions_file(book_path)

                if questions_file:
                    with open(questions_file, "r") as f:
                        text = f.read()
                        rows = text.split("\n\n")

                        for row in rows:
                            raw_questions = row.split("'questions': [", 1)[1].split(
                                "]}], 'documents': [<Document: {'content': "
                            )[0]
                            context = row.split(
                                "]}], 'documents': [<Document: {'content': ", 1
                            )[1].split(", 'content_type':")[0]
                            context = re.sub(r"\\n", " ", context)
                            context = context.replace("\\'", "'")

                            questions = raw_questions.split("?")
                            for question in questions:
                                question = question.replace("'", "")
                                if ", " in question:
                                    question = question.replace(", ", "")
                                if '"' in question:
                                    question = question.replace('"', "")
                                if question.strip():
                                    single_question = question + "?"
                                    inquiries.append(single_question)
                                    data.append(
                                        {
                                            "Source": source_prefix + book,
                                            "Question": single_question,
                                            "Context": context,
                                        }
                                    )
                else:
                    logging.warning(f"No 'questions.txt' file found in {book_path}")
            except Exception as e:
                logging.error(f"Error processing book {book}: {str(e)}")

        return data, inquiries

    def build_original_df(self):
        """Build dataframe for the original dataset."""
        try:
            http_path = "/home/marcos/GPU_EXPERT/Generated_Questions_Original/http:"
            https_path = "/home/marcos/GPU_EXPERT/Generated_Questions_Original/https:"

            data_http, inquiries_http = self.build_df_from_path(
                http_path, source_prefix="http://"
            )
            data_https, inquiries_https = self.build_df_from_path(
                https_path, source_prefix="https:"
            )

            return data_http + data_https, inquiries_http + inquiries_https
        except Exception as e:
            logging.error(f"Error building original dataframe: {str(e)}")
            return [], []

    def build_metasploit_df(self):
        """Build dataframe for the Metasploit dataset."""
        try:
            base_path = "/home/marcos/GPU_EXPERT/Generated_Questions_Metasploit/https:/"
            return self.build_df_from_path(base_path)
        except Exception as e:
            logging.error(f"Error building Metasploit dataframe: {str(e)}")
            return [], []

    def build_ref_question_df(self):
        """Build dataframe for the reference questions dataset."""
        try:
            path = "/home/marcos/GPU_EXPERT/Generated_Questions_References/"
            return self.build_df_from_path(path)
        except Exception as e:
            logging.error(f"Error building reference question dataframe: {str(e)}")
            return [], []

    def build_book_question_df(self):
        """Build dataframe for the book questions dataset."""
        try:
            path = "/home/marcos/GPU_EXPERT/Generated_Questions_Books/"
            return self.build_df_from_path(path)
        except Exception as e:
            logging.error(f"Error building book question dataframe: {str(e)}")
            return [], []

    def get_paper_embeddings(self):
        """Retrieve paper embeddings using HuggingFaceBgeEmbeddings."""
        model_name = "BAAI/bge-large-en-v1.5"
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
        )
        return embeddings

    def set_bm25_book(self):
        """Set BM25 retriever for the book dataset."""
        db_book = FAISS.load_local(
            "db_book_prova_single_question", self.get_paper_embeddings()
        )
        retriever_book = db_book.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        self.bm25_retrievers.append(retriever_book)
        self.vector_stores.append(db_book)

    def set_bm25_original(self):
        """Set BM25 retriever for the original dataset."""
        db_original = FAISS.load_local(
            "db_original_prova_single_question", self.get_paper_embeddings()
        )
        retriever_original = db_original.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7},
        )
        self.bm25_retrievers.append(retriever_original)
        self.vector_stores.append(db_original)

    def set_bm25_metasploit(self):
        """Set BM25 retriever for the Metasploit dataset."""
        db_metasploit = FAISS.load_local(
            "db_metasploit_prova_single_question", self.get_paper_embeddings()
        )
        retriever_metasploit = db_metasploit.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7},
        )
        self.bm25_retrievers.append(retriever_metasploit)
        self.vector_stores.append(db_metasploit)

    def set_bm25_refs(self):
        """Set BM25 retriever for the references dataset."""
        db = FAISS.load_local(
            "db_ref_prova_single_question", self.get_paper_embeddings()
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        self.bm25_retrievers.append(retriever)
        self.vector_stores.append(db)

    def setting_filters(self):
        """Set up all the BM25 retrievers."""
        self.set_bm25_book()
        self.set_bm25_metasploit()
        self.set_bm25_original()
        self.set_bm25_refs()

    def first_filter_compression(self, query, entity):
        """Perform the first filter compression to retrieve relevant documents."""
        start_time = time.time()
        doc_found = []
        contents = []

        original_question = isinstance(entity, list)

        try:
            for doc in self.compression_retriever_reordered.get_relevant_documents(
                query
            ):
                logging.info(doc.page_content)
                if self.should_include_document(doc, entity, original_question):
                    logging.info("Question: " + doc.page_content)
                    doc_found.append(doc)
                    print("Question: " + doc.page_content)
                    contents.append(self.get_content_result(doc))

        except Exception as e:
            logging.error(f"Error in first_filter_compression: {str(e)}")

        elapsed_time = time.time() - start_time
        logging.info(f"Execution Time: {elapsed_time} seconds")
        logging.info(f"Total DOCS: {len(doc_found)}")

        return contents

    def should_include_document(self, doc, entity, original_question):
        """Determine if a document should be included based on specific conditions."""
        return (
            (original_question and any(name in doc.page_content for name in entity))
            or (
                not original_question
                and any(
                    any(name in doc.page_content for name in sublist)
                    for sublist in entity
                )
            )
            or (
                not original_question
                and entity
                and any(name in doc.page_content for name in entity)
            )
            or not entity
        )

    def get_content_result(self, doc):
        """Retrieve the context result based on the document content."""
        content_result = self.df.loc[
            self.df["Question"] == doc.page_content, "Context"
        ].values
        return content_result[0] if content_result[0] else doc.page_content

    def run(self, query, entities_extracted):
        """Run the question retrieval process and return the context and elapsed time."""
        try:
            entities = []
            entity_names = []
            for extracted in entities_extracted:
                entity_name = extracted["word"]
                entity_names.append(entity_name)
                logging.info("Entity Extracted: %s", entity_name)
                if entity_name != "exploit" and entity_name != "Exploit":
                    new_question = f"What is {entity_name} ?"
                    if (
                        new_question not in entities
                        and query.replace("?", "") not in new_question
                    ):
                        entities.append((new_question, entity_name))

            entities.append((query, entity_names))
            logging.info("Total Questions: %s", entities)
            context = ""
            retrieved_contents_base = []
            for question, entity in entities:
                start_time = time.time()
                retrieved_contents = self.first_filter_compression(question, entity)
                end_time = time.time()
                elapsed_time = end_time - start_time
                for content in retrieved_contents:
                    if content not in retrieved_contents_base:
                        retrieved_contents_base.append(content)
                        context += content + "\n"

                logging.info(context)

        except Exception as e:
            logging.error(f"Error in run method: {str(e)}")

        print(context)
        return context, elapsed_time


# # Example Usage:

# if __name__ == "__main__":
#     question_processor = QuestionRetrievalSystem()
#     entity_extractor = EntityExtractor(model_name_or_path="dslim/bert-large-NER")
#     while True:
#         query = input("\nInsert a query: ")
#         entities_extracted = entity_extractor.extract(text=query)
#         question_processor.run(query, entities_extracted)


# # Example Usage for processing multiple queries:

# with open("MULTIHOP_QUESTIONS_4.txt", "r") as o:
#     lines = o.readlines()

# times = []
# total_contexts = []
# if __name__ == "__main__":
#     question_processor = QuestionRetrievalSystem()
#     entity_extractor = EntityExtractor(model_name_or_path="dslim/bert-large-NER", use_gpu=True)
#     total_scores = []
#     for query in lines:
#         try:
#             entities_extracted = entity_extractor.extract(text=query)
#             context, elapsed_time = question_processor.run(query, entities_extracted)
#             data = {"query": query, "context": context}
#             with open("MoRSE_Contents_MultiHop4.txt", "a") as f:
#                 f.write(str(data))
#                 f.write("\n")
#         except Exception as e:
#             pass
