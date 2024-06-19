import torch
import os
import pandas as pd
import json
import re
import time
import logging
from tqdm import tqdm
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

class CybersecurityAssistant:
    def __init__(self):
        self.cve_path = "cves"
        self.result_df = self.load_combined_dataframe()
        self.hf_bge_embeddings, self.hf_embeddings, self.multi_qa_mini = self.set_embeddings()
        self.text_documents = self.get_text_documents()
        self.metasploit_documents = self.get_metasploit_documents()
        self.code_documents = self.get_code_documents()
        self.paper_documents = self.get_paper_documents()
        self.total_documents = self.text_documents + self.metasploit_documents + self.paper_documents
        self.retrievers = self.build_retrievers()
        self.cve_retriever = self.build_cve_retriever()
        self.compression_retriever_reordered = self.compression_retriever()

        # Configure logging
        try:
            logging.basicConfig(
                filename="Assistant_Log.txt",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self.logger = logging.getLogger("logger")
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}", exc_info=True)

    def load_combined_dataframe(self):
        """Load and combine dataframes from multiple sources."""
        try:
            general_df = self.load_dataframe("cybersecurity_dataframe.csv")
            metasploit_df = self.load_dataframe("METASPLOIT_DATAFRAME.csv")
            paper_df = self.load_dataframe("PAPERS_DATAFRAME.csv")
            result_df = pd.concat([general_df, metasploit_df, paper_df], ignore_index=True)
            return result_df
        except Exception as e:
            self.logger.error(f"Error loading combined dataframe: {e}", exc_info=True)
            return None

    def load_dataframe(self, file_path):
        """Load a single dataframe from a CSV file."""
        df_loaded = pd.read_csv(file_path)
        df_loaded.rename(columns={'page_content': 'content'}, inplace=True)
        df_inverted = df_loaded.iloc[:, ::-1]
        return df_inverted

    def set_embeddings(self):
        """Set embeddings for various models."""
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        hf_bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en",
            encode_kwargs={'normalize_embeddings': True}
        )
        multi_qa_mini = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")
        return hf_bge_embeddings, hf_embeddings, multi_qa_mini

    def get_text_documents(self):
        """Retrieve text documents from the FAISS database."""
        bge_db = FAISS.load_local("/complete_links_database", self.hf_bge_embeddings)
        store = bge_db.docstore._dict
        text_documents = [value for key, value in store.items()]
        return text_documents

    def set_metasploit_embeddings(self):
        """Set embeddings for Metasploit documents."""
        model_name = "BAAI/bge-small-en"
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
        )
        return embeddings

    def get_metasploit_documents(self):
        """Retrieve Metasploit documents from the FAISS database."""
        embeddings = self.set_metasploit_embeddings()
        meta_bge = FAISS.load_local("METASPLOIT_LINKS/links_db", embeddings)
        meta_store = meta_bge.docstore._dict
        meta_docs = [meta for key, meta in meta_store.items()]
        return meta_docs

    def get_code_documents(self):
        """Retrieve code documents from the local filesystem."""
        source_path = "FINAL_COLLECTIONS/"
        paths = [
            "exploit_collection_final", 
            "fuzzing_collection_final", 
            "other_collection_final",
            "windows_collection_final"
        ]
        real_documents = []

        for path in paths:
            full_path = os.path.join(source_path, path)
            for file in os.listdir(full_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(full_path, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()

                    documents = file_content.split('\n\n\n------------------------------------------------\n\n\n')
                    documents = [doc.strip() for doc in documents]

                    for doc in documents:
                        real_doc = Document(
                            page_content=doc,
                            metadata={"source": file_path}
                        )
                        real_documents.append(real_doc)
        return real_documents

    def get_paper_embeddings(self):
        """Set embeddings for paper documents."""
        model_name = "BAAI/bge-large-en-v1.5"
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
        )
        return embeddings

    def get_paper_documents(self):
        """Retrieve paper documents from the FAISS database."""
        embeddings = self.get_paper_embeddings()
        db_1 = FAISS.load_local("PAPERS/paper_db_1", embeddings)
        db_2 = FAISS.load_local("PAPERS/paper_db_2", embeddings)
        db_1_docs = [value for id, value in db_1.docstore._dict.items()]
        db_2_docs = [value for id, value in db_2.docstore._dict.items()]
        db_total = db_1_docs + db_2_docs
        return db_total

    def build_retrievers(self):
        """Build multiple retrievers for different document types."""
        retrievers = []

        for i in range(1, 6):
            text_db = FAISS.load_local(f"text_chunk_{i}", self.hf_bge_embeddings)
            text_retriever = text_db.as_retriever(include_metadata=True, search_kwargs={"top_k": 5})
            store = text_db.docstore._dict
            docs = [value for key, value in store.items()]
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5
            ensemble_text_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, text_retriever], weights=[0.5, 0.5]
            )
            retrievers.append(ensemble_text_retriever)

        for i in range(1, 6):
            meta_db = FAISS.load_local(f"metasploit_chunk_{i}", self.hf_bge_embeddings)
            meta_retriever = meta_db.as_retriever(include_metadata=True, search_kwargs={"top_k": 5})
            store = meta_db.docstore._dict
            docs = [value for key, value in store.items()]
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5
            ensemble_meta_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, meta_retriever], weights=[0.5, 0.5]
            )
            retrievers.append(ensemble_meta_retriever)

        code_db = FAISS.load_local(f"code_chunk_1", self.hf_bge_embeddings)
        code_retriever = code_db.as_retriever(include_metadata=True, search_kwargs={"top_k": 5})
        store = code_db.docstore._dict
        docs = [value for key, value in store.items()]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5
        ensemble_code_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, code_retriever], weights=[0.5, 0.5]
        )
        retrievers.append(ensemble_code_retriever)

        print("\nPaper Document Chunks:")
        for i in range(1, 4):
            print(f"Chunk {i}")
            paper_db = FAISS.load_local(f"paper_chunk_{i}", self.get_paper_embeddings())
            paper_retriever = paper_db.as_retriever(include_metadata=True, search_kwargs={"top_k": 5})
            store = paper_db.docstore._dict
            docs = [value for key, value in store.items()]
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5
            ensemble_paper_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, paper_retriever], weights=[0.5, 0.5]
            )
            retrievers.append(ensemble_paper_retriever)

        complete_scraped = FAISS.load_local("scraping/COMPLETE_SPLITTED_SCRAPING", self.get_paper_embeddings())
        scraper_retriever = complete_scraped.as_retriever(include_metadata=True, search_kwargs={"top_k": 5})
        scraped_store = complete_scraped.docstore._dict
        scraped_docs = [value for key, value in scraped_store.items()]
        bm25_retriever_scrape = BM25Retriever.from_documents(scraped_docs)
        bm25_retriever_scrape.k = 5
        ensemble_scraper_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever_scrape, scraper_retriever], weights=[0.5, 0.5]
        )
        retrievers.append(ensemble_scraper_retriever)
        return retrievers

    def build_cve_retriever(self):
        """Build the CVE retriever."""
        languages = [".json"]
        for language in tqdm(languages):
            cves = self.cve_loader(language)

        document_list = []

        for cve in tqdm(cves):
            try:
                content = json.loads(cve.page_content)
                cve_id = cve.metadata["source"].split("x/", 1)[1].split(".json")[0]

                affected = content["containers"]["cna"]["affected"][0]["product"] if "affected" in content["containers"]["cna"] else "n/a"
                descriptions = content["containers"]["cna"]["descriptions"][0]["value"] if "descriptions" in content["containers"]["cna"] else "n/a"
                finders = (
                    content["containers"]["cna"]["credits"][0]["value"] if "credits" in content["containers"]["cna"] and len(content["containers"]["cna"]["credits"]) > 0 else "n/a",
                    content["containers"]["cna"]["credits"][1]["value"] if "credits" in content["containers"]["cna"] and len(content["containers"]["cna"]["credits"]) > 1 else "n/a"
                )

                doc = Document(
                    page_content=cve_id,
                    metadata={
                        'cve': cve_id,
                        'Affected': affected,
                        'Descriptions': descriptions,
                        'Finder1': finders[0],
                        'Finder2': finders[1]
                    },
                )
                document_list.append(doc)
            except Exception as e:
                print(e)
                pass

        bm25 = BM25Retriever.from_documents(document_list)
        bm25.k = 1
        return bm25

    def find_cve(self, text):
        """Find CVEs in the given text."""
        if '–' in text:
            text = text.replace("–", "-")

        cve_pattern = re.compile(r'CVE-\d{4}-\d{4,7}')
        matches = re.findall(cve_pattern, text)
        return matches

    def cve_loader(self, suffix):
        """Load CVE documents from the filesystem."""
        python_loader = GenericLoader.from_filesystem(
            self.cve_path,
            glob="**/*",
            suffixes=[suffix],
            parser=LanguageParser(),
        )
        python_documents = python_loader.load()
        return python_documents

    def compression_retriever(self):
        """Build the compression retriever using multiple retrievers."""
        start_time = time.time()
        lotr = MergerRetriever(retrievers=self.retrievers)
        elapsed_time = time.time() - start_time
        print(f"***********************  TIME MERGER: {elapsed_time} ************************************")

        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        filter_ = EmbeddingsRedundantFilter(embeddings=self.get_paper_embeddings())
        reordering = LongContextReorder()
        relevant_filter = EmbeddingsFilter(embeddings=self.get_paper_embeddings(), similarity_threshold=0.5)
        pipeline = DocumentCompressorPipeline(transformers=[splitter, filter_, relevant_filter, reordering])

        compression_retriever_reordered = ContextualCompressionRetriever(
            base_compressor=pipeline, base_retriever=lotr, search_kwargs={"top_k": 5}
        )
        return compression_retriever_reordered

    def find_and_print_content(self, source_value):
        """Find and print content for a specific source value."""
        dataframe = self.result_df
        row = dataframe[dataframe['source'] == source_value]

        if not row.empty:
            print("Content for source value '{}' is:".format(source_value))
            return row['content'].iloc[0]
        else:
            print("Source value '{}' not found in the DataFrame.".format(source_value))
            return None

    def extract_context_around_content(self, document_content, content, context_size=1000):
        """Extract context around specific content within a document."""
        target_words = " ".join(content.split()[:4])
        content_position = document_content.find(target_words)

        if content_position != -1:
            start_index = max(0, content_position - context_size)
            end_index = min(len(document_content), content_position + len(content) + context_size)
            context_around_content = document_content[start_index:end_index]
            return context_around_content
        else:
            return None

    def handle_query(self, query):
        """Handle the query by finding and replacing CVEs in the text."""
        CVE_results = self.find_cve(query)

        if CVE_results:
            print("CVEs found:")
            for cve in CVE_results:
                print(cve)
                cve_retrieved = self.cve_retriever.get_relevant_documents(cve)[0]
                if cve == cve_retrieved.page_content:
                    cve_id = cve_retrieved.metadata["cve"]
                    description = cve_retrieved.metadata["Descriptions"]
                    print(cve_retrieved.metadata)
                    query = query.replace(cve, cve_id + ' ' + description)
        else:
            print("No CVEs found.")

        return query

    def run(self, original_query):
        """Run the cybersecurity assistant to retrieve relevant documents for the query."""
        try:
            self.logger.info(f"Running with query >> {original_query}")
            query = self.handle_query(original_query)
            final_content = ""
            for doc in self.compression_retriever_reordered.get_relevant_documents(query):
                final_content += doc.page_content + '\n\n'
                source = doc.metadata["source"]
                if source not in final_content:
                    print(source)
                    final_content += doc.page_content + '\n\n'
                    print("------------------------")

            print("\n\nFINAL CONTENT\n\n", final_content)
            self.logger.info("FINAL_CONTENT: " + final_content)
            return final_content

        except Exception as e:
            self.logger.error(f"An error occurred in run: {e}", exc_info=True)
            return None

# Main loop for handling queries
# if __name__ == "__main__":
#     cybersecurity_assistant = CybersecurityAssistant()
#     while True:
#         try:
#             query = input("\n\nINSERT A QUERY: ")
#             retrieval_start_time = time.time()  # Record the start time
#             cybersecurity_assistant.run(query)
#             retrieval_end_time = time.time()  # Record the end time for the document retrieval
#             elapsed_time = retrieval_end_time - retrieval_start_time

#             print(f"Total Time: {elapsed_time:.2f} seconds")

#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
