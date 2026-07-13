#!/usr/bin/env python3
"""
Retrieval module for FinanceBench RAG pipeline.
Handles embedding generation, indexing, and similarity search.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
import pickle
import hashlib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding models and encoding."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize embedding model.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = 32,
                     normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings to unit length

        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        # Some models require specific prefixes
        if "e5" in self.model_name.lower():
            # E5 models require "passage: " prefix for documents
            texts = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: Query string to encode

        Returns:
            Numpy array of query embedding
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        # Some models require specific prefixes for queries
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding[0]


class VectorStore:
    """Base class for vector storage implementations."""

    def __init__(self, dimension: int):
        """
        Initialize vector store.

        Args:
            dimension: Dimensionality of vectors
        """
        self.dimension = dimension
        self.is_trained = False

    def add_vectors(self, vectors: np.ndarray, ids: List[str],
                    metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Array of vectors to add
            ids: List of string IDs for each vector
            metadata: List of metadata dictionaries
        """
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            Tuple of (ids, scores, metadata)
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the index to disk."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load the index from disk."""
        raise NotImplementedError


class FAISSStore(VectorStore):
    """FAISS-based vector store."""

    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """
        Initialize FAISS store.

        Args:
            dimension: Dimensionality of vectors
            index_type: Type of FAISS index to use
        """
        super().__init__(dimension)
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Maps internal index to external ID
        self.metadata_map = {}  # Maps external ID to metadata

        # Initialize index
        if index_type == "IndexFlatIP":
            # Inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            # L2 distance
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IndexIVFFlat":
            # IVF with flat quantizer (needs training)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids
            self.index.train(np.random.random((1000, dimension)).astype('float32'))
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.is_trained = True  # Flat indices don't need training

    def add_vectors(self, vectors: np.ndarray, ids: List[str],
                    metadata: List[Dict[str, Any]]) -> None:
        """Add vectors to FAISS index."""
        if not self.is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # Train if needed (for IVF indices)
            self.index.train(vectors)

        # Add vectors to index
        start_idx = self.index.ntotal
        self.index.add(vectors.astype('float32'))

        # Store ID and metadata mappings
        for i, (doc_id, meta) in enumerate(zip(ids, metadata)):
            internal_id = start_idx + i
            self.id_map[internal_id] = doc_id
            self.metadata_map[doc_id] = meta

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search for similar vectors using FAISS."""
        if self.index.ntotal == 0:
            return [], [], []

        # Ensure query is 2D
        query_vector = query_vector.reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        # Convert results
        ids = []
        scores_list = []
        metadata_list = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            doc_id = self.id_map.get(int(idx))
            if doc_id:
                ids.append(doc_id)
                scores_list.append(float(score))
                metadata_list.append(self.metadata_map.get(doc_id, {}))

        return ids, scores_list, metadata_list

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        # Save index
        faiss.write_index(self.index, f"{path}.index")

        # Save mappings
        with open(f"{path}_id_map.pkl", 'wb') as f:
            pickle.dump(self.id_map, f)

        with open(f"{path}_metadata_map.pkl", 'wb') as f:
            pickle.dump(self.metadata_map, f)

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        # Load index
        self.index = faiss.read_index(f"{path}.index")

        # Load mappings
        with open(f"{path}_id_map.pkl", 'rb') as f:
            self.id_map = pickle.load(f)

        with open(f"{path}_metadata_map.pkl", 'rb') as f:
            self.metadata_map = pickle.load(f)

        self.is_trained = True


class ChromaStore(VectorStore):
    """ChromaDB-based vector store."""

    def __init__(self, collection_name: str = "financebench_collection",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
        """
        super().__init__(dimension=0)  # Dimension will be set on first add
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

    def add_vectors(self, vectors: np.ndarray, ids: List[str],
                    metadata: List[Dict[str, Any]]) -> None:
        """Add vectors to ChromaDB collection."""
        # Set dimension from first vector if not set
        if self.dimension == 0 and len(vectors) > 0:
            self.dimension = vectors.shape[1]

        # Prepare data for ChromaDB
        documents = [meta.get("text", "") for meta in metadata]
        metadatas = []
        for meta in metadata:
            # ChromaDB requires string values for metadata
            flat_meta = {}
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flat_meta[key] = value
                else:
                    flat_meta[key] = str(value)
            metadatas.append(flat_meta)

        # Add to collection
        self.collection.add(
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search for similar vectors using ChromaDB."""
        if self.collection.count() == 0:
            return [], [], []

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, self.collection.count()),
            include=['distances', 'metadatas']
        )

        # Extract results
        ids = results['ids'][0] if results['ids'] else []
        # ChromaDB returns distances, convert to similarity scores (1 - distance for normalized vectors)
        distances = results['distances'][0] if results['distances'] else []
        scores = [1.0 - d for d in distances]  # Convert distance to similarity
        metadata_list = results['metadatas'][0] if results['metadatas'] else []

        return ids, scores, metadata_list

    def save(self, path: str) -> None:
        """ChromaDB persists automatically, but we can save metadata."""
        # ChromaDB persists to persist_directory automatically
        pass

    def load(self, path: str) -> None:
        """Load is handled by initialization with persist_directory."""
        pass


def hash_string_to_filename(s: str) -> str:
    """Convert a string to a safe filename using MD5 hash."""
    return hashlib.md5(s.encode()).hexdigest()


def create_vector_store(store_type: str = "faiss", **kwargs) -> VectorStore:
    """
    Factory function to create vector store instances.

    Args:
        store_type: Type of store ('faiss' or 'chroma')
        **kwargs: Additional arguments for specific store types

    Returns:
        VectorStore instance
    """
    if store_type.lower() == "faiss":
        return FAISSStore(**kwargs)
    elif store_type.lower() == "chroma":
        return ChromaStore(**kwargs)
    else:
        raise ValueError(f"Unsupported store type: {store_type}")


def load_chunks_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Load all chunk JSONL files from a directory.

    Args:
        directory: Directory containing chunk JSONL files

    Returns:
        List of all chunk dictionaries
    """
    all_chunks = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return all_chunks

    # Find all JSONL files
    jsonl_files = list(directory_path.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} chunk files to load")

    for file_path in jsonl_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    chunk = json.loads(line.strip())
                    all_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Loaded {len(all_chunks)} total chunks")
    return all_chunks


def build_index(chunks: List[Dict[str, Any]], embedding_model: str = "BAAI/bge-base-en-v1.5",
                store_type: str = "faiss", index_name: str = "financebench_index",
                batch_size: int = 32) -> VectorStore:
    """
    Build a vector index from document chunks.

    Args:
        chunks: List of chunk dictionaries
        embedding_model: Name of sentence-transformers model to use
        store_type: Type of vector store ('faiss' or 'chroma')
        index_name: Name for the index/files
        batch_size: Batch size for embedding generation

    Returns:
        Populated VectorStore instance
    """
    if not chunks:
        raise ValueError("No chunks provided for indexing")

    logger.info(f"Building index from {len(chunks)} chunks using {embedding_model}")

    # Initialize embedding model
    embedder = EmbeddingManager(model_name=embedding_model)

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadata = chunks  # Store full chunk as metadata

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedder.encode_texts(texts, batch_size=batch_size)

    # Create vector store
    if store_type.lower() == "faiss":
        dimension = embeddings.shape[1]
        vector_store = create_vector_store(
            store_type=store_type,
            dimension=dimension,
            index_type="IndexFlatIP"  # Inner product for cosine similarity
        )
    elif store_type.lower() == "chroma":
        vector_store = create_vector_store(
            store_type=store_type,
            collection_name=index_name,
            persist_directory=f"./{index_name}_chroma_db"
        )
    else:
        raise ValueError(f"Unsupported store type: {store_type}")

    # Add vectors to store
    logger.info(f"Adding {len(embeddings)} vectors to {store_type} store...")
    vector_store.add_vectors(embeddings, ids, metadata)

    # Save index
    logger.info(f"Saving index to {index_name}...")
    vector_store.save(index_name)

    return vector_store


def load_index(index_name: str = "financebench_index",
               store_type: str = "faiss") -> VectorStore:
    """
    Load a previously saved vector index.

    Args:
        index_name: Name of the index to load
        store_type: Type of vector store ('faiss' or 'chroma')

    Returns:
        Loaded VectorStore instance
    """
    logger.info(f"Loading {store_type} index from {index_name}...")
    vector_store = create_vector_store(store_type=store_type)

    if store_type.lower() == "faiss":
        vector_store.load(index_name)
    elif store_type.lower() == "chroma":
        # For Chroma, we just need to initialize with the same parameters
        vector_store = create_vector_store(
            store_type=store_type,
            collection_name=index_name,
            persist_directory=f"./{index_name}_chroma_db"
        )
    else:
        raise ValueError(f"Unsupported store type: {store_type}")

    return vector_store


def search_index(vector_store: VectorStore, query: str,
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the vector index for a query.

    Args:
        vector_store: Vector store to search
        query: Query string
        embedding_model: Embedding model to use for query encoding
        k: Number of results to return

    Returns:
        List of search result dictionaries
    """
    # Initialize embedding model (same as used for indexing)
    embedder = EmbeddingManager(model_name=embedding_model)

    # Encode query
    query_vector = embedder.encode_query(query)

    # Search
    ids, scores, metadata = vector_store.search(query_vector, k=k)

    # Format results
    results = []
    for uid, score, meta in zip(ids, scores, metadata):
        result = {
            "chunk_id": uid,
            "score": score,
            "text": meta.get("text", ""),
            "doc_name": meta.get("doc_name", ""),
            "company": meta.get("company", ""),
            "chunk_strategy": meta.get("chunk_strategy", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "metadata": meta
        }
        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage: build index from processed chunks
    chunks = load_chunks_from_directory("data/processed_chunks")
    if chunks:
        vector_store = build_index(chunks, store_type="faiss")
        print(f"Built index with {len(chunks)} vectors")
    else:
        print("No chunks found to index")