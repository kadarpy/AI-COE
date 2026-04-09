"""
Document loading module for RAG Bot - supports PDF and TXT files
"""
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
from validators import InputValidator, DocumentValidator, ValidationError

logger = logging.getLogger(__name__)

def load_documents(file_path: str):
    """
    Load and process documents from PDF or TXT file
    
    Args:
        file_path: Path to PDF or TXT file
        
    Returns:
        List of document chunks
        
    Raises:
        ValidationError: If file path or content is invalid
        Exception: If document loading fails
    """
    logger.info(f"Starting document load from: {file_path}")

    # Validate file path
    is_valid, error_msg = InputValidator.validate_file_path(file_path)
    if not is_valid:
        logger.error(f"File validation failed: {error_msg}")
        raise ValidationError(f"Invalid file path: {error_msg}")

    try:
        # Determine file type and load accordingly
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".pdf":
            logger.debug(f"Loading PDF from {file_path}")
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            logger.debug(f"Loading TXT from {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValidationError(f"Unsupported file type: {file_extension}. Use .pdf or .txt")
        
        docs = loader.load()

        if not docs:
            logger.error("No documents loaded from file")
            raise ValidationError("File appears to be empty or invalid")

        logger.info(f"Successfully loaded {len(docs)} documents from {file_extension} file")

        # Split into chunks
        logger.debug(f"Splitting documents with chunk_size={config.PDF_CHUNK_SIZE}, "
                    f"overlap={config.PDF_CHUNK_OVERLAP}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PDF_CHUNK_SIZE,
            chunk_overlap=config.PDF_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        # Attach basic metadata (page-based fallback)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = f"page_{chunk.metadata.get('page', i)}"

        if not chunks:
            logger.error("No chunks created from documents")
            raise ValidationError("Failed to create document chunks")

        # Validate chunks
        is_valid, error_msg = DocumentValidator.validate_chunks(chunks)
        if not is_valid:
            logger.error(f"Chunk validation failed: {error_msg}")
            raise ValidationError(f"Document chunk validation failed: {error_msg}")

        logger.info(f"Successfully created {len(chunks)} document chunks")
        logger.debug(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

        return chunks

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        raise


def load_documents_from_directory(directory_path: str):
    """
    Load and process all documents from a directory.
    Recursively scans for PDF and TXT files.
    
    Args:
        directory_path: Path to directory containing documents
        
    Returns:
        List of merged and chunked documents
        
    Raises:
        ValidationError: If directory is invalid
        Exception: If document loading fails
    """
    logger.info(f"Starting document load from directory: {directory_path}")
    
    # Validate directory
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValidationError(f"Directory not found: {directory_path}")
    if not dir_path.is_dir():
        raise ValidationError(f"Path is not a directory: {directory_path}")
    
    try:
        # Find all PDF and TXT files
        supported_extensions = {'.pdf', '.txt', '.md'}
        files_to_load = []
        
        for ext in supported_extensions:
            files_to_load.extend(dir_path.rglob(f"*{ext}"))
        
        if not files_to_load:
            logger.warning(f"No supported documents found in {directory_path}")
            raise ValidationError(f"No .pdf, .txt, or .md files found in {directory_path}")
        
        logger.info(f"Found {len(files_to_load)} documents to load")
        
        # Load all documents
        all_docs = []
        for file_path in sorted(files_to_load):
            try:
                logger.debug(f"Loading {file_path}")
                file_ext = file_path.suffix.lower()
                
                if file_ext == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif file_ext in [".txt", ".md"]:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                else:
                    logger.warning(f"Skipping unsupported file type: {file_path}")
                    continue
                
                docs = loader.load()
                
                # Tag documents with source file
                for doc in docs:
                    doc.metadata["source_file"] = str(file_path.name)
                    doc.metadata["source_path"] = str(file_path)
                
                all_docs.extend(docs)
                logger.debug(f"Loaded {len(docs)} documents from {file_path.name}")
            
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_docs:
            raise ValidationError("No documents could be loaded from directory")
        
        logger.info(f"Successfully loaded {len(all_docs)} total documents from all files")
        
        # Split into chunks
        logger.debug(f"Splitting documents with chunk_size={config.PDF_CHUNK_SIZE}, "
                    f"overlap={config.PDF_CHUNK_OVERLAP}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PDF_CHUNK_SIZE,
            chunk_overlap=config.PDF_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(all_docs)
        
        # Attach metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = i
        
        if not chunks:
            raise ValidationError("Failed to create document chunks")
        
        # Validate chunks
        is_valid, error_msg = DocumentValidator.validate_chunks(chunks)
        if not is_valid:
            logger.error(f"Chunk validation failed: {error_msg}")
            raise ValidationError(f"Document chunk validation failed: {error_msg}")
        
        logger.info(f"Successfully created {len(chunks)} document chunks from {len(all_docs)} documents")
        avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
        logger.debug(f"Average chunk size: {avg_size} characters")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error loading documents from directory: {str(e)}", exc_info=True)
        raise