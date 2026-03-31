"""
RAG Bot Demo with validation and error handling
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from config import config
from validators import InputValidator, OutputValidator, ValidationError
from rag.loader import load_documents
from rag.vector_store import build_vector_store, load_vector_store
from rag.rag_chain import build_rag_chain

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class RAGBotDemo:
    """Main RAG Bot Demo class"""

    def __init__(self, pdf_path: Optional[str] = None):
        """
        Initialize RAG Bot
        
        Args:
            pdf_path: Path to document file (PDF or TXT) - defaults to document.txt
        """
        # Default to document.txt instead of sample_doc.pdf
        self.pdf_path = pdf_path or str(config.DEFAULT_DOCUMENT_PATH)
        self.qa_chain = None
        self.vectordb = None

    def setup(self) -> bool:
        """
        Setup RAG Bot
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Validate configuration
            logger.info("Validating configuration...")
            config.validate()
            logger.info("Configuration validated successfully")

            # Check if vector store exists
            if config.CHROMA_DB_DIR.exists():
                logger.info("Loading existing vector store...")
                self.vectordb = load_vector_store()
            elif self.pdf_path:
                logger.info(f"Loading documents from {self.pdf_path}...")
                chunks = load_documents(self.pdf_path)
                logger.info("Building vector store...")
                self.vectordb = build_vector_store(chunks)
            else:
                logger.error("No vector store found and no PDF path provided")
                return False

            # Build RAG chain
            logger.info("Building RAG chain...")
            self.qa_chain = build_rag_chain(self.vectordb)
            logger.info("RAG Bot setup completed successfully")

            return True

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            return False

    def process_question(self, question: str) -> Optional[dict]:
        """
        Process user question through RAG chain
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources, or None if failed
        """
        # Validate question
        is_valid, error_msg = InputValidator.validate_question(question)
        if not is_valid:
            logger.warning(f"Invalid question: {error_msg}")
            print(f" Invalid question: {error_msg}")
            return None

        try:
            logger.debug(f"Processing question: {question[:100]}...")
            response = self.qa_chain.invoke({"query": question})

            # Validate answer
            answer = response.get("result", "")
            is_valid, error_msg = OutputValidator.validate_answer(answer)

            if not is_valid:
                logger.warning(f"Answer validation warning: {error_msg}")
                print(f"  {error_msg}")
            else:
                logger.debug("Answer validated successfully")

            return response

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f" Error processing question: {e}")
            return None

    def display_response(self, response: dict):
        """Display response with formatting"""
        if not response:
            return

        answer = response.get("result", "")
        sources = response.get("source_documents", [])

        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)

        if sources:
            print("\n" + "=" * 80)
            print("SOURCES:")
            print("=" * 80)
            for i, doc in enumerate(sources, 1):
                metadata = doc.metadata
                print(f"\n{i}. Source: {metadata.get('source', 'Unknown')}")
                if 'page' in metadata:
                    print(f"   Page: {metadata['page']}")
                print(f"   Content preview: {doc.page_content[:200]}...")
        print("\n" + "=" * 80)

    def interactive_mode(self):
        """Run interactive question-answer mode"""
        print("\n" + "=" * 80)
        print("RAG BOT - Interactive Mode")
        print("=" * 80)
        print("Type 'exit' or 'quit' to end conversation")
        print("Type 'stats' to see bot statistics")
        print("=" * 80 + "\n")

        question_count = 0

        while True:
            try:
                question = input("\n❓ Ask a question: ").strip()

                if not question:
                    continue

                if question.lower() in ["exit", "quit"]:
                    logger.info("User exited interactive mode")
                    print("\nGoodbye!")
                    break

                if question.lower() == "stats":
                    print(f"\nStatistics:")
                    print(f"   Questions asked: {question_count}")
                    continue

                question_count += 1
                response = self.process_question(question)

                if response:
                    self.display_response(response)

            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted by user")
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}", exc_info=True)
                print(f" An error occurred: {e}")

    def run_single_question(self, question: str):
        """Run a single question"""
        logger.info(f"Running single question: {question}")
        response = self.process_question(question)

        if response:
            self.display_response(response)
            return True
        return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Bot with validation and error handling"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to load",
        default=None
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (optional)",
        default=None
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector store and reload from PDF"
    )

    args = parser.parse_args()

    try:
        # Handle reset
        if args.reset:
            logger.info("Resetting vector store...")
            from rag.vector_store import delete_vector_store
            delete_vector_store()

        # Initialize bot
        pdf_path = args.pdf or "data/documents/sample_doc.pdf"
        bot = RAGBotDemo(pdf_path=pdf_path)

        # Setup
        if not bot.setup():
            logger.error("Failed to setup RAG Bot")
            sys.exit(1)

        # Run
        if args.question:
            bot.run_single_question(args.question)
        else:
            bot.interactive_mode()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f" Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
