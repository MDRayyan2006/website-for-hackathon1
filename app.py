# app.py - Main Flask Backend
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import json
from datetime import datetime
import uuid
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence transformer: {e}")
    model = None

# Global storage for processed documents
processed_documents = {}
document_embeddings = {}

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 500  # Characters per chunk
        self.overlap = 50      # Overlap between chunks
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def chunk_text(self, text):
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
        
        return [chunk for chunk in chunks if chunk]
    
    def create_embeddings(self, chunks):
        """Create embeddings for text chunks"""
        if not model:
            return None
        
        try:
            embeddings = model.encode(chunks)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def process_pdf(self, pdf_path, filename):
        """Complete PDF processing pipeline"""
        logger.info(f"Processing PDF: {filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Create chunks
        chunks = self.chunk_text(text)
        if not chunks:
            return None
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Create document data
        doc_data = {
            'id': str(uuid.uuid4()),
            'filename': filename,
            'text': text,
            'chunks': chunks,
            'processed_at': datetime.now().isoformat(),
            'chunk_count': len(chunks)
        }
        
        return doc_data, embeddings

class QuestionAnswerer:
    def __init__(self):
        self.max_context_chunks = 3
    
    def find_relevant_chunks(self, question, doc_id):
        """Find most relevant chunks for a question"""
        if doc_id not in processed_documents or doc_id not in document_embeddings:
            return []
        
        if not model:
            # Fallback: simple keyword matching
            return self.keyword_search(question, doc_id)
        
        try:
            # Create question embedding
            question_embedding = model.encode([question])
            
            # Get document embeddings
            doc_embeddings = document_embeddings[doc_id]
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
            
            # Get top chunks
            top_indices = np.argsort(similarities)[-self.max_context_chunks:][::-1]
            
            chunks = processed_documents[doc_id]['chunks']
            relevant_chunks = [(chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
            
            return relevant_chunks
        
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            return self.keyword_search(question, doc_id)
    
    def keyword_search(self, question, doc_id):
        """Fallback keyword-based search"""
        if doc_id not in processed_documents:
            return []
        
        question_words = set(question.lower().split())
        chunks = processed_documents[doc_id]['chunks']
        
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:self.max_context_chunks]
    
    def generate_answer(self, question, relevant_chunks):
        """Generate answer based on relevant chunks"""
        if not relevant_chunks:
            return "I couldn't find relevant information in your documents to answer this question."
        
        # Combine relevant chunks
        context = "\n\n".join([chunk[0] for chunk in relevant_chunks])
        
        # Simple rule-based response generation
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
            return self.generate_summary(context)
        elif any(word in question_lower for word in ['what is', 'define', 'definition']):
            return self.generate_definition(question, context)
        elif any(word in question_lower for word in ['how', 'explain', 'why']):
            return self.generate_explanation(question, context)
        elif any(word in question_lower for word in ['example', 'examples', 'instance']):
            return self.generate_examples(context)
        else:
            return self.generate_general_answer(question, context)
    
    def generate_summary(self, context):
        """Generate a summary response"""
        sentences = re.split(r'[.!?]+', context)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        if not key_sentences:
            return "The document appears to contain limited content for summarization."
        
        summary = "Based on your document, here are the key points:\n\n"
        for i, sentence in enumerate(key_sentences, 1):
            summary += f"{i}. {sentence}.\n"
        
        return summary
    
    def generate_definition(self, question, context):
        """Generate a definition response"""
        # Extract the term being asked about
        term_patterns = [r'what is (.*?)\?', r'define (.*?)[\?\.]', r'definition of (.*?)[\?\.]']
        term = None
        
        for pattern in term_patterns:
            match = re.search(pattern, question.lower())
            if match:
                term = match.group(1).strip()
                break
        
        if term:
            return f"Based on your document, here's information about '{term}':\n\n{context[:300]}..."
        else:
            return f"Based on your document:\n\n{context[:300]}..."
    
    def generate_explanation(self, question, context):
        """Generate an explanation response"""
        return f"Here's an explanation based on your document:\n\n{context[:400]}...\n\nThis information directly relates to your question about the topic."
    
    def generate_examples(self, context):
        """Generate examples from context"""
        return f"Here are relevant examples from your document:\n\n{context[:350]}..."
    
    def generate_general_answer(self, question, context):
        """Generate a general answer"""
        return f"Based on your document, here's what I found relevant to your question:\n\n{context[:400]}..."

# Initialize processors
pdf_processor = PDFProcessor()
qa_system = QuestionAnswerer()

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process PDF
        doc_data, embeddings = pdf_processor.process_pdf(filepath, file.filename)
        
        if not doc_data:
            os.remove(filepath)  # Clean up failed upload
            return jsonify({'error': 'Failed to process PDF'}), 500
        
        # Store processed data
        doc_id = doc_data['id']
        processed_documents[doc_id] = doc_data
        
        if embeddings is not None:
            document_embeddings[doc_id] = embeddings
        
        # Save processed data to file
        processed_path = os.path.join(PROCESSED_FOLDER, f"{doc_id}.json")
        with open(processed_path, 'w') as f:
            json.dump({k: v for k, v in doc_data.items() if k != 'chunks'}, f)
        
        logger.info(f"Successfully processed PDF: {file.filename}")
        
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'filename': file.filename,
            'chunk_count': doc_data['chunk_count'],
            'text_length': len(doc_data['text'])
        })
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/question', methods=['POST'])
def answer_question():
    """Handle question answering"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        document_ids = data.get('document_ids', [])
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not document_ids:
            document_ids = list(processed_documents.keys())
        
        if not document_ids:
            return jsonify({'error': 'No documents available'}), 400
        
        # Find relevant chunks from all documents
        all_relevant_chunks = []
        for doc_id in document_ids:
            if doc_id in processed_documents:
                chunks = qa_system.find_relevant_chunks(question, doc_id)
                all_relevant_chunks.extend(chunks)
        
        # Sort by relevance score and take top chunks
        if all_relevant_chunks and len(all_relevant_chunks[0]) > 1:
            all_relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = all_relevant_chunks[:3]
        else:
            top_chunks = all_relevant_chunks[:3]
        
        # Generate answer
        answer = qa_system.generate_answer(question, top_chunks)
        
        logger.info(f"Answered question: {question[:50]}...")
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources_count': len(top_chunks),
            'question': question
        })
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all processed documents"""
    try:
        documents = []
        for doc_id, doc_data in processed_documents.items():
            documents.append({
                'id': doc_id,
                'filename': doc_data['filename'],
                'processed_at': doc_data['processed_at'],
                'chunk_count': doc_data['chunk_count']
            })
        
        return jsonify({
            'success': True,
            'documents': documents
        })
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a processed document"""
    try:
        if doc_id not in processed_documents:
            return jsonify({'error': 'Document not found'}), 404
        
        # Remove from memory
        filename = processed_documents[doc_id]['filename']
        del processed_documents[doc_id]
        
        if doc_id in document_embeddings:
            del document_embeddings[doc_id]
        
        # Remove processed file
        processed_path = os.path.join(PROCESSED_FOLDER, f"{doc_id}.json")
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        logger.info(f"Deleted document: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'Document {filename} deleted successfully'
        })
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_processed': len(processed_documents),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting StudyMate Backend Server...")
    print("📚 Features enabled:")
    print("   - PDF text extraction (PyMuPDF)")
    print("   - Semantic search (Sentence Transformers)" if model else "   - Keyword search (fallback)")
    print("   - Question answering system")
    print("   - File upload and management")
    print("\n🌐 Server will be available at: http://localhost:5000")
    print("📄 Make sure your frontend points to this URL for API calls")
    if __name__=="__main__":
     app.run(debug=True, port=5000, host='0.0.0.0')
