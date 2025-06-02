import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
import glob
import arxiv
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Set up the Streamlit app
st.set_page_config(page_title="Patent Analysis Agentic Workflow", layout="wide")

# Title and description
st.title("🔬 Patent Analysis Agentic Workflow")
st.markdown("""
This app analyzes recent patents in a specific research area (like Lithium Battery technology) 
using an agentic workflow with GEN AI principles. It will:
1. Fetch recent patents from arXiv
2. Process and analyze the documents
3. Predict future innovations
4. Recommend technologies to adopt
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    years = st.slider("Number of years to look back", 1, 5, 3)
    max_results = st.slider("Maximum number of patents to fetch", 10, 100, 50)
    
    default_keywords = "lithium battery, solid state electrolyte"
    keywords_input = st.text_area("Enter keywords (comma separated)", value=default_keywords)
    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
    
    if st.button("Run Analysis"):
        if not groq_api_key:
            st.error("Please enter your Groq API key")
            st.stop()

# Main app functionality
def main():
    if not keywords:
        st.info("Please enter keywords in the sidebar to begin analysis")
        return
    
    if 'GROQ_API_KEY' not in os.environ:
        st.error("Groq API key not found. Please enter it in the sidebar.")
        return
    
    try:
        # Initialize LLM
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Patent Fetching", 
        "Document Processing", 
        "Analysis", 
        "Predictions", 
        "Recommendations"
    ])
    
    with tab1:
        st.header("Patent Fetching")
        with st.spinner(f"Fetching patents for: {', '.join(keywords)}..."):
            try:
                papers_df = fetch_arxiv_papers(keywords, years=years, max_results=max_results)
            except Exception as e:
                st.error(f"Error fetching patents: {e}")
                st.stop()
        
        if not papers_df.empty:
            st.success(f"Found {len(papers_df)} patents")
            st.dataframe(papers_df)
            
            # Download PDFs
            with st.spinner("Downloading patent PDFs..."):
                try:
                    pdf_directory = download_pdfs(papers_df)
                    pdf_count = len(glob.glob(os.path.join(pdf_directory, '*.pdf')))
                    if pdf_count > 0:
                        st.success(f"Downloaded {pdf_count} PDFs")
                    else:
                        st.error("No PDFs were downloaded. Check the patent URLs.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error downloading PDFs: {e}")
                    st.stop()
        else:
            st.error("No patents found. Try different keywords.")
            st.stop()
    
    with tab2:
        st.header("Document Processing")
        try:
            with st.spinner("Processing documents..."):
                all_documents, db = process_documents(pdf_directory)
            
            if all_documents:
                st.success(f"Processed {len(all_documents)} document pages")
                st.info("Sample document content:")
                st.text(all_documents[0].page_content[:500] + "...")
                
                if db:
                    st.success("FAISS vector store created successfully")
                else:
                    st.error("Failed to create FAISS vector store")
                    st.stop()
            else:
                st.error("No documents were processed")
                st.stop()
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            st.stop()
    
    # Create the agentic workflow chains
    try:
        with st.spinner("Setting up agentic workflow..."):
            chains = setup_chains(llm, db)
    except Exception as e:
        st.error(f"Error setting up chains: {e}")
        st.stop()
    
    with tab3:
        st.header("Patent Analysis")
        if 'analysis_chain' in chains:
            try:
                with st.spinner("Analyzing patents..."):
                    analysis_result = chains['analysis_chain'].invoke(" ".join(keywords))
                
                st.subheader("Analysis Results")
                st.markdown(analysis_result)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
        else:
            st.error("Analysis chain not available")
    
    with tab4:
        st.header("Future Predictions")
        if 'analysis_result' in locals() and 'prediction_chain' in chains:
            try:
                with st.spinner("Predicting future trends..."):
                    prediction_result = chains['prediction_chain'].invoke(analysis_result)
                
                st.subheader("Predicted Future Developments")
                st.markdown(prediction_result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Prediction chain not available")
    
    with tab5:
        st.header("Recommendations")
        if 'prediction_result' in locals() and 'recommendation_chain' in chains:
            try:
                with st.spinner("Generating recommendations..."):
                    recommendation_result = chains['recommendation_chain'].invoke(prediction_result)
                
                st.subheader("Recommended Actions")
                st.markdown(recommendation_result)
            except Exception as e:
                st.error(f"Recommendation generation failed: {e}")
        else:
            st.error("Recommendation chain not available")

# Helper functions
def fetch_arxiv_papers(keywords, years=3, max_results=50):
    """Fetch papers from arXiv based on keywords and time range"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    query = " AND ".join([f'abs:"{kw}"' for kw in keywords])
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    client = arxiv.Client()
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "abstract": result.summary,
                "published": result.published,
                "authors": [a.name for a in result.authors],
                "doi": result.entry_id,
                "pdf_url": result.pdf_url
            })
    except Exception as e:
        raise Exception(f"Error fetching papers from arXiv: {e}")
    
    return pd.DataFrame(papers)

def download_pdfs(papers_df):
    """Download PDFs from the papers dataframe"""
    pdf_directory = 'patent_dump'
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
    
    if not papers_df.empty:
        for index, row in papers_df.iterrows():
            pdf_url = row['pdf_url']
            if pdf_url:
                try:
                    response = requests.get(pdf_url, timeout=10)
                    filename_base = row['title'].replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]  # Limit filename length
                    filename = os.path.join(pdf_directory, f"{filename_base}_{index}.pdf")
                    
                    if response.status_code == 200:
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                except Exception as e:
                    print(f"Warning: Failed to download {pdf_url}: {e}")  # Print to console instead of showing in UI
    
    return pdf_directory

def process_documents(pdf_directory):
    """Process PDF documents and create embeddings"""
    all_documents = []
    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    
    if pdf_files:
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Error loading {pdf_file}: {e}")  # Print to console
    
    # Initialize the embedding model
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise Exception(f"Error initializing embedding model: {e}")
    
    # Create FAISS index
    db = None
    if all_documents and embeddings is not None:
        try:
            db = FAISS.from_documents(all_documents, embeddings)
            db_path = "faiss_index"
            db.save_local(db_path)
        except Exception as e:
            raise Exception(f"Error creating FAISS index: {e}")
    
    return all_documents, db

def setup_chains(llm, db):
    """Set up the agentic workflow chains"""
    chains = {}
    
    # Data Processing Chain (conceptual)
    chains['data_processing_chain'] = (
        PromptTemplate.from_template("Data processing complete. Documents are ready for analysis.")
        | llm
        | StrOutputParser()
    )
    
    # Analysis Chain
    analysis_prompt = PromptTemplate.from_template(
        """Analyze the provided patent documents and summarize the key technological advancements and trends related to the keywords used for fetching.

        Documents:
        {documents}

        Analysis:
        """
    )
    
    def format_docs(docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)
    
    if db is not None:
        retriever = db.as_retriever(search_kwargs={"k": 5})  # Limit to top 5 relevant documents
        
        chains['analysis_chain'] = (
            {"documents": retriever | format_docs}
            | analysis_prompt
            | llm
            | StrOutputParser()
        )
    
    # Prediction Chain
    if 'analysis_chain' in chains:
        prediction_prompt = PromptTemplate.from_template(
            """Based on the provided analysis of patent trends, predict potential future developments and breakthrough areas in the field.

            Analysis:
            {analysis}

            Predictions:
            """
        )
        
        chains['prediction_chain'] = (
            {"analysis": RunnablePassthrough()}
            | prediction_prompt
            | llm
            | StrOutputParser()
        )
    
    # Recommendation Chain
    if 'prediction_chain' in chains:
        recommendation_prompt = PromptTemplate.from_template(
            """Based on the predicted future developments and breakthroughs, recommend potential research directions or business opportunities.

            Predictions:
            {predictions}

            Recommendations:
            """
        )
        
        chains['recommendation_chain'] = (
            {"predictions": RunnablePassthrough()}
            | recommendation_prompt
            | llm
            | StrOutputParser()
        )
    
    return chains

if __name__ == "__main__":
    main()