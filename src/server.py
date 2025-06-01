from fastapi import FastAPI, HTTPException
from typing import Dict, Any , List
import os
import sys
from file_pipeline import add_file_pipeline 
from get_relevant_files import semantic_search_engine
from pydantic import BaseModel

# Add the directory containing your semantic search module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)



# Create FastAPI app
app = FastAPI(
    title="File Processing API",
    description="API for processing files through AI pipeline",
    version="1.0"
)

@app.post("/upload_new_file", response_model=Dict[str, Any])
async def upload_new_file(file_path: str):
    """
    Process a file through the AI pipeline
    
    - **file_path**: Relative path to the file in the File System Simulation directory
    """
    try:
        # Call your processing pipeline
        result = add_file_pipeline(file_path)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="File processing failed. Check server logs for details."
            )
        
        return {
            "status": "success",
            "file_id": result.get("file_id"),
            "file_type": result.get("file_type"),
            "content_summary": result.get("content"),
            "storage_path": result.get("storage_path")
        }
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )



app = FastAPI(
    title="Semantic File Search API",
    description="API for searching files using semantic similarity",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    search_query: str

class SearchResponse(BaseModel):
    results: List[str]
    count: int

@app.post("/semantic_search", response_model=SearchResponse, summary="Search files by semantic similarity")
async def search_files(request: SearchRequest):
    """
    Search for files based on semantic similarity to a text query
    
    - **search_query**: Text description of what you're looking for
    """
    try:
        # Call your semantic search engine
        results = semantic_search_engine(request.search_query)
        
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)