from fastapi import APIRouter, HTTPException
from app.schemas.completions import CompletionRequest, CompletionResponse
import requests
import os
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
VLLM_COMPLETION_ENDPOINT = os.getenv("VLLM_COMPLETION_ENDPOINT")

@router.post("/v1/completions",response_model=CompletionResponse)
async def create_completions(request_data:CompletionRequest):
    print("HITIING ENDPOINT : ",VLLM_COMPLETION_ENDPOINT)
    try:
        response = requests.post(VLLM_COMPLETION_ENDPOINT,json=request_data.dict())
        if response.status_code !=200:
            raise HTTPException(status_code=response.status_code,detail=response.text)
        return CompletionResponse(**response.json())
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
