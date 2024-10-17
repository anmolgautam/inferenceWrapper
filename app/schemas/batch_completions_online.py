from typing import List
from pydantic import BaseModel
from completions import CompletionRequest, CompletionResponse

class BatchRequest(BaseModel):
    requests: List[CompletionRequest]

class BatchResponse(BaseModel):
    responses: List[CompletionResponse]
