from typing import List, Optional, Dict
from pydantic import BaseModel

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class Logprob(BaseModel):
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None