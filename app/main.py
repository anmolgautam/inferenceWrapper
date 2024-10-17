from fastapi import FastAPI
from app.routes import completions
from app.utils.logging import log_request_response
app = FastAPI()

app.middleware("http")(log_request_response)
app.include_router(completions.router)
