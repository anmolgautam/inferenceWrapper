import logging 
import time
import uuid
import aiofiles
from pythonjsonlogger import jsonlogger

logging.basicConfig(level=logging.INFO,format="%(asctime)s - [%(levelname)s] - %(message)s")

def generate_request_id() -> str:
    return str(uuid.uuid4())

async def log_request_response(request, call_next):
    request_id = generate_request_id()
    start_time = time.time()
    logging.info(f"Request ID: {request_id}, Method: {request.method}, URL: {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"Request ID: {request_id}, Status: {response.status_code}, Time: {process_time:.4f}s")
    response.headers["X-Request-ID"] = request_id
    return response

