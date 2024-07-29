
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
import os
from typing import Dict

from vllm import LLM,SamplingParams
from vllm.outputs import RequestOutput
import time
import datetime
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel

class RequestBody(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    top_k:int=-1
    max_tokens: int = 256
    min_tokens:  int = 0


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.post(
        "/",
        responses={200: {"content": {}}},
        response_class=JSONResponse,
    )
    async def generate(self, body: RequestBody):
        start_timestamp = datetime.datetime.now().isoformat()
        start_time = time.time()
        output_data:RequestOutput = await self.handle.generate.remote(body)
        metrics = output_data.metrics
        generated_text = output_data.outputs[0].text
        end_time = time.time()
        completion_timestamp = datetime.datetime.now().isoformat()
        resp = {
            "completed_at": completion_timestamp,
            "created_at": start_timestamp,
            "error": None,
            "input":body.dict(),
            "metrics": {
                "total_time": end_time-start_time,
                "input_token_count": len(output_data.prompt_token_ids),
                "tokens_per_second": len(output_data.outputs[0].token_ids)/(metrics.finished_time-metrics.first_scheduled_time),
                "output_token_count": len(output_data.outputs[0].token_ids),
                "predict_time": metrics.finished_time-metrics.first_scheduled_time,
                "time_to_first_token":metrics.first_token_time- metrics.arrival_time
            },
            "output": [
                generated_text
            ],
            "started_at": start_timestamp,
            "status": "succeeded"
                    
                }
        return JSONResponse(content=resp)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class LLMApplication:
    def __init__(self,model_id:str):
        os.environ["HF_TOKEN"]="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"
        print("n\n","model_id",model_id,"\n\n")
        self.model = LLM(model=model_id)
    def generate(self, body:RequestBody):
        prompts = []
        prompts.append(body.prompt)
        sampling_params=SamplingParams(temperature=body.temperature,top_p=body.top_p,min_tokens=body.min_tokens,max_tokens=body.max_tokens,top_k=body.top_k)
        output = self.model.generate(prompts,sampling_params)
        return output[0]
        
def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.options(route_prefix=args["route_prefix"]).bind(LLMApplication.bind(args["model_id"]))


