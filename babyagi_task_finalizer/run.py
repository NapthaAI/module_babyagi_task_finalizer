import os
import yaml
import instructor
from litellm import Router
from babyagi_task_finalizer.schemas import InputSchema, TaskFinalizer
from babyagi_task_finalizer.utils import get_logger


logger = get_logger(__name__)

client = instructor.patch(
    Router(
        model_list=
        [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ],
        # default_litellm_params={"acompletion": True},
    )
)

def llm_call(messages, response_model=None):
    if response_model:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=response_model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
        )
    return response

def run(inputs: InputSchema, worker_nodes = None, orchestrator_node = None, flow_run = None, cfg: dict = None):
    logger.info(f"Running with inputs {inputs.objective}")
    logger.info(f"Running with inputs {inputs.task}")

    user_prompt = cfg["inputs"]["user_message_template"].replace("{{task}}", inputs.task).replace("{{objective}}", inputs.objective)

    messages = [
        {"role": "system", "content": cfg["inputs"]["system_message"]},
        {"role": "user", "content": user_prompt}
    ]

    response = llm_call(messages, response_model=TaskFinalizer)

    logger.info(f"Result: {response}")

    return response.model_dump_json()


if __name__ == "__main__":
    with open("babyagi_task_finalizer/component.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    inputs = InputSchema(
        task="Weather pattern between year 1900 and 2000 was cool andry", 
        objective="Write a blog post about the weather in London."
    )

    r = run(inputs, cfg=cfg)
    logger.info(f"Result: {type(r)}")


    import json
    t = TaskFinalizer(**json.loads(r))
    logger.info(f"Final report: {type(t)}")

