from dotenv import load_dotenv
from babyagi_task_finalizer.schemas import (
    InputSchema
)
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.utils import get_logger
from naptha_sdk.user import sign_consumer_id
from typing import Dict
from schemas import TaskFinalizer
import json
import asyncio

load_dotenv()
logger = get_logger(__name__)

class TaskFinalizerAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment

        self.user_message_template = """
            You are given the following objective: {{objective}}.
                Your colleagues have accomplished the following tasks with the following results: {{tasks}}.

                <INSTRUCTIONS>
                1. Your task is to study the results of the tasks and prepare a final report.
                2. The final report should be very detailed.
                3. The report should encompass all the tasks that have been performed.
                4. The report should be in MARKDOWN format.
                5. Only prepare the final report if the objective have been met.
                6. If the objective have not been met, prepare the new tasks that need to be performed.
                </INSTRUCTIONS>
                """

    async def generate_tasks(self, inputs: InputSchema) -> str:
        user_prompt = self.user_message_template.replace(
            "{{objective}}",
            inputs["tool_input_data"]["objective"]
        )

        # Prepare context if available
        context = inputs["tool_input_data"]["context"]
        if context:
            user_prompt += f"\nContext: {context}"

        # Prepare messages
        messages = [
            {"role": "system", "content": json.dumps(self.agent_deployment.config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]

        # Prepare LLM configuration
        llm_config = self.agent_deployment.config.llm_config

        def get_openai_structured_schema():
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "User",
                    "schema": TaskFinalizer.model_json_schema()
                }
            }
        schema = get_openai_structured_schema()

        input_ = {
            "messages": messages,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            'response_format': schema
        }

        response = await naptha.node.run_inference(
            input_
        )

        try:
            response_content = response.choices[0].message.content
            return response_content
        
        except Exception as e:
            logger.error(f"Failed to parse response: {response}. Error: {e}")
            return
        

async def run(module_run: Dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run)
    logger.info(f"Running with inputs {module_run.inputs['tool_input_data']}")
    task_initiator_agent = TaskFinalizerAgent(module_run.deployment)
    method = getattr(task_initiator_agent, module_run.inputs['tool_name'], None)
    return await method(module_run.inputs)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()
    
    # Load agent deployments
    print(os.getenv("NODE_URL"))
    deployment = asyncio.run(setup_module_deployment("agent", "babyagi_task_finalizer/configs/deployment.json", node_url = os.getenv("NODE_URL")))
    deployment = AgentDeployment(**deployment.model_dump())
    print("BabyAGI Task Finalizer Deployment:", deployment)

    # Prepare input parameters
    input_params: Dict = {
        "tool_name": "generate_tasks",
        "tool_input_data": {
            "objective": "Write a blog post about the weather in London.",
            "context": "Focus on historical weather patterns between 1900 and 2000"
        }
    }

    # Create agent run input
    agent_run: Dict = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    # Run the agent
    response = asyncio.run(run(agent_run))
    logger.info(f"Final Response: {response}")