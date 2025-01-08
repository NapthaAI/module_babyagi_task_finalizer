from pydantic import BaseModel, Field
from typing import List

class TaskFinalizerPromptSchema(BaseModel):
    """Schema for task initiator input data"""
    objective: str
    context: str = Field(default="", description="Optional context for task generation")

class InputSchema(BaseModel):
    """Input schema matching the task executor's structure"""
    tool_name: str = Field(default="generate_tasks", description="Name of the method to call")
    tool_input_data: TaskFinalizerPromptSchema

class Task(BaseModel):
    """Class for defining a task to be performed."""
    name: str = Field(..., description="The name of the task to be performed.")
    description: str = Field(..., description="The description of the task to be performed.")
    done: bool = Field(False, description="The status of the task. True if the task is done, False otherwise.")
    result: str = Field("", description="The result of the task.")

class TaskFinalizer(BaseModel):
    """Class for finalizing the tasks."""
    final_report: str = Field("", description="The final report of the tasks.")
    new_tasks: List[Task] = Field([], description="A list of new tasks to be performed.")
    objective_met: bool = Field(False, description="The status of the objective. True if the objective have been met, False otherwise.")