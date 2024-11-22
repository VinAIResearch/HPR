from .task import *


TASK_MAPPING = {
    "truthful_qa.generation": TruthfulQAGenerationTask,
    "truthful_qa.multiple_choice.mc1": TruthfulQAMultipleChoice1Task,
    "truthful_qa.multiple_choice.mc2": TruthfulQAMultipleChoice2Task,
    "truthful_qa.multiple_choice": TruthfulQAMultipleChoice1Task,   # Defaults to MC1 for now
    "toxigen/toxigen-datasets.annotated": ToxigenTask,
    "tasksource/bigbench.simple_ethical_questions": BigBench,
    "tasksource/bigbench.bbq_lite_json": BigBench,
}
