import pytest

from processor.import_processor.main_graph import ImportWorkflow
from processor.query_processor.main_graph import KBQueryWorkflow


@pytest.mark.parametrize(
    ("workflow", "prefix"),
    [
        (ImportWorkflow(), "import-"),
        (KBQueryWorkflow(), "query-"),
    ],
)
def test_workflow_generates_task_id_when_missing(workflow, prefix):
    state = {}

    task_id = workflow._ensure_task_id(state)

    assert task_id.startswith(prefix)
    assert state["task_id"] == task_id


@pytest.mark.parametrize("workflow", [ImportWorkflow(), KBQueryWorkflow()])
def test_workflow_preserves_caller_task_id(workflow):
    state = {"task_id": "  external-task-001  "}

    task_id = workflow._ensure_task_id(state)

    assert task_id == "external-task-001"
    assert state["task_id"] == "external-task-001"


@pytest.mark.parametrize("workflow", [ImportWorkflow(), KBQueryWorkflow()])
def test_workflow_rejects_non_string_task_id(workflow):
    with pytest.raises(ValueError, match="task_id 必须是字符串"):
        workflow._ensure_task_id({"task_id": 123})
