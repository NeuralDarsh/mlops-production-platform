# MLOps Orchestration & Workflow Engine: Kahn's topological validation, async parallel execution, and failure cascading

import asyncio
import json
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UPSTREAM_FAILED = "UPSTREAM_FAILED"

class PipelineTask:
    """Represents a single executable node within a Directed Acyclic Graph (DAG)."""
    def __init__(
        self,
        task_id: str,
        action: Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]],
        dependencies: Optional[List[str]] = None
    ):
        self.task_id = task_id
        self.action = action
        self.dependencies: Set[str] = set(dependencies) if dependencies else set()
        self.status = TaskStatus.PENDING
        self.output: Any = None
        self.error: Optional[str] = None
        self.duration_ms: float = 0.0

class DAGPipelineOrchestrator:
    """
    Validates and executes a Directed Acyclic Graph (DAG) of MLOps pipeline tasks
    with maximum parallel concurrency and deterministic upstream failure cascading.
    """
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.tasks: Dict[str, PipelineTask] = {}

    def add_task(self, task: PipelineTask):
        if task.task_id in self.tasks:
            raise ValueError(f"Duplicate task_id '{task.task_id}' in DAG.")
        self.tasks[task.task_id] = task

    def _validate_and_build_adjacency(self) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Verifies all dependencies exist and uses Kahn's algorithm to detect cycles.
        Returns: (children_map, in_degree_map)
        """
        children: Dict[str, Set[str]] = {tid: set() for tid in self.tasks}
        in_degree: Dict[str, int] = {tid: 0 for tid in self.tasks}

        for tid, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Task '{tid}' depends on unknown task '{dep_id}'.")
                children[dep_id].add(tid)
                in_degree[tid] += 1

        temp_in_degree = dict(in_degree)
        queue = deque([tid for tid, deg in temp_in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for child in children[curr]:
                temp_in_degree[child] -= 1
                if temp_in_degree[child] == 0:
                    queue.append(child)

        if visited_count != len(self.tasks):
            raise RuntimeError(f"Cycle detected in DAG '{self.pipeline_name}'! Execution aborted.")

        return children, in_degree

    def _cascade_upstream_failure(self, failed_task_id: str, children: Dict[str, Set[str]]):
        """Marks all transitive downstream descendants as UPSTREAM_FAILED."""
        queue = deque(children[failed_task_id])
        while queue:
            curr_id = queue.popleft()
            task = self.tasks[curr_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.UPSTREAM_FAILED
                print(f"[SKIPPED] Task '{curr_id}' marked UPSTREAM_FAILED due to '{failed_task_id}'.")
                queue.extend(children[curr_id])

    async def execute(self) -> Dict[str, Any]:
        """Executes the DAG asynchronously, dispatching independent tasks in parallel."""
        t_start = time.time()
        children, in_degree = self._validate_and_build_adjacency()
        upstream_outputs: Dict[str, Any] = {}

        print(f"[DAG START] Executing pipeline '{self.pipeline_name}' ({len(self.tasks)} tasks)...")

        ready_queue = [tid for tid, deg in in_degree.items() if deg == 0]
        active_futures: Dict[asyncio.Task, str] = {}

        def launch_ready_tasks():
            while ready_queue:
                tid = ready_queue.pop(0)
                task = self.tasks[tid]
                if task.status != TaskStatus.PENDING:
                    continue
                task.status = TaskStatus.RUNNING
                print(f"[DISPATCHED] Running task '{tid}'...")
                coro = self._run_single_task(task, upstream_outputs)
                future = asyncio.create_task(coro)
                active_futures[future] = tid

        launch_ready_tasks()

        while active_futures:
            done, _ = await asyncio.wait(active_futures.keys(), return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                tid = active_futures.pop(future)
                task = self.tasks[tid]
                success = await future

                if success:
                    upstream_outputs[tid] = task.output
                    for child_id in children[tid]:
                        in_degree[child_id] -= 1
                        if in_degree[child_id] == 0 and self.tasks[child_id].status == TaskStatus.PENDING:
                            ready_queue.append(child_id)
                else:
                    self._cascade_upstream_failure(tid, children)

            launch_ready_tasks()

        total_ms = round((time.time() - t_start) * 1000.0, 2)
        summary = {
            "pipeline": self.pipeline_name,
            "total_duration_ms": total_ms,
            "tasks": {
                tid: {
                    "status": t.status.value,
                    "duration_ms": t.duration_ms,
                    "output": t.output,
                    "error": t.error
                }
                for tid, t in self.tasks.items()
            }
        }
        return summary

    async def _run_single_task(self, task: PipelineTask, context: Dict[str, Any]) -> bool:
        t0 = time.time()
        try:
            dep_context = {dep: context.get(dep) for dep in task.dependencies}
            task.output = await task.action(dep_context)
            task.status = TaskStatus.SUCCESS
            task.duration_ms = round((time.time() - t0) * 1000.0, 2)
            print(f"[SUCCESS] Task '{task.task_id}' completed in {task.duration_ms}ms.")
            return True
        except Exception as exc:
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            task.duration_ms = round((time.time() - t0) * 1000.0, 2)
            print(f"[FAILED] Task '{task.task_id}' failed ({exc}) in {task.duration_ms}ms.")
            return False

from typing import Tuple

if __name__ == "__main__":
    print("--- MLOps Orchestration: DAG Workflow & Parallel Scheduler Engine ---\n")

    async def main():
        async def ingest_raw_logs(deps: dict):
            await asyncio.sleep(0.05)
            return {"rows_extracted": 250000, "partition": "2026-11-26"}
        async def build_user_features(deps: dict):
            await asyncio.sleep(0.06)
            rows = deps["ingest_raw_logs"]["rows_extracted"]
            return {"feature_matrix": f"users_{rows}.parquet"}
        async def build_item_embeddings(deps: dict):
            await asyncio.sleep(0.06)
            return {"embedding_table": "items_v4.faiss"}
        async def train_xgboost_ranker(deps: dict):
            await asyncio.sleep(0.08)
            return {"model": "xgb_v2", "auc": 0.914}
        async def train_experimental_gnn(deps: dict):
            await asyncio.sleep(0.04)
            raise RuntimeError("CUDA Out-Of-Memory on experimental GNN branch")
        async def promote_xgb_to_registry(deps: dict):
            await asyncio.sleep(0.03)
            auc = deps["train_xgboost_ranker"]["auc"]
            return {"promoted": True, "stage": "Production", "auc": auc}
        async def deploy_gnn_shadow(deps: dict):
            await asyncio.sleep(0.03)
            return {"shadow_deployed": True}

        dag = DAGPipelineOrchestrator("Daily_RecSys_Retraining_DAG")
        dag.add_task(PipelineTask("ingest_raw_logs", ingest_raw_logs))
        dag.add_task(PipelineTask("build_user_features", build_user_features, dependencies=["ingest_raw_logs"]))
        dag.add_task(PipelineTask("build_item_embeddings", build_item_embeddings, dependencies=["ingest_raw_logs"]))
        dag.add_task(
            PipelineTask(
                "train_xgboost_ranker",
                train_xgboost_ranker,
                dependencies=["build_user_features", "build_item_embeddings"]
            )
        )
        dag.add_task(
            PipelineTask(
                "train_experimental_gnn",
                train_experimental_gnn,
                dependencies=["build_user_features", "build_item_embeddings"]
            )
        )
        dag.add_task(
            PipelineTask("promote_xgb_to_registry", promote_xgb_to_registry, dependencies=["train_xgboost_ranker"])
        )
        dag.add_task(
            PipelineTask("deploy_gnn_shadow", deploy_gnn_shadow, dependencies=["train_experimental_gnn"])
        )

        report = await dag.execute()

        print("\nFinal DAG Execution Telemetry Report:")
        print(json.dumps(report, indent=2))

    asyncio.run(main())