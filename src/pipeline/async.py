"""Async pipeline for concurrent execution of pipeline stages."""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a pipeline stage."""

    stage_name: str
    success: bool
    data: Any
    error: Optional[str]
    duration: float
    timestamp: str


@dataclass
class PipelineStage:
    """A stage in the pipeline."""

    name: str
    func: Callable
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class AsyncPipeline:
    """Async pipeline for concurrent stage execution."""

    def __init__(self):
        """Initialize async pipeline."""
        self.stages: Dict[str, PipelineStage] = {}
        self.results: Dict[str, PipelineResult] = {}
        self.context: Dict = {}

    def add_stage(
        self,
        name: str,
        func: Callable,
        dependencies: List[str] = None,
    ):
        """Add a pipeline stage.

        Args:
            name: Stage name
            func: Async function to execute
            dependencies: List of stage names this stage depends on
        """
        self.stages[name] = PipelineStage(
            name=name,
            func=func,
            dependencies=dependencies,
        )

    async def run_stage(self, stage: PipelineStage) -> PipelineResult:
        """Run a single stage.

        Args:
            stage: Stage to run

        Returns:
            PipelineResult
        """
        # Check dependencies
        for dep in stage.dependencies:
            if dep not in self.results:
                raise ValueError(f"Dependency {dep} not found")
            if not self.results[dep].success:
                return PipelineResult(
                    stage_name=stage.name,
                    success=False,
                    data=None,
                    error=f"Dependency {dep} failed",
                    duration=0.0,
                    timestamp=datetime.now().isoformat(),
                )

        start_time = datetime.now()

        try:
            logger.info(f"Running stage: {stage.name}")

            # Run stage function
            data = await stage.func(self.context)

            duration = (datetime.now() - start_time).total_seconds()

            result = PipelineResult(
                stage_name=stage.name,
                success=True,
                data=data,
                error=None,
                duration=duration,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(f"Stage {stage.name} completed in {duration:.2f}s")

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            logger.error(f"Stage {stage.name} failed: {e}")

            return PipelineResult(
                stage_name=stage.name,
                success=False,
                data=None,
                error=str(e),
                duration=duration,
                timestamp=datetime.now().isoformat(),
            )

    async def run(
        self,
        context: Dict = None,
        parallel: bool = True,
    ) -> Dict[str, PipelineResult]:
        """Run the pipeline.

        Args:
            context: Pipeline context (shared data)
            parallel: Run independent stages in parallel

        Returns:
            Dict of stage results
        """
        self.context = context or {}
        self.results = {}

        if parallel:
            # Run stages in parallel where possible
            pending = set(self.stages.keys())
            completed = set()

            while pending:
                # Find stages whose dependencies are complete
                ready = []
                for stage_name in pending:
                    stage = self.stages[stage_name]
                    if all(dep in completed for dep in stage.dependencies):
                        ready.append(stage_name)

                if not ready:
                    logger.error("Circular dependency detected")
                    break

                # Run ready stages in parallel
                tasks = [
                    self.run_stage(self.stages[name])
                    for name in ready
                ]

                results = await asyncio.gather(*tasks)

                for result in results:
                    self.results[result.stage_name] = result
                    completed.add(result.stage_name)
                    pending.discard(result.stage_name)

        else:
            # Run stages sequentially
            for stage_name in self.stages:
                stage = self.stages[stage_name]

                # Check dependencies
                for dep in stage.dependencies:
                    if dep not in self.results or not self.results[dep].success:
                        logger.error(f"Stage {stage_name} skipped (dependency {dep} failed)")
                        self.results[stage_name] = PipelineResult(
                            stage_name=stage_name,
                            success=False,
                            data=None,
                            error=f"Dependency {dep} failed",
                            duration=0.0,
                            timestamp=datetime.now().isoformat(),
                        )
                        continue

                result = await self.run_stage(stage)
                self.results[stage_name] = result

        return self.results

    def get_summary(self) -> Dict:
        """Get pipeline execution summary.

        Returns:
            Summary dict
        """
        total_duration = sum(r.duration for r in self.results.values())
        successful = sum(1 for r in self.results.values() if r.success)

        return {
            "total_stages": len(self.stages),
            "successful": successful,
            "failed": len(self.stages) - successful,
            "total_duration": total_duration,
            "stages": {
                name: {
                    "success": result.success,
                    "duration": result.duration,
                    "error": result.error,
                }
                for name, result in self.results.items()
            },
        }


# Predefined pipeline stages
async def stage_download_model(context: Dict) -> Dict:
    """Download base model stage.

    Args:
        context: Pipeline context

    Returns:
        Stage result
    """
    # Placeholder: would download actual model
    await asyncio.sleep(1)  # Simulate download
    return {"model_path": "artifacts/base/model"}


async def stage_build_dataset(context: Dict) -> Dict:
    """Build dataset stage.

    Args:
        context: Pipeline context

    Returns:
        Stage result
    """
    # Placeholder: would build actual dataset
    await asyncio.sleep(2)
    return {"dataset_path": "data/train/dataset"}


async def stage_train_model(context: Dict) -> Dict:
    """Train model stage.

    Args:
        context: Pipeline context

    Returns:
        Stage result
    """
    # Placeholder: would train actual model
    await asyncio.sleep(5)
    return {"checkpoint_path": "artifacts/checkpoints/ckpt1"}


async def stage_evaluate_model(context: Dict) -> Dict:
    """Evaluate model stage.

    Args:
        context: Pipeline context

    Returns:
        Stage result
    """
    # Placeholder: would evaluate actual model
    await asyncio.sleep(3)
    return {"pass_at_1": 0.42}


async def stage_quantize_model(context: Dict) -> Dict:
    """Quantize model stage.

    Args:
        context: Pipeline context

    Returns:
        Stage result
    """
    # Placeholder: would quantize actual model
    await asyncio.sleep(2)
    return {"quantized_path": "artifacts/quantized/model"}


def create_default_pipeline() -> AsyncPipeline:
    """Create default LeanCoder pipeline.

    Returns:
        Configured AsyncPipeline
    """
    pipeline = AsyncPipeline()

    pipeline.add_stage("download", stage_download_model, [])
    pipeline.add_stage("build_dataset", stage_build_dataset, [])
    pipeline.add_stage("train", stage_train_model, ["download", "build_dataset"])
    pipeline.add_stage("evaluate", stage_evaluate_model, ["train"])
    pipeline.add_stage("quantize", stage_quantize_model, ["train"])

    return pipeline


async def run_full_pipeline(context: Dict = None) -> Dict:
    """Run the full LeanCoder pipeline asynchronously.

    Args:
        context: Pipeline context

    Returns:
        Pipeline results
    """
    pipeline = create_default_pipeline()

    print("\n" + "=" * 60)
    print("Running LeanCoder Pipeline (Async)")
    print("=" * 60 + "\n")

    results = await pipeline.run(context or {}, parallel=True)

    summary = pipeline.get_summary()

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Successful: {summary['successful']}/{summary['total_stages']}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print("=" * 60 + "\n")

    return results
