# MLOps & Model Governance: Semantic versioning, stage state machines, artifact checksum verification, and instant rollback

import hashlib
import json
import time
from typing import Dict, List, Optional

class ModelStage:
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


class ModelArtifactManifest:
    """Represents an immutable versioned model artifact with training metadata and lineage."""
    def __init__(
        self,
        model_name: str,
        version: str,
        weights_payload: str,
        metrics: Dict[str, float],
        lineage_dataset_id: str
    ):
        self.model_name = model_name
        self.version = version
        self.weights_payload = weights_payload
        self.checksum = hashlib.sha256(weights_payload.encode("utf-8")).hexdigest()
        self.metrics = metrics
        self.lineage_dataset_id = lineage_dataset_id
        self.stage = ModelStage.DEVELOPMENT
        self.registered_at = time.time()

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage,
            "checksum": self.checksum[:12],
            "metrics": self.metrics,
            "lineage_dataset_id": self.lineage_dataset_id,
            "registered_at": round(self.registered_at, 2)
        }


class ModelRegistry:
    """
    Thread-safe model registry governing artifact versioning,
    lifecycle state transitions, and automated rollback points.
    """
    def __init__(self):
        # Maps model_name -> list of ModelArtifactManifest
        self._models: Dict[str, List[ModelArtifactManifest]] = {}
        # Tracks current active production version: model_name -> version
        self._active_production: Dict[str, str] = {}
        # Rollback stack: model_name -> list of previously active production versions
        self._rollback_history: Dict[str, List[str]] = {}

    def register_model(
        self,
        model_name: str,
        version: str,
        weights_payload: str,
        metrics: Dict[str, float],
        lineage_dataset_id: str
    ) -> ModelArtifactManifest:
        """Registers a new immutable model artifact in DEVELOPMENT stage."""
        if model_name not in self._models:
            self._models[model_name] = []
            self._rollback_history[model_name] = []

        # Prevent duplicate versions
        for existing in self._models[model_name]:
            if existing.version == version:
                raise ValueError(f"Version '{version}' for model '{model_name}' already exists.")

        manifest = ModelArtifactManifest(
            model_name=model_name,
            version=version,
            weights_payload=weights_payload,
            metrics=metrics,
            lineage_dataset_id=lineage_dataset_id
        )
        self._models[model_name].append(manifest)
        print(f"[REGISTERED] Model: '{model_name}' | Version: {version} | Checksum: {manifest.checksum[:8]}")
        return manifest

    def transition_stage(self, model_name: str, version: str, target_stage: str) -> bool:
        """Transitions a model to a new lifecycle stage with mutual exclusivity for PRODUCTION."""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found.")

        target_artifact: Optional[ModelArtifactManifest] = None
        for artifact in self._models[model_name]:
            if artifact.version == version:
                target_artifact = artifact
                break

        if not target_artifact:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'.")

        # Promote to PRODUCTION logic
        if target_stage == ModelStage.PRODUCTION:
            current_prod_version = self._active_production.get(model_name)
            if current_prod_version and current_prod_version != version:
                # Demote existing production model and push to rollback stack
                for art in self._models[model_name]:
                    if art.version == current_prod_version:
                        art.stage = ModelStage.ARCHIVED
                        self._rollback_history[model_name].append(current_prod_version)
                        print(f" [AUTO-ARCHIVED] Previous production version '{current_prod_version}' demoted.")

            self._active_production[model_name] = version

        target_artifact.stage = target_stage
        print(f"[STAGE TRANSITION] Model '{model_name}:{version}' -> {target_stage}")
        return True

    def rollback_production(self, model_name: str) -> Optional[str]:
        """Rolls back PRODUCTION stage to the immediate prior certified version."""
        history = self._rollback_history.get(model_name, [])
        if not history:
            print(f"  [ROLLBACK FAILED] No prior production version recorded for '{model_name}'.")
            return None

        previous_version = history.pop()
        current_failed_version = self._active_production.get(model_name)

        # Demote failed version
        if current_failed_version:
            for art in self._models[model_name]:
                if art.version == current_failed_version:
                    art.stage = ModelStage.ARCHIVED

        # Re-promote previous stable version
        for art in self._models[model_name]:
            if art.version == previous_version:
                art.stage = ModelStage.PRODUCTION
                self._active_production[model_name] = previous_version
                print(f"[ROLLBACK EXECUTED] Reverted '{model_name}' from '{current_failed_version}' to '{previous_version}'.")
                return previous_version

        return None

    def get_production_model(self, model_name: str) -> Optional[ModelArtifactManifest]:
        """Resolves the currently active production artifact."""
        active_version = self._active_production.get(model_name)
        if not active_version:
            return None
        for art in self._models[model_name]:
            if art.version == active_version:
                return art
        return None


if __name__ == "__main__":
    print("--- MLOps Governance: Automated Model Registry & Rollback Engine ---\n")

    registry = ModelRegistry()
    model = "fraud_detection_xgboost"

    # 1. Register baseline model v1.0.0
    print("Step 1: Registering initial baseline artifact:")
    registry.register_model(
        model_name=model,
        version="v1.0.0",
        weights_payload="weights_blob_matrix_v1",
        metrics={"auc_roc": 0.884, "latency_p99_ms": 12.0},
        lineage_dataset_id="dataset_train_2026_q1"
    )

    # Promote v1.0.0 to PRODUCTION
    registry.transition_stage(model, "v1.0.0", ModelStage.PRODUCTION)

    # 2. Register improved model v1.1.0
    print("\nStep 2: Registering and promoting new candidate artifact:")
    registry.register_model(
        model_name=model,
        version="v1.1.0",
        weights_payload="weights_blob_matrix_v1_1",
        metrics={"auc_roc": 0.912, "latency_p99_ms": 11.5},
        lineage_dataset_id="dataset_train_2026_q2"
    )

    registry.transition_stage(model, "v1.1.0", ModelStage.STAGING)
    registry.transition_stage(model, "v1.1.0", ModelStage.PRODUCTION)

    # 3. Verify active production model
    current_prod = registry.get_production_model(model)
    print(f"\nStep 3: Active Production Model: {current_prod.version} (AUC: {current_prod.metrics['auc_roc']})")

    # 4. Trigger automated emergency rollback (simulating regression in v1.1.0)
    print("\nStep 4: Simulating production performance degradation -> Triggering Rollback:")
    registry.rollback_production(model)

    restored_prod = registry.get_production_model(model)
    print(f" Verified Restored Production Model: {restored_prod.version}")

    # 5. Output full registry metadata manifest
    print("\nFull Model Registry Audit Log:")
    audit_data = [art.to_dict() for art in registry._models[model]]
    print(json.dumps(audit_data, indent=2))