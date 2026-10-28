# MLOps & Data Quality Engineering: Declarative schema contracts, type coercion, and Dead-Letter Queue (DLQ) routing

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Type

class FieldConstraint:
    """Defines type, presence, and boundary rules for a single payload field."""
    def __init__(
        self,
        expected_type: Type,
        required: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[List[Any]] = None
    ):
        self.expected_type = expected_type
        self.required = required
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values


class DataContract:
    """Declarative schema contract representing expected data structure and boundaries."""
    def __init__(self, name: str, schema: Dict[str, FieldConstraint]):
        self.name = name
        self.schema = schema


class DataContractValidator:
    """
    Enforces data contracts on incoming feature payloads, quarantining
    malformed records into a Dead-Letter Queue (DLQ).
    """
    def __init__(self, contract: DataContract, allow_coercion: bool = True):
        self.contract = contract
        self.allow_coercion = allow_coercion
        self.dead_letter_queue: List[Dict[str, Any]] = []
        self.valid_records: List[Dict[str, Any]] = []

    def validate_and_route(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validates record against contract.
        If valid: returns (True, coerced_record) and buffers in valid_records.
        If invalid: returns (False, None) and routes record with violations to DLQ.
        """
        violations = []
        sanitized = dict(record)

        # Check all constraints defined in contract
        for field_name, constraint in self.contract.schema.items():
            if field_name not in sanitized or sanitized[field_name] is None:
                if constraint.required:
                    violations.append(f"Missing required field: '{field_name}'")
                continue

            val = sanitized[field_name]

            # Type checking and optional coercion
            if not isinstance(val, constraint.expected_type):
                if self.allow_coercion:
                    try:
                        val = constraint.expected_type(val)
                        sanitized[field_name] = val
                    except (ValueError, TypeError):
                        violations.append(
                            f"Field '{field_name}' has invalid type {type(val).__name__}, "
                            f"expected {constraint.expected_type.__name__} (coercion failed)"
                        )
                        continue
                else:
                    violations.append(
                        f"Field '{field_name}' has invalid type {type(val).__name__}, "
                        f"expected {constraint.expected_type.__name__}"
                    )
                    continue

            # Numerical range validation
            if constraint.min_value is not None and val < constraint.min_value:
                violations.append(f"Field '{field_name}' value {val} is below minimum {constraint.min_value}")
            if constraint.max_value is not None and val > constraint.max_value:
                violations.append(f"Field '{field_name}' value {val} exceeds maximum {constraint.max_value}")

            # Categorical enum membership validation
            if constraint.allowed_values is not None and val not in constraint.allowed_values:
                violations.append(f"Field '{field_name}' value '{val}' not in allowed set {constraint.allowed_values}")

        if violations:
            dlq_entry = {
                "raw_payload": record,
                "violations": violations,
                "quarantined_at": time.time(),
                "contract_name": self.contract.name
            }
            self.dead_letter_queue.append(dlq_entry)
            print(f"[QUARANTINED TO DLQ] Violations: {violations}")
            return False, None

        self.valid_records.append(sanitized)
        print(f"[VALIDATED] Payload passed contract '{self.contract.name}'")
        return True, sanitized


if __name__ == "__main__":
    print("--- MLOps Ingestion: Data Contract & Schema Enforcement Engine ---\n")

    # Define contract for transaction fraud feature ingestion
    fraud_features_contract = DataContract(
        name="TransactionFraudIngestContract_v1",
        schema={
            "transaction_id": FieldConstraint(expected_type=str, required=True),
            "amount": FieldConstraint(expected_type=float, required=True, min_value=0.01, max_value=50000.0),
            "user_age": FieldConstraint(expected_type=int, required=True, min_value=18, max_value=120),
            "payment_type": FieldConstraint(expected_type=str, required=True, allowed_values=["credit_card", "upi", "bank_transfer"]),
            "is_risk_flagged": FieldConstraint(expected_type=bool, required=False)
        }
    )

    validator = DataContractValidator(contract=fraud_features_contract, allow_coercion=True)

    print("Test 1: Ingesting valid transaction record:")
    sample_valid = {
        "transaction_id": "tx_tokyo_101",
        "amount": 2500.50,
        "user_age": 28,
        "payment_type": "credit_card",
        "is_risk_flagged": False
    }
    validator.validate_and_route(sample_valid)

    print("\nTest 2: Ingesting record requiring type coercion ('amount' as string):")
    sample_coercible = {
        "transaction_id": "tx_tokyo_102",
        "amount": "120.75",  # String coercible to float
        "user_age": 45,
        "payment_type": "upi"
    }
    validator.validate_and_route(sample_coercible)

    print("\nTest 3: Ingesting corrupt payload violating multiple constraints:")
    sample_corrupt = {
        "transaction_id": "tx_tokyo_103",
        "amount": -50.0,      
        "user_age": 12,              
        "payment_type": "crypto"     
    }
    validator.validate_and_route(sample_corrupt)

    print("\nDead-Letter Queue (DLQ) Audit Dump:")
    print(json.dumps(validator.dead_letter_queue, indent=2))