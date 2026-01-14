from dataclasses import asdict

from src.pump1d.models import Pump1DResult


def export_result(result: Pump1DResult) -> dict:
    data = asdict(result)
    data["schema_version"] = result.schema_version
    return data
