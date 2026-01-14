from src.pump1d.export import export_result
from src.pump1d.models import Pump1DInput
from src.pump1d.pump_solver import PumpSolver


def run_pump1d(input_data: dict) -> dict:
    pump_input = Pump1DInput.from_dict(input_data)
    solver = PumpSolver(pump_input)
    result = solver.solve()
    return export_result(result)
