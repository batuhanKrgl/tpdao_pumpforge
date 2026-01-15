from dataclasses import dataclass


@dataclass
class Station:
    name: str
    total_pressure_Pa: float
    static_pressure_Pa: float
    mass_flow_kg_s: float

    def as_dict(self) -> dict:
        return {
            "total_pressure_Pa": self.total_pressure_Pa,
            "static_pressure_Pa": self.static_pressure_Pa,
            "mass_flow_kg_s": self.mass_flow_kg_s,
        }
