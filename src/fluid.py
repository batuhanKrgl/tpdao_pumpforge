import json


class Fluid:
    def __init__(self, file=None):
        self.name = ""
        self.density = 0
        self.dynamicViscosity = 0
        self.vaporPressure = 0
        self.kinematicViscosity = 0
        self.type = ""
        self.gamma = 0
        self.cp = 0
        self.r = 0
        self.k = 0

        if file is not None:
            self.load_pump_properties(file)

    def load_pump_properties(self, fluid_file):
        with open(fluid_file, 'r') as file:
            data = json.load(file)

        self.name = data.get("name", "")
        self.type = data.get("type", "")
        self.density = data.get("density", 0)
        self.dynamicViscosity = data.get("dynamicViscosity", 0)
        self.vaporPressure = data.get("vaporPressure", 0)
        self.kinematicViscosity = self.dynamicViscosity / self.density

    def load_turbine_properties(self, fluid_file):
        with open(fluid_file, 'r') as file:
            data = json.load(file)

        self.gamma = data.get("gamma", 0)
        self.cp = data.get("cp", 0)
        self.r = data.get("r", 0)

    def update_properties(self, property_name, value):
        if property_name == "name":
            self.name = value
        elif property_name == "type":
            self.type = value
        elif property_name == "density":
            self.density = float(value)
        elif property_name == "vaporPressure":
            self.vaporPressure = float(value)
        elif property_name == "dynamicViscosity":
            self.dynamicViscosity = float(value)
        elif property_name == "gamma":
            self.gamma = float(value)
        elif property_name == "cp":
            self.cp = float(value)
        elif property_name == "r":
            self.r = float(value)
        else:
            print("abc")
        # elif property_name == "thermalConductivity":
        #     self.k = float(value)
