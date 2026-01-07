import salome
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
import GEOM
from salome.geom import geomBuilder
import math
import numpy as np

file_path = "D:/TPDAO/temp/impeller.stp.stp"
read_path = "/temp/"

print("numpy dosyaları okunuyor...")
pressure_np = np.load(read_path + "pressure.npy")
suction_np = np.load(read_path + "suction.npy")
leading_np = np.load(read_path + "leading.npy")
trailing_np = np.load(read_path + "trailing.npy")
hub_np = np.load(read_path + "hub.npy")
tip_np = np.load(read_path + "tip.npy")
lea_np = np.load(read_path + "lea.npy")
tra_np = np.load(read_path + "tra.npy")
print("OKUNDU\n")

geompy = geomBuilder.New()

OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

print("Noktolar oluşturuluyor...")
pressure_vertexes = []
pressure_curves = []
for curve in pressure_np:
    pressure_vertexes.append([])
    for point in curve:
        pressure_vertexes[-1].append(geompy.MakeVertex(point[0], point[1], point[2]))

thickness = np.sqrt(np.sum(np.square(pressure_np[0, -1] - suction_np[0, -1])))
suction_vertexes = []
suction_curves = []
for i, curve in enumerate(suction_np):
    suction_vertexes.append([])
    for point in curve:
        suction_vertexes[-1].append(geompy.MakeVertex(point[0], point[1], point[2]))
    point = trailing_np[i,-1]
    suction_vertexes[-1].append(geompy.MakeVertex(point[0], point[1], point[2]))

leading_vertexes = []
leading_curves = []
for curve in leading_np:
    leading_vertexes.append([])
    for point in curve:
        leading_vertexes[-1].append(geompy.MakeVertex(point[0], point[1], point[2]))

trailing_vertexes = []
trailing_curves = []
for curve in trailing_np:
    trailing_vertexes.append([])
    for point in curve:
        trailing_vertexes[-1].append(geompy.MakeVertex(point[0], point[1], point[2]))

hub_vertexes = []
for point in hub_np:
    hub_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))

tip_vertexes = []
for point in tip_np:
    tip_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))

lea_vertexes = []
for point in lea_np:
    lea_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))

tra_vertexes = []
for point in tra_np:
    tra_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))
print("OLUŞTURULDU\n")
print("Meridyenel eğriler oluşturuluyor...")
hub = geompy.MakeInterpol(hub_vertexes, False, False)
tip = geompy.MakeInterpol(tip_vertexes, False, False)
lea = geompy.MakeInterpol(lea_vertexes, False, False)
tra = geompy.MakeInterpol(tra_vertexes, False, False)
print("OLUŞTURULDU\n")
print("Göbek ve uç yüzeyleri oluşturuluyor...")
hub_surface = geompy.MakeRevolution(hub, OZ, 2 * math.pi)
tip_surface = geompy.MakeRevolution(tip, OZ, 2 * math.pi)
print("OLUŞTURULDU\n")
print("Kanat yüzeyleri oluşturuluyor...")
foil_curves = []
for i, (curve_p, curve_s, curve_l) in enumerate(zip(pressure_vertexes, suction_vertexes, leading_vertexes)):
    foil_curves.append(geompy.MakeInterpol(curve_p[::-1] + curve_l[1:-1] + curve_s, False, False))
foil_surface = geompy.MakeFilling(foil_curves, theMethod=GEOM.FOM_AutoCorrect)


pressure_le = geompy.MakeInterpol([leading_vertexes[i][0] for i in range(len(leading_vertexes))], False,
                                  False)
suction_le = geompy.MakeInterpol([leading_vertexes[i][-1] for i in range(len(leading_vertexes))], False,
                                 False)

pressure_le_projected = geompy.MakeProjection(pressure_le, foil_surface)
suction_le_projected = geompy.MakeProjection(suction_le, foil_surface)

Partition_1 = geompy.MakePartition([foil_surface], [pressure_le_projected, suction_le_projected], [], [],
                                   geompy.ShapeType["FACE"], 0, [], 0)
[suction_offset2, pressure_offset2, leading_edge_offset] = geompy.ExtractShapes(Partition_1, geompy.ShapeType["FACE"],
                                                                                True)

hub_trailing_line = geompy.MakeInterpol([pressure_vertexes[0][-1], suction_vertexes[0][-1]], False, False)
tip_trailing_line = geompy.MakeInterpol([pressure_vertexes[-1][-1], suction_vertexes[-1][-1]], False, False)

suction_offset2_edge_8 = geompy.GetSubShape(suction_offset2, [8])
pressure_offset2_edge_3 = geompy.GetSubShape(pressure_offset2, [3])

trailing_wire = geompy.MakeWire([suction_offset2_edge_8, pressure_offset2_edge_3, hub_trailing_line, tip_trailing_line],
                                1e-07)

trailing = geompy.MakeFaceWires([trailing_wire], 0)

blade = geompy.MakeShell([suction_offset2, pressure_offset2, leading_edge_offset, trailing])
blades = geompy.MultiRotate1DByStep(blade, OZ, math.pi / 3, 6)

impeller = geompy.MakeCompound([blades, hub_surface, tip_surface, hub, tip, lea, tra])
print("OLUŞTURULDU\n")
print(".step dosyası oluşturuluyor...")
geompy.ExportSTEP(impeller, file_path, GEOM.LU_METER)
print("OLUŞTURULDU\n")
