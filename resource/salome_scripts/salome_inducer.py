import salome
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
import GEOM
from salome.geom import geomBuilder
import math
import numpy as np

file_path = "C:/Users/batuhan.koroglu/Desktop/inducer.stp.stp"
read_path = "C:/Users/batuhan.koroglu/Desktop/TPDAO-REF/temp/"

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

print("Noktalar ve Eğriler oluşturuluyor...")

pressure_vertexes = []
pressure_curves = []
for i in range(pressure_np.shape[1]):
    pressure_vertexes.append([])
    for j in range(pressure_np.shape[0]):
        pressure_vertexes[-1].append(geompy.MakeVertex(pressure_np[j, i, 0], pressure_np[j, i, 1], pressure_np[j, i, 2]))
    pressure_curves.append(geompy.MakeInterpol(pressure_vertexes[-1], False, False))

suction_vertexes = []
suction_curves = []
for i in range(suction_np.shape[1]):
    suction_vertexes.append([])
    for j in range(suction_np.shape[0]):
        suction_vertexes[-1].append(geompy.MakeVertex(suction_np[j, i, 0], suction_np[j, i, 1], suction_np[j, i, 2]))
    suction_curves.append(geompy.MakeInterpol(suction_vertexes[-1], False, False))

leading_vertexes = []
leading_curves = []
for i in range(leading_np.shape[1]):
    leading_vertexes.append([])
    for j in range(leading_np.shape[0]):
        leading_vertexes[-1].append(geompy.MakeVertex(leading_np[j, i, 0], leading_np[j, i, 1], leading_np[j, i, 2]))
    leading_curves.append(geompy.MakeInterpol(leading_vertexes[-1], False, False))

trailing_vertexes = []
trailing_curves = []
for i in range(trailing_np.shape[1]):
    trailing_vertexes.append([])
    for j in range(pressure_np.shape[0]):
        trailing_vertexes[-1].append(geompy.MakeVertex(trailing_np[j, i, 0], trailing_np[j, i, 1], trailing_np[j, i, 2]))
    trailing_curves.append(geompy.MakeInterpol(trailing_vertexes[-1], False, False))

hub_vertexes = []
for point in hub_np:
    hub_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))
    # geompy.addToStudy(hub_vertexes[-1], "hub_vertexes")

tip_vertexes = []
for point in tip_np:
    tip_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))
    # geompy.addToStudy(tip_vertexes[-1], "tip_vertexes")

lea_vertexes = []
for point in lea_np:
    lea_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))
    # geompy.addToStudy(lea_vertexes[-1], "lea_vertexes")

tra_vertexes = []
for point in tra_np:
    tra_vertexes.append(geompy.MakeVertex(point[0], point[1], point[2]))
    # geompy.addToStudy(tra_vertexes[-1], "tra_vertexes")

hub = geompy.MakeInterpol(hub_vertexes, False, False)
tip = geompy.MakeInterpol(tip_vertexes, False, False)
lea = geompy.MakeInterpol(lea_vertexes, False, False)
tra = geompy.MakeInterpol(tra_vertexes, False, False)
print("OLUŞTURULDU\n")
print("Göbek ve uç yüzeyleri oluşturuluyor...")
hub_surface = geompy.MakeRevolution(hub, OZ, 2 * math.pi)
tip_surface = geompy.MakeRevolution(tip, OZ, 2 * math.pi)
hub_surface2 = geompy.MakeRotation(hub_surface, OZ, -np.pi/6)
# geompy.addToStudy(hub_surface,"hub_surface")
# geompy.addToStudy(tip_surface,"tip_surface")
print("OLUŞTURULDU\n")
print("Kanat yüzeyleri oluşturuluyor...")

pressure_surface = geompy.MakeFilling(pressure_curves, theMethod=GEOM.FOM_AutoCorrect)
suction_surface = geompy.MakeFilling(suction_curves, theMethod=GEOM.FOM_AutoCorrect)
leading_surface = geompy.MakeFilling(leading_curves, theMethod=GEOM.FOM_AutoCorrect)
trailing_surface = geompy.MakeFilling(trailing_curves, theMethod=GEOM.FOM_AutoCorrect)

blade1 = geompy.MakeShell([pressure_surface, suction_surface, leading_surface, trailing_surface])

Partition_p = geompy.MakePartition([pressure_surface], [tip_surface, hub_surface], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_p_Faces = geompy.ExtractShapes(Partition_p, geompy.ShapeType["FACE"], True)

Partition_s = geompy.MakePartition([suction_surface], [tip_surface, hub_surface], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_s_Faces = geompy.ExtractShapes(Partition_s, geompy.ShapeType["FACE"], True)

Partition_l = geompy.MakePartition([leading_surface], [tip_surface, hub_surface], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_l_Faces = geompy.ExtractShapes(Partition_l, geompy.ShapeType["FACE"], True)

Partition_t = geompy.MakePartition([trailing_surface], [tip_surface, hub_surface], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_t_Faces = geompy.ExtractShapes(Partition_t, geompy.ShapeType["FACE"], True)

blade_faces = []
for faces in [Partition_p_Faces, Partition_s_Faces, Partition_l_Faces, Partition_t_Faces]:
    areas = []
    for face in faces:
        areas.append(geompy.BasicProperties(face)[1])
    blade_faces.append(faces[np.argmax(areas)])

blade2 = geompy.MakeShell(blade_faces)

Partition_2 = geompy.MakePartition([tip_surface], [blade1], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_2_Faces = geompy.ExtractShapes(Partition_2, geompy.ShapeType["FACE"], True)
blade_tip = Partition_2_Faces[np.argmin([geompy.BasicProperties(face)[1] for face in Partition_2_Faces])]

Partition_3 = geompy.MakePartition([hub_surface2], [blade1], [], [], geompy.ShapeType["FACE"], 0, [], 0)
Partition_3_Faces = geompy.ExtractShapes(Partition_3, geompy.ShapeType["FACE"], True)
blade_hub = Partition_3_Faces[np.argmin([geompy.BasicProperties(face)[1] for face in Partition_3_Faces])]

blade = geompy.MakeShell([blade2, blade_hub, blade_tip])
casing_surface = geompy.MakeOffset(tip_surface, 1e-4)
inducer = geompy.MakeCompound([blade, hub_surface, casing_surface, hub, tip, lea, tra])

print("OLUŞTURULDU\n")
print(".step dosyası oluşturuluyor...")
geompy.ExportSTEP(inducer, file_path, GEOM.LU_METER)
print("OLUŞTURULDU\n")
