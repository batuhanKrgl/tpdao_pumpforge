#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.11.0 with dump python functionality
###
import os
import json
import sys
salome_installation_path = "C:/Users/anil.kucuk/Desktop/SALOME-9.11.0"
sys.path.append(salome_installation_path)

#script_folder = os.getcwd()
#script_path = os.path.join(script_folder, "drawBlade.py")
design_tool_output_path = 'C:/Users/anil.kucuk/Desktop/V2_TurbineDesignTool/outputs'
import numpy as np
import salome
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()


###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


dosya_yolu = f"{design_tool_output_path}/turb_specifications.json"
with open(dosya_yolu, "r") as dosya:
    TURB_specifications = json.load(dosya)
##Parameters
Dhub                        = TURB_specifications["Rotor"]["hub_diameter"] * 1000 # Hub diameter of rotor(mm)
D_blade_tip                 = TURB_specifications["Rotor"]["blade_tip_diameter"] * 1000 # Rotor blade tip diamater(mm)
D_case_tip                  = TURB_specifications["Rotor"]["case_tip_diameter"] *1000 # Case tip diameter of rotor(mm)       
Dmid                        = TURB_specifications["Rotor"]["mean_diameter"] * 1000 # Mean diameter of rotor(mm)
blade_height                = TURB_specifications["Rotor"]["blade_height"]   *1000 # Blade height(mm)
dummy_hb                    = blade_height * 1.2 #Dummy blade extrusion length(It needs to be higher than 1.1 because the blades will be cut by shapes)
rotor_span                  = 15  # Distance between rotor inlet and outlet(mm)
blade_nozzle_exit_distance  = 1   # Distance between rotor inlet and blade leading edge(mm)
number_of_blades            = 103 # Number of rotor blades

ent_len                     = 0.005 # Nozzle entrance length(mm)
nozzle_translation_len      = 20    # Nozzle axial translation distance(mm)
nozzle_spouting_angle       = 67    # Nozzle spouting angle(deg)
nozzle_number               = TURB_specifications["Nozzle"]["number_of_nozzles"]    # Number of nozzles
deg_between_two_nozzle      = 25    # Degree between two of nozzles(deg)
geompy = geomBuilder.New()


O                               = geompy.MakeVertex(0, 0, 0)
OX                              = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY                              = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ                              = geompy.MakeVectorDXDYDZ(0, 0, 1)

#coor_lower_blade_transition     = np.load('C:/Users/anil.kucuk/Desktop/TurbineDesignTool/outputs/lower_blade_transition.npy')
coor_lower_blade_transition     = np.load(f"{design_tool_output_path}/lower_blade_transition.npy")
points_lower_blade_transition   = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_lower_blade_transition]
interpol_lower_transition       = geompy.MakeInterpol(points_lower_blade_transition)

#coor_lower_blade_circular       = np.load('C:/Users/anil.kucuk/Desktop/TurbineDesignTool/outputs/lower_blade_circular.npy')
coor_lower_blade_circular       = np.load(f"{design_tool_output_path}/lower_blade_circular.npy")
points_lower_blade_circular     = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_lower_blade_circular]
interpol_lower_circular         = geompy.MakeInterpol(points_lower_blade_circular)

coor_lower_blade_sym_transition = np.load(f"{design_tool_output_path}/lower_blade_sym_transition.npy")
points_low_blade_sym_transition = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_lower_blade_sym_transition]
interpol_lower_sym_transition   = geompy.MakeInterpol(points_low_blade_sym_transition)

coor_upper_blade_straight       = np.load(f"{design_tool_output_path}/upper_blade_straight.npy")
points_upper_blade_straight     = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_upper_blade_straight]
interpol_upper_straight         = geompy.MakeInterpol(points_upper_blade_straight)

coor_upper_blade_transition     = np.load(f"{design_tool_output_path}/upper_blade_transition.npy")
points_upper_blade_transition   = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_upper_blade_transition]
interpol_upper_transition       = geompy.MakeInterpol(points_upper_blade_transition)

coor_upper_blade_circular       = np.load(f"{design_tool_output_path}/upper_blade_circular.npy")
points_upper_blade_circular     = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_upper_blade_circular]
interpol_upper_circular         = geompy.MakeInterpol(points_upper_blade_circular)

coor_upper_blade_sym_transition = np.load(f"{design_tool_output_path}/upper_blade_sym_transition.npy")
points_up_blade_sym_transition  = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_upper_blade_sym_transition]
interpol_upper_sym_transition   = geompy.MakeInterpol(points_up_blade_sym_transition)

coor_upper_blade_sym_straight   = np.load(f"{design_tool_output_path}/upper_blade_sym_straight.npy")
points_upper_blade_sym_straight = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in coor_upper_blade_sym_straight]
interpol_upper_sym_straight     = geompy.MakeInterpol(points_upper_blade_sym_straight)

leading_edge_coordinates        = np.load(f"{design_tool_output_path}/blade_leading_edge.npy")
points_leading_edge             = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in leading_edge_coordinates]
#polyline_leading_edge           = geompy.MakePolyline(points_leading_edge)
interpol_leading_edge           = geompy.MakeInterpol(points_leading_edge,False,False)

trailing_edge_coordinates       = np.load(f"{design_tool_output_path}/blade_trailing_edge.npy")
points_trailing_edge            = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in trailing_edge_coordinates]
#polyline_trailing_edge          = geompy.MakePolyline(points_trailing_edge)
interpol_trailing_edge          = geompy.MakeInterpol(points_trailing_edge,False,False)

nozzle_surface_coordinates      = np.load(f"{design_tool_output_path}/nozzle_surface_coordinates.npy")
points_nozzle                   = [geompy.MakeVertex(coord[0], coord[1], coord[2]) for coord in nozzle_surface_coordinates] 
polyline_nozzle                 = geompy.MakePolyline(points_nozzle)
#polyline_nozzle                 = geompy.MakeInterpol(points_nozzle,False,False)
hub_circle                      = geompy.MakeCircle(None, None, Dhub / 2)
tip_circle                      = geompy.MakeCircle(None, None, D_case_tip / 2)
blade_tip_circle_center         = geompy.MakeVertex(0, 0, -5)
blade_tip_circle                = geompy.MakeCircle(blade_tip_circle_center, None, D_blade_tip / 2)
dummy_circle_to_cut             = geompy.MakeCircle(None,None, (D_blade_tip / 2) + dummy_hb)

hub_circle_face                 = geompy.MakeFaceWires([hub_circle], 1)
tip_circle_face                 = geompy.MakeFaceWires([tip_circle], 1)
blade_tip_face                  = geompy.MakeFaceWires([blade_tip_circle],1)
dummy_circle_to_cut_face        = geompy.MakeFaceWires([dummy_circle_to_cut],1)

first_point_nozzle              = nozzle_surface_coordinates[0]
last_point_nozzle               = nozzle_surface_coordinates[-1]
nozzle_entry_vertex             = geompy.MakeVertex(first_point_nozzle[0],first_point_nozzle[1],first_point_nozzle[2] - ent_len)
nozzle_inlet_vertex             = geompy.MakeVertex(0, 0, first_point_nozzle[2] - ent_len)
nozzle_outlet_vertex            = geompy.MakeVertex(0, 0, last_point_nozzle[2])

nozzle_entry_line               = geompy.MakeLineTwoPnt(points_nozzle[0], nozzle_entry_vertex)
nozzle_inlet_line               = geompy.MakeLineTwoPnt(nozzle_entry_vertex, nozzle_inlet_vertex)
nozzle_outlet_line              = geompy.MakeLineTwoPnt(points_nozzle[-1], nozzle_outlet_vertex)
nozzle_sym_line                 = geompy.MakeLineTwoPnt(nozzle_outlet_vertex, nozzle_inlet_vertex)
nozzle_surface                  = geompy.MakeFaceWires([nozzle_entry_line, nozzle_inlet_line, nozzle_outlet_line, nozzle_sym_line,polyline_nozzle],1)
nozzle_surface_scale            = geompy.MakeScaleTransform(nozzle_surface,None,1000)
nozzle_revolution               = geompy.MakeRevolution(nozzle_surface_scale, OZ, 360*math.pi/180.0)
[nozzle_outlet_face]            = geompy.SubShapes(nozzle_revolution, [12])
nozzle_outlet_extrusion         = geompy.MakePrismVecH(nozzle_outlet_face, OZ, 50)
nozzle_fuse                     = geompy.MakeFuseList([nozzle_revolution, nozzle_outlet_extrusion], True, True)
nozzle_axial_translation        = geompy.MakeTranslation(nozzle_fuse, 0, 0, -1 * nozzle_translation_len)
nozzle_rotation                 = geompy.MakeRotation(nozzle_axial_translation, OY, nozzle_spouting_angle*math.pi/180.0)
nozzle_radial_translation       = geompy.MakeTranslation(nozzle_rotation, 0, Dmid / 2, 0)

nozzle_cut_plane_extrusion      = geompy.MakePrismVecH(tip_circle_face, OZ, 50)
nozzle_rotor_inlet_plane_cut    = geompy.MakeCutList(nozzle_radial_translation, [nozzle_cut_plane_extrusion], True)
nozzle_circular_pattern         = geompy.MultiRotate1DByStep(nozzle_rotor_inlet_plane_cut, OZ, deg_between_two_nozzle*math.pi/180.0, nozzle_number)

blade_upper_line                = geompy.MakeWire([interpol_upper_straight, interpol_upper_transition, interpol_upper_circular, interpol_upper_sym_transition, interpol_upper_sym_straight], 1e-07)
blade_lower_line                = geompy.MakeWire([interpol_lower_transition, interpol_lower_circular, interpol_lower_sym_transition])
blade_surface                   = geompy.MakeFaceWires([blade_upper_line, blade_lower_line,interpol_leading_edge, interpol_trailing_edge], 1)
blade_scale                     = geompy.MakeScaleTransform(blade_surface, None, 10)
blade_radial_translation        = geompy.MakeTranslation(blade_scale, Dmid / 2, 0, 0)
blade_extrusion                 = geompy.MakePrismDXDYDZ2Ways(blade_radial_translation, dummy_hb, 0, 0)
blade_axial_translation         = geompy.MakeTranslation(blade_extrusion,0, 0, blade_nozzle_exit_distance)

hub_face_extrusion              = geompy.MakePrismDXDYDZ2Ways(hub_circle_face, 0, 0, 1 * rotor_span)
tip_face_extrusion              = geompy.MakePrismDXDYDZ(tip_circle_face, 0, 0, 1 * rotor_span)
blade_tip_face_extrusion_2      = geompy.MakePrismDXDYDZ2Ways(blade_tip_face, 0, 0, 3 * rotor_span)
blade_tip_face_extrusion        = geompy.MakePrismDXDYDZ(blade_tip_face, 0, 0, 3 * rotor_span)
dummy_face_extrusion            = geompy.MakePrismDXDYDZ2Ways(dummy_circle_to_cut_face, 0, 0, 1*rotor_span)


rotor_hub                       = geompy.MakePrismDXDYDZ(hub_circle_face, 0, 0, rotor_span)
rotor_tip                       = geompy.MakePrismDXDYDZ(tip_circle,0 , 0, rotor_span)
dummy_face_cut                  = geompy.MakeCutList(dummy_face_extrusion, [blade_tip_face_extrusion_2],True)
rotor_cut                       = geompy.MakeCutList(rotor_tip, [rotor_hub], True)
blade_cut                       = geompy.MakeCutList(blade_axial_translation, [hub_face_extrusion], True)
blade_final_cut                 = geompy.MakeCutList(blade_cut, [dummy_face_cut],True)
#blade_final_cut                 = geompy.MakeCutList(blade_extrusion, [blade_upper_portion_cut], True)
#blade_axial_translation         = geompy.MakeTranslation(blade_final_cut,0, 0, blade_nozzle_exit_distance)

#rotation_vector                 = geompy.MakeVectorDXDYDZ(0, 0, 1)
#blade_pattern                   = geompy.MultiRotate1DNbTimes(blade_axial_translation, rotation_vector, number_of_blades)

#rotor_fluid_volume              = geompy.MakeCutList(rotor_cut,[blade_pattern], True)
#turbine_fluid_volume            = geompy.MakeCompound([nozzle_circular_pattern, rotor_fluid_volume])
#geompy.ExportSTEP(turbine_fluid_volume, "C:/Users/anil.kucuk/Desktop/TurbineDesignTool/outputs/Geometry/turbine_fluid_volume.step", GEOM.LU_METER )

geompy.addToStudy(nozzle_entry_vertex, 'Nozzle Entry Vertex')
geompy.addToStudy(nozzle_inlet_vertex, 'Nozzle Inlet Vertex')
geompy.addToStudy(nozzle_outlet_vertex, 'Nozzle Outlet Vertex')
geompy.addToStudy(nozzle_entry_line, 'Nozzle Entry Line')
geompy.addToStudy(nozzle_inlet_line, 'Nozzle Inlet Line')
geompy.addToStudy(nozzle_outlet_line, 'Nozzle Outlet Line')
geompy.addToStudy(nozzle_sym_line, 'Nozzle Symmetry Axis Line')
geompy.addToStudy(polyline_nozzle, 'Nozzle Boundaries')
geompy.addToStudy(nozzle_surface, 'Nozzle Surface')
geompy.addToStudy(nozzle_surface_scale, 'Scaled Nozzle Surfaces')
geompy.addToStudy(nozzle_revolution, 'Nozzle Revolution')
geompy.addToStudy(nozzle_outlet_face, 'Nozzle Outlet Face')
geompy.addToStudy(nozzle_outlet_extrusion, 'Nozzle Outlet Face Extrusion')
geompy.addToStudy(nozzle_fuse, 'Nozzle Fuse')
geompy.addToStudy(nozzle_axial_translation, 'Nozzle Axial Translation')
geompy.addToStudy(nozzle_rotation, 'Nozzle Spouting Angle Rotation')
geompy.addToStudy(nozzle_radial_translation, 'Nozzle Radial Translation')
geompy.addToStudy(nozzle_cut_plane_extrusion, 'Nozzle Cut Plane Extrusion')
geompy.addToStudy(nozzle_rotor_inlet_plane_cut, 'Nozzle Rotor Inlet Cut')
geompy.addToStudy(nozzle_circular_pattern, 'Nozzle Circular Pattern')
#geompy.addToStudy(interpol_lower_transition, 'Lower Transition Line')
#geompy.addToStudy(interpol_lower_circular, 'Lower Circular Line')
#geompy.addToStudy(interpol_lower_sym_transition, 'Lower Symmetrical Transition Line')
#geompy.addToStudy(interpol_upper_straight, 'Upper Straight Line')
#geompy.addToStudy(interpol_upper_transition, 'Upper Transition Line')
#geompy.addToStudy(interpol_upper_circular, 'Upper Circular Line')
#geompy.addToStudy(interpol_upper_sym_transition, 'Upper Symmetrical Transitiom Line')
#geompy.addToStudy(interpol_upper_sym_straight, 'Upper Symmetrical Straight Line')
geompy.addToStudy(blade_upper_line, 'Blade Upper Line')
geompy.addToStudy(blade_lower_line, 'Blade Lower Line')
geompy.addToStudy(interpol_leading_edge, "Blade Leading Edge")
geompy.addToStudy(interpol_trailing_edge, "Blade Trailing Edge")
geompy.addToStudy(blade_surface, 'Blade Surface')
geompy.addToStudy(blade_scale, 'Scale Of Blade' )
geompy.addToStudy(blade_radial_translation, 'Blade Translation' )
geompy.addToStudy(blade_extrusion, 'Adding Thickness to the Blade' )
geompy.addToStudy(hub_circle, 'Hub Circle' )
geompy.addToStudy(tip_circle, 'Tip Circle' )
geompy.addToStudy(dummy_circle_to_cut, 'Dummy Circle')
geompy.addToStudy(blade_tip_circle, 'Blade Tip Circle')
geompy.addToStudy(hub_circle_face, 'Hub Circle Face' )
geompy.addToStudy(tip_circle_face, 'Tip Circle Face' )
geompy.addToStudy(blade_tip_face,'Blade Tip Circle Face')
geompy.addToStudy(dummy_circle_to_cut_face, 'Dummy Circle Face')
geompy.addToStudy(hub_face_extrusion, 'Hub Extrusion')
geompy.addToStudy(tip_face_extrusion, 'Casing Tip Extrusion')
geompy.addToStudy(blade_tip_face_extrusion, 'Blade Tip Extrusion')
geompy.addToStudy(dummy_face_extrusion,'Dummy Face Extrusion')
geompy.addToStudy(dummy_face_cut, 'Dummy Cut')
geompy.addToStudy(rotor_cut, 'Rotor Cut' )
#geompy.addToStudy(blade_upper_portion_cut, 'Blade Tip Cut')
geompy.addToStudy(blade_final_cut, 'Blade Final Cut')
geompy.addToStudy(blade_axial_translation, 'Final Blade Form')
#geompy.addToStudy(rotation_vector, 'Blade Rotation Vector')
#geompy.addToStudy(blade_pattern, 'Blade Circular Pattern')
#geompy.addToStudy(rotor_fluid_volume, 'Rotor Fluid Volume')
#geompy.addToStudy(turbine_fluid_volume, 'Turbine Fluid Volume')

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
