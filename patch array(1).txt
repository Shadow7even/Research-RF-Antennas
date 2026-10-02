"""
MTS: Unit Cell Simulation
Experiment 4

Automated construction of a transmission line metasurface antenna.
"""

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import scipy.signal
import socket
from datetime import datetime
import platform
import pyaedt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np

import uncc_mts_unit_cell as unit_cell
import uncc_mts_compute_config as compute_config

if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
else:
    os.environ["ANSYSEM_ROOT231"] = "C:\\Program Files\\AnsysEM\\v231\\Win64\\"

aedt_version = "2023.1"

solver_configuration = compute_config.SolverConfig().solver_config

###############################################################################
# Define program variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed_of_light = 2.99792458e8

frequency_GHz = 80.0
wavelength_mm = 1e3 * speed_of_light / (frequency_GHz * 1e9)

# DIELECTRIC MATERIALS
# dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "glass"
# dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
ground_plane_material_name = "pec"
# ground_plane_material_name = "copper"
unit_cell_material_name = "pec"
# unit_cell_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]  # green
dielectric_color = [255, 255, 128]  # yellow
radiation_box_color = [128, 255, 255]  # neon blue
subtract_tool_color = [64, 64, 64]  # dary gray
perfectly_matched_layer_color = [255, 128, 128]  # light red
port_color = [31, 81, 255]  # blue

colors = {"metal": [143, 175, 143],
          "dielectric": [255, 255, 128],
          "radiation box": [128, 255, 255],
          "boolean tool": [64, 64, 64],
          "perfectly matched layer": [255, 128, 128],
          "port": [31, 81, 255]}

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "PatchArray Antenna " + current_time_str
design_name = "PatchArray HFSS " + current_time_str

save_file_prefix = "patcharray_antenna_100GHz"
save_filename_no_extension = save_file_prefix + "_" + current_time_str + "_" + socket.gethostname()
save_filename_matlab = save_filename_no_extension + ".mat"
save_filename_log = save_filename_no_extension + ".log"
file_data = open(save_filename_log, "w")
optimization_parameters = ["num_patches", "trapezoid_width", "trapezoid_size_pct"]
column_titles = ["s21_avg", "s11_avg", "error"]
file_data.write("{}, {}, {}, {}, {}, {}\n".format(*optimization_parameters, *column_titles))

board_length_mm = 22

# strip_width_mm = 0.8  # mm
# gap_length_mm = 1.5  # mm
# patch_length_mm = 0.9  # mm

i_height_mm = 0.09  # mm   <=== dielectric slab height in millimeters
height_mm = i_height_mm

# i_strip_width_mm = wavelength_mm / 4 - 0.1  # mm
# i_gap_length_mm = wavelength_mm / 2  # mm
# i_patch_length_mm = wavelength_mm / 4  # mm
i_num_patches = 5
i_strip_width_mm = 0.2  # mm
i_strip_length_mm = wavelength_mm / 2  # mm
i_patch_width_mm = wavelength_mm / 2  # mm
i_patch_length_mm = wavelength_mm / 2  # mm
i_gap_width_mm = wavelength_mm / 10
i_gap_length_mm = wavelength_mm / 8
i_feed_trap_length_pct = 0.5

antenna_length_mm = i_num_patches * i_patch_length_mm + i_num_patches * (i_strip_length_mm + 1)  # mm
feed_length_mm = 0.5 * (board_length_mm - antenna_length_mm)
feed_trapezoid_start_width_mm = 0.25  # mm
feed_trapezoid_length_mm = i_feed_trap_length_pct * feed_length_mm  # mm
feed_rectangle_length_mm = feed_length_mm - feed_trapezoid_length_mm

board_margin_xy_mm = np.array([2, 0])
# board_width_mm = i_strip_width_mm + 0.5  # 2 * board_margin_xy_mm[0]
board_width_mm = 3  # 2 * board_margin_xy_mm[0]
antenna_dimensions_xy_mm = np.array([board_length_mm, board_width_mm])

non_linear_feval_count = 0


def construct_antenna(o_antenna_parameters):
    global hfss
    port_size_y_mm = 0.5  # mm

    o_antenna_parameters[0] = int(o_antenna_parameters[0])
    print("Current parameters:", o_antenna_parameters)
    component_attributes = []
    component_geometries = []

    # unpack optimization parameters
    o_num_patches = int(o_antenna_parameters[0])
    # o_antenna_length_mm = o_antenna_parameters[0]
    o_feed_trapezoid_start_width_mm = port_size_y_mm
    strip_width_mm = o_antenna_parameters[1]
    strip_length_mm = o_antenna_parameters[2]
    patch_width_mm = o_antenna_parameters[3]
    patch_length_mm = o_antenna_parameters[4]
    o_antenna_length_mm = o_num_patches * patch_length_mm + strip_length_mm * (o_num_patches + 1)  # mm
    o_feed_length_mm = 0.5 * (board_length_mm - o_antenna_length_mm)
    o_feed_trapezoid_length_mm = o_antenna_parameters[5] * o_feed_length_mm
    o_feed_rectangle_length_mm = o_feed_length_mm - o_feed_trapezoid_length_mm

    o_offset = -0.5 * o_antenna_length_mm
    patch_geom_list = []
    patch_name_list = []
    for patch_index in np.arange(0, o_num_patches):
        transmission_line_position = np.array([o_offset,
                                               -0.5 * strip_width_mm,
                                               0.5 * height_mm])
        transmission_line_size = np.array([strip_length_mm, strip_width_mm])
        transmission_line_params = {"name": "transmission_line_" + str(patch_index),
                                    "csPlane": "XY",
                                    "position": "{}mm,{}mm,{}mm".format(transmission_line_position[0],
                                                                        transmission_line_position[1],
                                                                        transmission_line_position[2]).split(","),
                                    "dimension_list": "{}mm,{}mm".format(transmission_line_size[0],
                                                                         transmission_line_size[1]).split(","),
                                    "matname": ground_plane_material_name,
                                    "is_covered": True}
        transmission_line_geom = hfss.modeler.create_rectangle(**transmission_line_params)
        transmission_line_geom.color = metal_color
        component_geometries.append(transmission_line_geom)
        patch_name_list.append(transmission_line_geom.name)

        patch_position = np.array([o_offset + strip_length_mm,
                                   -0.5 * patch_width_mm,
                                   0.5 * height_mm])
        patch_size = np.array([patch_length_mm, patch_width_mm])
        patch_params = {"name": "patch_" + str(patch_index),
                        "csPlane": "XY",
                        "position": "{}mm,{}mm,{}mm".format(patch_position[0],
                                                            patch_position[1],
                                                            patch_position[2]).split(","),
                        "dimension_list": "{}mm,{}mm".format(patch_size[0],
                                                             patch_size[1]).split(","),
                        "matname": ground_plane_material_name,
                        "is_covered": True}
        patch_geom = hfss.modeler.create_rectangle(**patch_params)
        patch_geom.color = metal_color
        o_offset += strip_length_mm + patch_length_mm
        patch_geom_list.append(patch_geom)
        patch_name_list.append(patch_geom.name)
        component_geometries.append(patch_geom)

    transmission_line_position = np.array([o_offset,
                                           -0.5 * strip_width_mm,
                                           0.5 * height_mm])
    transmission_line_size = np.array([strip_length_mm, strip_width_mm])
    transmission_line_params = {"name": "transmission_line_" + str(patch_index),
                                "csPlane": "XY",
                                "position": "{}mm,{}mm,{}mm".format(transmission_line_position[0],
                                                                    transmission_line_position[1],
                                                                    transmission_line_position[2]).split(","),
                                "dimension_list": "{}mm,{}mm".format(transmission_line_size[0],
                                                                     transmission_line_size[1]).split(","),
                                "matname": ground_plane_material_name,
                                "is_covered": True}
    transmission_line_geom = hfss.modeler.create_rectangle(**transmission_line_params)
    transmission_line_geom.color = metal_color
    component_geometries.append(transmission_line_geom)
    patch_name_list.append(transmission_line_geom.name)

    antenna_pec_boundary = hfss.assign_perfecte_to_sheets(
        **{"sheet_list": patch_name_list, "sourcename": None, "is_infinite_gnd": False})
    component_attributes.append(antenna_pec_boundary)

    port_index = 0
    #  build a feed structure having feed_length_mm length along the X axis (the propagation axis)
    feed_rect_length_mm = o_feed_rectangle_length_mm
    feed_rect_width_mm = strip_width_mm
    trapezoid_top_width = o_feed_trapezoid_start_width_mm
    trapezoid_bottom_width = strip_width_mm
    feed_rect_position = np.array([-0.5 * board_length_mm + o_feed_trapezoid_length_mm,
                                   -0.5 * strip_width_mm,
                                   0.5 * height_mm])
    feed_rect_size = np.array([o_feed_rectangle_length_mm, feed_rect_width_mm])
    feed_rect_params = {"name": "feed_rectanglar_portion_" + str(port_index),
                        "csPlane": "XY",
                        "position": "{}mm,{}mm,{}mm".format(feed_rect_position[0],
                                                            feed_rect_position[1],
                                                            feed_rect_position[2]).split(","),
                        "dimension_list": "{}mm,{}mm".format(feed_rect_size[0],
                                                             feed_rect_size[1]).split(","),
                        "matname": ground_plane_material_name,
                        "is_covered": True}
    feed_rect_geom_0 = hfss.modeler.create_rectangle(**feed_rect_params)
    feed_rect_geom_0.color = metal_color
    component_geometries.append(feed_rect_geom_0)

    board_edge = [-0.5 * board_length_mm, -0.5 * o_feed_trapezoid_start_width_mm]
    trap_position_list = np.array([np.array([board_edge[0],
                                             board_edge[1], feed_rect_position[2]]),
                                   np.array([board_edge[0],
                                             board_edge[1] + o_feed_trapezoid_start_width_mm, feed_rect_position[2]]),
                                   np.array([board_edge[0] + o_feed_trapezoid_length_mm,
                                             0.5 * strip_width_mm, feed_rect_position[2]]),
                                   np.array([board_edge[0] + o_feed_trapezoid_length_mm,
                                             -0.5 * strip_width_mm, feed_rect_position[2]])])
    trap_position_list = [elem.tolist() for elem in trap_position_list]
    trap_polyline_params = {
        "position_list": trap_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": "feed_trapezoid_portion_" + str(port_index),
        "matname": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    feed_trap_geom_0 = hfss.modeler.create_polyline(**trap_polyline_params)
    feed_trap_geom_0.color = metal_color
    feed_trap_geom_0.transparency = 0
    component_geometries.append(feed_trap_geom_0)

    port_index = 1
    #  build a feed structure having feed_length_cm length along the Y axis (the propagation axis)
    feed_rect_position = np.array([0.5 * board_length_mm - o_feed_trapezoid_length_mm - o_feed_rectangle_length_mm,
                                   -0.5 * strip_width_mm,
                                   0.5 * height_mm])
    feed_rect_size = np.array([o_feed_rectangle_length_mm, feed_rect_width_mm])
    feed_rect_params = {"name": "feed_rectanglar_portion_" + str(port_index),
                        "csPlane": "XY",
                        "position": "{}mm,{}mm,{}mm".format(feed_rect_position[0],
                                                            feed_rect_position[1],
                                                            feed_rect_position[2]).split(","),
                        "dimension_list": "{}mm,{}mm".format(feed_rect_size[0],
                                                             feed_rect_size[1]).split(","),
                        "matname": ground_plane_material_name,
                        "is_covered": True}
    feed_rect_geom_1 = hfss.modeler.create_rectangle(**feed_rect_params)
    feed_rect_geom_1.color = metal_color
    component_geometries.append(feed_rect_geom_1)

    board_edge = [0.5 * board_length_mm, -0.5 * o_feed_trapezoid_start_width_mm]
    trap_position_list = np.array([np.array([board_edge[0],
                                             board_edge[1], feed_rect_position[2]]),
                                   np.array([board_edge[0],
                                             board_edge[1] + o_feed_trapezoid_start_width_mm, feed_rect_position[2]]),
                                   np.array([board_edge[0] - o_feed_trapezoid_length_mm,
                                             0.5 * strip_width_mm, feed_rect_position[2]]),
                                   np.array([board_edge[0] - o_feed_trapezoid_length_mm,
                                             -0.5 * strip_width_mm, feed_rect_position[2]])])
    trap_position_list = [elem.tolist() for elem in trap_position_list]
    trap_polyline_params = {
        "position_list": trap_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": "feed_trapezoid_portion_" + str(port_index),
        "matname": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    feed_trap_geom_1 = hfss.modeler.create_polyline(**trap_polyline_params)
    feed_trap_geom_1.color = metal_color
    feed_trap_geom_1.transparency = 0
    component_geometries.append(feed_trap_geom_1)

    feed_pec_boundary = hfss.assign_perfecte_to_sheets(
        **{"sheet_list": [feed_rect_geom_0.name, feed_trap_geom_0.name,
                          feed_rect_geom_1.name, feed_trap_geom_1.name],
           "sourcename": None,
           "is_infinite_gnd": False})
    # feed_pec_boundary = hfss.assign_perfecte_to_sheets(
    #     **{"sheet_list": [feed_rect_geom_0.name, feed_trap_geom_0.name],
    #        "sourcename": None,
    #        "is_infinite_gnd": False})
    component_attributes.append(feed_pec_boundary)

    port_1_position = np.array([-0.5 * board_length_mm,
                                # -0.5 * o_feed_trapezoid_start_width_mm,
                                -0.5 * port_size_y_mm,
                                -0.5 * height_mm])
    # port_1_size = np.array([o_feed_trapezoid_start_width_mm, height_mm])
    port_1_size = np.array([port_size_y_mm, height_mm])
    port_1_params = {"name": "port_1",
                     "csPlane": "YZ",
                     "position": "{}mm,{}mm,{}mm".format(port_1_position[0],
                                                         port_1_position[1],
                                                         port_1_position[2]).split(","),
                     "dimension_list": "{}mm,{}mm".format(port_1_size[0],
                                                          port_1_size[1]).split(","),
                     "matname": None,
                     "is_covered": True}
    port_1_geom = hfss.modeler.create_rectangle(**port_1_params)
    port_1_geom.color = radiation_box_color
    component_geometries.append(port_1_geom)

    port_1_excitation_params = {"signal": port_1_geom,
                                "reference": ground_plane_geom,
                                "create_port_sheet": False,
                                "port_on_plane": True,
                                "integration_line": 0,
                                "impedance": 50,
                                "name": "port_1_excitation",
                                "renormalize": True,
                                "deembed": False,
                                "terminals_rename": True}
    lumped_port_1 = hfss.lumped_port(**port_1_excitation_params)
    component_attributes.append(lumped_port_1)

    port_2_position = np.array([0.5 * board_length_mm,
                                # -0.5 * o_feed_trapezoid_start_width_mm,
                                -0.5 * port_size_y_mm,
                                -0.5 * height_mm])
    port_2_size = port_1_size
    port_2_params = {"name": "port_2",
                     "csPlane": "YZ",
                     "position": "{}mm,{}mm,{}mm".format(port_2_position[0],
                                                         port_2_position[1],
                                                         port_2_position[2]).split(","),
                     "dimension_list": "{}mm,{}mm".format(port_2_size[0],
                                                          port_2_size[1]).split(","),
                     "matname": None,
                     "is_covered": True}
    port_2_geom = hfss.modeler.create_rectangle(**port_2_params)
    port_2_geom.color = radiation_box_color
    component_geometries.append(port_2_geom)

    port_2_excitation_params = {"signal": port_2_geom,
                                "reference": ground_plane_geom,
                                "create_port_sheet": False,
                                "port_on_plane": True,
                                "integration_line": 0,
                                "impedance": 50,
                                "name": "port_2_excitation",
                                "renormalize": True,
                                "deembed": False,
                                "terminals_rename": True}
    lumped_port_2 = hfss.lumped_port(**port_2_excitation_params)
    component_attributes.append(lumped_port_2)
    return component_attributes + component_geometries


def antenna_design_error_function(o_antenna_parameters):
    global non_linear_feval_count
    design_elements = construct_antenna(o_antenna_parameters)
    setup_ok = hfss.validate_full_design()
    setup_solver_configuration_params = {
        "name": "LW_TL_Antenna_Setup",
        "num_cores": solver_configuration["num_cores"],
        "num_tasks": 1,
        "num_gpu": solver_configuration["num_gpu"],
        "acf_file": None,
        "use_auto_settings": True,
        "num_variations_to_distribute": None,
        "allowed_distribution_types": None,
        "revert_to_initial_mesh": False,
        "blocking": True
    }
    hfss.analyze_setup(**setup_solver_configuration_params)

    non_linear_feval_count += 1

    s11_data_channel_str = "dB(St(port_1_excitation_T1,port_1_excitation_T1))"
    s21_data_channel_str = "dB(St(port_1_excitation_T1,port_2_excitation_T1))"
    gain_theta_channel_str = "dB(GainTheta)"

    s11_solution_data = hfss.post.get_solution_data(expressions=s11_data_channel_str,
                                                    setup_sweep_name="LW_TL_Antenna_Setup : LastAdaptive",
                                                    report_category="Terminal S Parameter")
    s21_solution_data = hfss.post.get_solution_data(expressions=s21_data_channel_str,
                                                    setup_sweep_name="LW_TL_Antenna_Setup : LastAdaptive",
                                                    report_category="Terminal S Parameter")
    gain_theta_phi_0_data = hfss.post.get_solution_data(expressions=gain_theta_channel_str,
                                                        setup_sweep_name="LW_TL_Antenna_Setup : LastAdaptive",
                                                        variations={"Freq": "{}GHz".format(int(frequency_GHz)),
                                                                    "Phi": "0deg"},
                                                        primary_sweep_variable="Theta",
                                                        report_category="Far Fields",
                                                        context="Elevation")

    gain_theta_phi_90_data = hfss.post.get_solution_data(expressions=gain_theta_channel_str,
                                                         setup_sweep_name="LW_TL_Antenna_Setup : LastAdaptive",
                                                         variations={"Freq": "{}GHz".format(int(frequency_GHz)),
                                                                     "Phi": "90deg"},
                                                         primary_sweep_variable="Theta",
                                                         report_category="Far Fields",
                                                         context="Elevation")

    vals_np_real = np.array(list(s11_solution_data.full_matrix_real_imag[0][s11_data_channel_str].values()))
    s11_vals = np.squeeze(vals_np_real)
    s11_freqs = np.array(s21_solution_data.primary_sweep_values)
    s11_freqs_units_str = s11_solution_data.units_sweeps['Freq']  # 'GHz'
    vals_np_real = np.array(
        list(s21_solution_data.full_matrix_real_imag[0][s21_data_channel_str].values()))
    s21_vals = np.squeeze(vals_np_real)
    s21_freqs = s21_solution_data.primary_sweep_values
    s21_freqs_units_str = s21_solution_data.units_sweeps['Freq']  # 'GHz'
    s11_vals = s11_vals.reshape(s11_vals.size)
    s21_vals = s21_vals.reshape(s21_vals.size)
    freq_idxs_of_interest = np.argwhere((s11_freqs > frequency_GHz - 1) & (s11_freqs < frequency_GHz + 1))
    freq_idxs_of_interest = freq_idxs_of_interest.flatten()
    s21_term = np.average(s21_vals[freq_idxs_of_interest])
    s11_term = np.average(s11_vals[freq_idxs_of_interest])

    gain_theta_phi_0_vals = np.array(
        list(gain_theta_phi_0_data.full_matrix_real_imag[0][gain_theta_channel_str].values()))
    gain_theta_phi_90_vals = np.array(
        list(gain_theta_phi_90_data.full_matrix_real_imag[0][gain_theta_channel_str].values()))
    gain_theta_angles = np.array(gain_theta_phi_0_data.primary_sweep_values)
    theta_idxs_of_interest = np.argwhere((gain_theta_angles > -180) & (gain_theta_angles < 180)).flatten()
    theta_idxs_of_interest.sort()
    import scipy.signal.windows as windows
    # gaussian_window_samples = windows.gaussian(len(theta_idxs_of_interest), 15)
    # Shape parameter. p = 1 is identical to gaussian, p = 0.5 is the same shape as the Laplace distribution.
    gaussian_window_samples = windows.general_gaussian(theta_idxs_of_interest.size, p=0.5, sig=10, sym=True)
    gauss_max = np.max(gaussian_window_samples)
    gauss_min = np.min(gaussian_window_samples)
    gaussian_window_samples_shifted = 20 * ((gaussian_window_samples - gauss_min) / gauss_max)
    gain_avg = np.average(gain_theta_phi_0_vals[theta_idxs_of_interest])
    gain_term = -np.average(gaussian_window_samples_shifted * gain_theta_phi_0_vals[theta_idxs_of_interest])

    error = -20 * gain_term + 3 * s11_term + s21_term
    print("Error = {} Gain = {} S11 = {} S21 = {}".format(error, -1 * gain_term, 3 * s11_term, s21_term))
    file_data.write("{:.2f}, {:.2f}, {:.2f}, ".format(*o_antenna_parameters) +
                    "{:.2f}, {:.2f}, {:.2f}\n".format(np.average(s21_vals), np.average(s11_vals), error))

    plt.close(1)
    plt.close(2)

    plt.figure(1)
    plt.plot(gain_theta_angles[theta_idxs_of_interest], gaussian_window_samples_shifted)
    plt.plot(gain_theta_angles[theta_idxs_of_interest], gain_theta_phi_0_vals[theta_idxs_of_interest])
    plt.plot(gain_theta_angles[theta_idxs_of_interest],
             gaussian_window_samples_shifted * gain_theta_phi_0_vals[theta_idxs_of_interest])
    plt.draw()
    plt.pause(1)

    plt.figure(2)
    plt.plot(gain_theta_angles, gain_theta_phi_0_vals)
    plt.title("Antenna Gain @  {:.1f} GHz iteration {} S11={:.1f} S21={:.1f}".format(frequency_GHz,
                                                                                     non_linear_feval_count,
                                                                                     s11_term, s21_term))
    plt.plot(gain_theta_angles, gain_theta_phi_90_vals)
    plt.ylabel("Gain (dB)")
    x_axis_units_str = str(gain_theta_phi_0_data.units_sweeps['Theta'])
    plt.xlabel("Theta (" + x_axis_units_str + ")")
    plt.legend(["Phi=0 deg", "Phi=90 deg"])
    plt.draw()
    plt.pause(1)

    plt.savefig('antenna_opt_gain_it{:03d}.png'.format(non_linear_feval_count))

    plot_obj = hfss.plot(show=False, view="xy", plot_air_objects=False,
                         show_legend=False, force_opacity_value=True)
    plot_obj.background_color = [153, 203, 255]
    plot_obj.zoom = 1.0
    plot_obj.show_grid = False
    plot_obj.show_axes = False
    plot_obj.bounding_box = False
    plot_obj.plot("antenna_opt_model_it{:03d}.jpg".format(non_linear_feval_count))

    for element in design_elements:
        element.delete()
    return error


def matplotlib_pyplot_setup():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('x-axis', fontsize = SMALL_SIZE)
    # plt.rc('y-axis', fontsize = SMALL_SIZE)


matplotlib_pyplot_setup()

###############################################################################
# Set non-graphical mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
non_graphical = False

###############################################################################
# Launch AEDT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NewThread = True
desktop = pyaedt.launch_desktop(specified_version=aedt_version,
                                non_graphical=non_graphical,
                                new_desktop_session=NewThread)

# Solution Types are: { "Modal", "Terminal", "Eigenmode", "Transient Network", "SBR+", "Characteristic"}
hfss = pyaedt.Hfss(
    specified_version=aedt_version,
    solution_type="Terminal",
    new_desktop_session=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical
)

hfss.modeler.model_units = 'mm'
# cm2mm = 10
hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_plane_position = -0.5 * np.array([antenna_dimensions_xy_mm[0],
                                         antenna_dimensions_xy_mm[1] + 4,
                                         height_mm])
ground_plane_size = np.array([antenna_dimensions_xy_mm[0],
                              antenna_dimensions_xy_mm[1] + 4])

#  csPlane is either "XY", "YZ", or "XZ"

ground_plane_params = {"name": "cell_ground_plane",
                       "csPlane": "XY",
                       "position": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
                                                           ground_plane_position[1],
                                                           ground_plane_position[2]).split(","),
                       "dimension_list": "{}mm,{}mm".format(ground_plane_size[0],
                                                            ground_plane_size[1]).split(","),
                       "matname": ground_plane_material_name,
                       "is_covered": True}
ground_plane_geom = hfss.modeler.create_rectangle(**ground_plane_params)
ground_plane_geom.color = metal_color

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": [ground_plane_geom.name],
       "sourcename": None,
       "is_infinite_gnd": False})

# FIT ALL
hfss.modeler.fit_all()

dielectric_slab_position = -0.5 * np.array([antenna_dimensions_xy_mm[0],
                                            antenna_dimensions_xy_mm[1],
                                            height_mm])
dielectric_slab_size = np.array([antenna_dimensions_xy_mm[0],
                                 antenna_dimensions_xy_mm[1],
                                 height_mm])
dielectric_slab_params = {"name": "dielectric_slab",
                          "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
                                                              dielectric_slab_position[1],
                                                              dielectric_slab_position[2]).split(","),
                          "dimensions_list": "{}mm,{}mm,{}mm".format(dielectric_slab_size[0],
                                                                     dielectric_slab_size[1],
                                                                     dielectric_slab_size[2]).split(","),
                          "matname": dielectric_material_name}
dielectric_slab_geom = hfss.modeler.create_box(**dielectric_slab_params)
dielectric_slab_geom.color = dielectric_color

open_region_params = {
    "Frequency": "{}GHz".format(frequency_GHz),
    "Boundary": "Radiation",
    "ApplyInfiniteGP": False,
    "GPAXis": "-z"}
success = hfss.create_open_region(**open_region_params)

solver_setup = hfss.create_setup(setupname="LW_TL_Antenna_Setup", setuptype="HFSSDriven")
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '{}GHz'.format(frequency_GHz),
                       "MaxDeltaS": 0.03,
                       "PortsOnly": False,
                       "UseMatrixConv": False,
                       "MaximumPasses": 30,
                       "MinimumPasses": 1,
                       "MinimumConvergedPasses": 1,
                       "PercentRefinement": 30,
                       "IsEnabled": True,
                       # ('MeshLink', SetupProps([('ImportMesh', False)])),
                       "BasisOrder": 1,
                       "DoLambdaRefine": True,
                       "DoMaterialLambda": True,
                       "SetLambdaTarget": False,
                       "Target": 0.3333,
                       "UseMaxTetIncrease": False,
                       "PortAccuracy": 2,
                       "UseABCOnPort": False,
                       "SetPortMinMaxTri": False,
                       "UseDomains": False,
                       "UseIterativeSolver": False,
                       "SaveRadFieldsOnly": False,
                       "SaveAnyFields": True,
                       "IESolverType": "Auto",
                       "LambdaTargetForIESolver": 0.15,
                       "UseDefaultLambdaTgtForIESolver": True,
                       "IE Solver Accuracy": 'Balanced'
                       }
solver_setup.props.update(solver_setup_params)

# frequency_sweep_params = {
#      "unit": "GHz",
#      "freqstart": frequency_GHz - 1.5,
#      "freqstop": frequency_GHz + 1.5,
#      "num_of_freq_points": 200,
#      "sweepname": "sweep",
#      "save_fields": True,
#      "save_rad_fields": False,
#      "sweep_type": "Discrete",
#      "interpolation_tol": 0.5,
#      "interpolation_max_solutions": 250
#  }
# solver_setup.create_frequency_sweep(**frequency_sweep_params)

##########################
# NON-LINEAR OPTIMIZER   #
##########################
from scipy.optimize import minimize

initial_antenna_parameters = [i_num_patches,
                              i_strip_width_mm, i_strip_length_mm,
                              i_patch_width_mm, i_patch_length_mm,
                              i_feed_trap_length_pct]
# i_gap_length_mm, i_gap_length_mm]
parameter_bounds = [(2, 5),
                    (i_strip_width_mm * 0.5, i_strip_width_mm * 1.2),
                    (i_strip_length_mm * 0.1, i_strip_length_mm * 1.2),
                    (i_patch_width_mm * 0.5, i_patch_width_mm * 1.2),
                    (i_patch_length_mm * 0.5, i_patch_length_mm * 1.2),
                    (0.1, 0.9)]
# (i_gap_width_mm * 0.5, i_gap_length_mm *1.5),
# (i_gap_length_mm - 0.7, i_gap_length_mm + 0.3)]
minimize_options = {"maxiter": 80, "disp": True, "eps": [1, 0.1, 0.1, 0.1, 0.1, 0.05]}
# Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, and COBYLA methods
result = minimize(antenna_design_error_function, initial_antenna_parameters,
                  bounds=parameter_bounds, method="L-BFGS-B", options=minimize_options)

print("Optimal parameters:", result.x)
print("Minimum value:", result.fun)
file_data.close()
# setup_ok = hfss.validate_full_design()
#
# setup_solver_configuration_params = {
#     "name": "LW_TL_Antenna_Setup",
#     "num_cores": solver_configuration["num_cores"],
#     "num_tasks": 1,
#     "num_gpu": solver_configuration["num_gpu"],
#     "acf_file": None,
#     "use_auto_settings": True,
#     "num_variations_to_distribute": None,
#     "allowed_distribution_types": None,
#     "revert_to_initial_mesh": False,
#     "blocking": True
# }
# hfss.analyze_setup(**setup_solver_configuration_params)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)
# matlab_dict = {
#     # "num_cells": len(database),
#     "global_cell_size": cell_size,
#     # "database": database,
#     "compute_start_timestamp": current_time_str,
#     "compute_host": socket.gethostname(),
#     "compute_duration": time_difference_str
# }
#
# scipy.io.savemat(save_filename_matlab, matlab_dict)
###############################################################################
# Close Ansys Electronics Desktop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hfss.release_desktop(close_projects=True, close_desktop=True)

# analysis_plane_position = np.array([ground_plane_position[0], ground_plane_position[1], 0])
# analysis_plane_size = ground_plane_size
# analysis_plane_params = {"name": "plot_waveguide_mode",
#                          "csPlane": "XY",
#                          "position": "{}mm,{}mm,{}mm".format(analysis_plane_position[0],
#                                                              analysis_plane_position[1],
#                                                              analysis_plane_position[2]).split(","),
#                          "dimension_list": "{}mm,{}mm".format(analysis_plane_size[0],
#                                                               analysis_plane_size[1]).split(","),
#                          "matname": None,
#                          "is_covered": True}
# analysis_plane_geom = hfss.modeler.create_rectangle(**analysis_plane_params)
# analysis_plane_geom.color = radiation_box_color
#
# nf_x_direction = [1, 0, 0]
# nf_y_direction = [0, 1, 0]
# nf_z_direction = [0, 0, 1]
# top_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0, 0, 2 * wavelength_mm).split(","),
#                              "reference_cs": "Global",
#                              "name": "top_2lambda_CS",
#                              "mode": "axis",
#                              "view": "iso",
#                              "x_pointing": nf_x_direction,
#                              "y_pointing": nf_y_direction,
#                              "psi": 0,
#                              "theta": 0,
#                              "phi": 0,
#                              "u": None
#                              }
# top_plane_field_cs = hfss.modeler.create_coordinate_system(**top_plane_field_cs_params)
#
# bottom_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0, 0, -2 * wavelength_mm).split(","),
#                                 "reference_cs": "Global",
#                                 "name": "bottom_2lambda_CS",
#                                 "mode": "axis",
#                                 "view": "iso",
#                                 "x_pointing": nf_x_direction,
#                                 "y_pointing": nf_y_direction,
#                                 "psi": 0,
#                                 "theta": 0,
#                                 "phi": 0,
#                                 "u": None
#                                 }
# bottom_plane_field_cs = hfss.modeler.create_coordinate_system(**bottom_plane_field_cs_params)
#
# left_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0,
#                                                                 -0.5 * board_width_mm - 2 * wavelength_mm,
#                                                                 0).split(","),
#                               "reference_cs": "Global",
#                               "name": "left_2lambda_CS",
#                               "mode": "axis",
#                               "view": "iso",
#                               "x_pointing": nf_x_direction,
#                               "y_pointing": nf_z_direction,
#                               "psi": 0,
#                               "theta": 0,
#                               "phi": 0,
#                               "u": None
#                               }
# left_plane_field_cs = hfss.modeler.create_coordinate_system(**left_plane_field_cs_params)
# right_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0,
#                                                                  0.5 * board_width_mm + 2 * wavelength_mm,
#                                                                  0).split(","),
#                                "reference_cs": "Global",
#                                "name": "right_2lambda_CS",
#                                "mode": "axis",
#                                "view": "iso",
#                                "x_pointing": nf_x_direction,
#                                "y_pointing": nf_z_direction,
#                                "psi": 0,
#                                "theta": 0,
#                                "phi": 0,
#                                "u": None
#                                }
# right_plane_field_cs = hfss.modeler.create_coordinate_system(**right_plane_field_cs_params)
#
# back_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(-0.5 * board_length_mm - 2 * wavelength_mm,
#                                                                 0,
#                                                                 0).split(","),
#                               "reference_cs": "Global",
#                               "name": "back_2lambda_CS",
#                               "mode": "axis",
#                               "view": "iso",
#                               "x_pointing": nf_y_direction,
#                               "y_pointing": nf_z_direction,
#                               "psi": 0,
#                               "theta": 0,
#                               "phi": 0,
#                               "u": None
#                               }
# back_plane_field_cs = hfss.modeler.create_coordinate_system(**back_plane_field_cs_params)
# front_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0.5 * board_length_mm + 2 * wavelength_mm,
#                                                                  0,
#                                                                  0).split(","),
#                                "reference_cs": "Global",
#                                "name": "front_2lambda_CS",
#                                "mode": "axis",
#                                "view": "iso",
#                                "x_pointing": nf_y_direction,
#                                "y_pointing": nf_z_direction,
#                                "psi": 0,
#                                "theta": 0,
#                                "phi": 0,
#                                "u": None
#                                }
# front_plane_field_cs = hfss.modeler.create_coordinate_system(**front_plane_field_cs_params)
