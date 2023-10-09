"""
make_SWMM_inp.py 
@author: Xiating Chen
Last Edited: 2023/10/07

This code is to run SWMM and record results. 
    - Input: network and green infrasturcture configurations, environmental conditions (e.g. rainfall hydrographs)
    - Output: hydrological outcomes (e.g. peak flow, flooding)

Note: 
    1. The file is written in US/imperial unit.  
    2. Remember to specify "my_swmm_path" in line 24 to map to the SWMM engine
    on your computer.
"""

import hydro_network as hn
import numpy as np
import pandas as pd
import pickle
import subprocess
import os
from math import prod

my_swmm_path = '/Users/xchen/Applications/swmm5/build/runswmm5'

def make_inp(f,outlet_node,soil_nodes,simulation_date,infiltration,pcntimp,flowrouting,precip_name,graph,flood_level,antecedent_soil_moisture,mean_rainfall_inch):
    """ make input file for SWMM """
    info_header(f=f,simulation_date=simulation_date,infiltration=infiltration,flowrouting=flowrouting)
    info_evaporations(f=f,et_rate=5)
    info_temperature(f=f)
    info_raingages(f=f,precip_name=precip_name,raingage=1)
    info_subcatchments(f=f,graph=graph,raingage=1,soil_nodes=soil_nodes,outlet_node=outlet_node,pcntimp=pcntimp)
    info_subareas(f=f,graph=graph,outlet_node=outlet_node)
    info_infiltration(f=f,infiltration=infiltration,graph=graph,outlet_node=outlet_node)
    info_lid_controls(f=f)
    info_lid_usage(f=f,graph=graph,soil_nodes=soil_nodes,initsat=antecedent_soil_moisture)
    info_snowpacks(f=f)
    info_junctions(f=f,graph=graph,flood_level=flood_level,outlet_node=outlet_node)
    info_outfalls(f=f,graph=graph,outlet_node=outlet_node)
    info_conduits(f=f,graph=graph)
    info_xsections(f=f,graph=graph)
    info_timeseries(f=f,simulation_date=simulation_date,precip_name=precip_name,precip_duration_hour=2,total_precip=mean_rainfall_inch,precip_interval=1)
    info_report(f=f)
    info_tag(f=f)
    info_map(f=f) 
    info_coordinates(f=f,graph=graph)
    info_vertices(f=f)
    info_polygons(f=f)
    info_symbols(f=f)
    info_backdrop(f=f)

def info_header(f,simulation_date,infiltration='HORTON',flowrouting='KINWAVE'):
    lines = ['[TITLE]', 
    ';;Project Title/Notes\n', 
    '[OPTIONS]',
    ';;Option             Value',
    'FLOW_UNITS           CFS',
    'INFILTRATION         '+infiltration,
    'FLOW_ROUTING         '+flowrouting,
    'LINK_OFFSETS         DEPTH',
    'MIN_SLOPE            0',
    'ALLOW_PONDING        NO',
    'SKIP_STEADY_STATE    NO\n',
    'START_DATE           '+simulation_date,
    'START_TIME           00:00:00',
    'REPORT_START_DATE    06/01/2021',
    'REPORT_START_TIME    00:00:00',
    'END_DATE             06/02/2021',
    'END_TIME             06:00:00',
    'SWEEP_START          01/01',
    'SWEEP_END            12/31',
    'DRY_DAYS             0',
    'REPORT_STEP          00:15:00',
    'WET_STEP             00:05:00',
    'DRY_STEP             00:15:00',
    'ROUTING_STEP         0:00:2.5 ',
    'RULE_STEP            00:00:00\n',
    'INERTIAL_DAMPING     PARTIAL',
    'NORMAL_FLOW_LIMITED  BOTH',
    'FORCE_MAIN_EQUATION  H-W',
    'VARIABLE_STEP        0.75',
    'LENGTHENING_STEP     0',
    'MIN_SURFAREA         1.167',
    'MAX_TRIALS           8',
    'HEAD_TOLERANCE       0.0015',
    'SYS_FLOW_TOL         5',
    'LAT_FLOW_TOL         5',
    'MINIMUM_STEP         0.5',
    'THREADS              1']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_evaporations(f,et_rate,option_count = 17):
    constant = add_whitespace('CONSTANT', option_count, et_rate)
    dry_only = add_whitespace('DRY_ONLY', option_count, 'NO')
    lines = ['[EVAPORATION]',
';;Data Source    Parameters',
';;-------------- ----------------', constant, dry_only]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_temperature(f):
    lines = [';;Data Element     Values     ']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_raingages(f,precip_name,raingage):
    raingage = '1                VOLUME    0:01     1.0      TIMESERIES '+precip_name
    lines = ['[RAINGAGES]',
';;Name           Format    Interval SCF      Source',    
';;-------------- --------- ------ ------ ----------', 
raingage]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_subcatchments(f,graph,raingage,soil_nodes,outlet_node, pcntimp):
    subcatchments = [] ## This is where the node information is entered.
    for node in graph.nodes():
        if node == outlet_node:
            continue
        name = 'SC'+str(node).replace(', ','_')
        raingage = raingage
        outlet = str(node).replace(', ','_')
        totalarea = graph.nodes[node].get('node_drainage_area')
        if pcntimp is None:
            pcntimp=0 if node in soil_nodes else 100
        width = 5
        pcntslope = 0.5
        curblength = 0
        snowpack = ''
        subcatchment = add_whitespace(name,17,'') + add_whitespace(raingage, 17, '') + \
                add_whitespace(outlet,17,'')+add_whitespace(totalarea,9,'')+ \
                add_whitespace(pcntimp,9,'')+add_whitespace(width,9,'') + \
                add_whitespace(pcntslope,9,'')+add_whitespace(curblength,9,'') + \
                add_whitespace(snowpack,8,'')+'\n'
        subcatchments.append(subcatchment)
    lines = ['[SUBCATCHMENTS]',
';;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack      ',  
';;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- ----------------',
subcatchments]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_subareas(f,graph,outlet_node):
    subareas = []
    for node in graph.nodes():
        if node == outlet_node:
            continue
        subcatchment = 'SC'+str(node).replace(', ','_')
        nimperv = 0.014
        nperv = 0.2
        simperv = 0.02
        sperv = 0.1
        pctzero = 25
        pcntslope = 0.5
        routeto = 'OUTLET'
        snowpack = ''
        subarea = add_whitespace(subcatchment,17,'') + add_whitespace(nimperv, 11, '') + \
                add_whitespace(nperv,11,'')+add_whitespace(simperv,11,'')+ \
                add_whitespace(sperv,11,'')+add_whitespace(pctzero,11,'') + \
                add_whitespace(routeto,10,'')+ '\n'
        subareas.append(subarea)
    lines = ['[SUBAREAS]',
';;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted ',
';;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------',
subareas]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')
    
def info_infiltration(f,graph,outlet_node,infiltration = 'HORTON'):
    if infiltration == 'HORTON':
        infiltration = []
        for node in graph.nodes():
            if node == outlet_node:
                continue
            subcatchment = 'SC'+str(node).replace(', ','_')
            maxrate = 3.0
            minrate = 0.5
            decay = 4
            drytime = 7
            maxinfil = 0
            infil_line = add_whitespace(subcatchment,17,'')+add_whitespace(maxrate,11,'')+ \
                add_whitespace(minrate,11,'') + add_whitespace(decay,11,'') + \
                add_whitespace(drytime,11,'') + add_whitespace(maxinfil,10,'') + '\n'
            infiltration.append(infil_line)  
        lines = ['[INFILTRATION]',
# ';;Subcatchment   MaxRate    MinRate    Decay      DryTime    MaxInfil  ',
';;Subcatchment   Param1     Param2     Param3     Param4     Param5    ',
';;-------------- ---------- ---------- ---------- ---------- ----------',
infiltration]
    elif infiltration == 'GREEN-AMPT':
        infiltration = []
        for node in graph.nodes():
            if node == outlet_node:
                continue
            subcatchment = 'SC'+str(node).replace(', ','_')
            suction = 10
            hydcon = 0.25
            imdmax = 0.1
            infil_line = add_whitespace(subcatchment,17,'')+add_whitespace(suction,11,'')+ \
            add_whitespace(hydcon,11,'')+add_whitespace(imdmax,10,'')+'\n'
            infiltration.append(infil_line)
        lines = ['[INFILTRATION]',
';;Subcatchment   Suction    HydCon     IMDmax    ',
';;-------------- ---------- ---------- ----------',
infiltration]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_lid_controls(f, name = 'bioret_cell', type = 'BC', surf_height = 9, surf_veg = 0.1, surf_n = 0.1, 
surf_slope = 1.0, soil_height = 24, soil_n = 0.43, soil_fc = 0.4, soil_wp = 0.18,
soil_k = 1.5, soil_kslope = 40, soil_psi = 3.5, stor_height = 0, stor_voidratio = 0.25,
stor_seepage = 1.3, stor_clog = 0, drain_coef = 0, drain_exp = 0.5, drain_offset = 6,
drain_open = 0, drain_close = 0):
    """
    This is for the LID module, specifically for the bio-retention cell or a rain garden. 
    The soil properties for sandy loam soil are used here for reference. 
        surf_height: Berm Height (in. or mm)
        surf_veg: Vegetation Volume Fraction
        surf_n: Surface Roughness (Mannings n)
        surf_slope: Surface Slope (percent)
        soil_height: Soil Thickness (in. or mm)
        soil_n: Porosity (volume fraction)
        soil_fc: Field Capacity (volume fraction)
        soil_wp: Wilting Point (volume fraction)
        soil_k: Conductivity (in/hr or mm/hr)
        soil_kslope: Conductivity Slope
        soil_psi: Suction Head (in. or mm)
        stor_height: Storage Thickness (in. or mm)
        stor_voidratio: Void Ratio (Voids/Solids)
        stor_seepage: Seepage Rate (in/hr or mm/hr)
        stor_clog: Clogging Factor
        drain_coef: Flow Coefficient
        drain_exp: Flow Exponent
        drain_offset: Offset (in. or mm)
        drain_open: Open Level (in. or mm)
        drain_close: Closed Level (in. or mm)
    """
    lid_type = add_whitespace(name,17,'') + add_whitespace(type,11,'')
    lid_surface = add_whitespace(name,17,'') + add_whitespace('SURFACE',11,'') + \
        add_whitespace(surf_height,11,'') + add_whitespace(surf_veg,11,'') + \
        add_whitespace(surf_n,11,'') + add_whitespace(surf_slope,11,'') + \
        add_whitespace('5',11,'')
    lid_soil = add_whitespace(name,17,'') + add_whitespace('SOIL',11,'') + \
        add_whitespace(soil_height,11,'') + add_whitespace(soil_n,11,'') + \
        add_whitespace(soil_fc,11,'') + add_whitespace(soil_wp,11,'') + \
        add_whitespace(soil_k,11,'') + add_whitespace(soil_kslope,11,'') + \
        add_whitespace(soil_psi,11,'')
    lid_storage = add_whitespace(name,17,'') + add_whitespace('STORAGE',11,'') + \
        add_whitespace(stor_height,11,'') + add_whitespace(stor_voidratio,11,'') + \
        add_whitespace(stor_seepage,11,'') + add_whitespace(stor_clog,11,'') 
    lid_drain = add_whitespace(name,17,'') + add_whitespace('DRAIN',11,'') + \
        add_whitespace(drain_coef,11,'') + add_whitespace(drain_exp,11,'') + \
        add_whitespace(drain_offset,11,'') + add_whitespace(drain_offset,11,'') + \
        add_whitespace(drain_open,11,'') + add_whitespace(drain_close,11,'')
    lines = ['[LID_CONTROLS]',
';;Name           Type/Layer Parameters',
';;-------------- ---------- ----------']
    if type == 'BC':
        for one_line in [lid_type, lid_surface, lid_soil, lid_storage, lid_drain]:
            lines.append(one_line)
    elif type == 'RG':
        for one_line in [lid_type, lid_surface, lid_soil, lid_storage]:
            lines.append(one_line)  
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_lid_usage(f, graph, soil_nodes, initsat, name = 'bioret_cell', surf_width = 200):
    """ LID takes up 5% of the subcatchment area. """
    lid_usages = []
    for node in soil_nodes:
        sc_name = 'SC'+str(node).replace(', ','_')
        number = 1
        # Assume 5% of the area is covered by LID
        area = graph.nodes[node].get('node_drainage_area') * 43560 * 0.05
        from_imp = 100
        to_perv = 0
        from_perv = 100
        lid_usage = add_whitespace(sc_name,17,'') + add_whitespace(name,17,'') + \
            add_whitespace(number,8,'') + add_whitespace(area,11,'') + \
            add_whitespace(surf_width,11,'') + add_whitespace(initsat,11,'') + \
            add_whitespace(from_imp,11,'') + add_whitespace(to_perv,11,'') + \
            add_whitespace('*',25,'') + add_whitespace('*',17,'') + \
            add_whitespace(from_perv,10,'') + '\n'
        lid_usages.append(lid_usage)
    lines = ['[LID_USAGE]',
';;Subcatchment   LID Process      Number  Area       Width      InitSat    FromImp    ToPerv     RptFile                  DrainTo          FromPerv  ',
';;-------------- ---------------- ------- ---------- ---------- ---------- ---------- ---------- ------------------------ ---------------- ----------',
lid_usages]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_snowpacks(f):
    lines = ['[SNOWPACKS]',
';;Name           Surface    Parameters',
';;-------------- ---------- ----------',
'NoPlow           PLOWABLE   1          2          3          0.10       0          0.00       0.0   ']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_junctions(f,graph,flood_level,outlet_node):
    junctions = [] ## This is where the nodes information is entered
    for node in graph.nodes():
        if node == outlet_node:
            continue
        name = str(node).replace(', ','_')
        elevation = round(graph.nodes[node].get('elev'),5)
        maxdepth = flood_level
        initdepth = 0
        surdepth = 0
        aponded = 0
        node_junction = add_whitespace(name,17,'') + add_whitespace(elevation, 11, '') + \
                add_whitespace(maxdepth,11,'')+add_whitespace(initdepth,11,'')+ \
                add_whitespace(surdepth,11,'')+add_whitespace(aponded,11,'') + '\n'
        junctions.append(node_junction)
    lines = ['[JUNCTIONS]',
';;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded   ',
';;-------------- ---------- ---------- ---------- ---------- ----------',
junctions]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_outfalls(f,graph,outlet_node):
    outfalls = add_whitespace(outlet_node,17,'')+add_whitespace(graph.nodes[outlet_node].get('elev'),11,'')+ \
    add_whitespace('FREE',11,'')+add_whitespace('',17,'')+ \
    add_whitespace('NO',9,'')+add_whitespace('',16,'')
    lines = ['[OUTFALLS]',
';;Name           Elevation  Type       Stage Data       Gated    Route To        ',
';;-------------- ---------- ---------- ---------------- -------- ----------------',
outfalls]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_conduits(f,graph):
    conduits = []   # This is for pipes
    for edge in graph.edges():
        from_node = str(edge[0]).replace(', ','_')
        to_node = str(edge[1]).replace(', ','_')
        name = from_node+'_'+to_node
        length = round(graph.edges[edge].get('length'),2)
        roughness = graph.edges[edge].get('n')
        edge_cond = add_whitespace(name,17,'') + add_whitespace(from_node,17,'') + \
                add_whitespace(to_node,17,'')+add_whitespace(length,11,'') + \
                add_whitespace(roughness,11,'')+add_whitespace(0,11,'')*4 + '\n'
        conduits.append(edge_cond)
    lines = ['[CONDUITS]',
';;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow   ',
';;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------',
conduits]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_xsections(f,graph):
    xsections = []  # This is for pipes
    for edge in graph.edges():
        from_node = str(edge[0]).replace(', ','_')
        to_node = str(edge[1]).replace(', ','_')
        diameter = graph.edges[edge].get('diam')
        name = from_node+'_'+to_node
        shape = 'CIRCULAR'
        edge_xsect = add_whitespace(name,17,'') + add_whitespace(shape,13,'') + \
            add_whitespace(diameter,17,'') + \
            '0          0          0          1                    \n'
        xsections.append(edge_xsect)
    lines = ['[XSECTIONS]',
';;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert   ',
';;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------',
xsections]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')

def info_timeseries(f,simulation_date,precip_name,precip_duration_hour,total_precip,precip_interval):
    """ precip_interval: minutes for hydrograph """
    interval_precip=total_precip/(precip_duration_hour*60/precip_interval)
    duration_precip = 0
    timeseries = []
    for hour in range(2,round(3+precip_duration_hour,0)):
        for minutes in np.linspace(0,60,num=int(60/precip_interval),endpoint=False,dtype=int):
            timeserie = add_whitespace(precip_name, 17, '')
            date_ts = add_whitespace(simulation_date,11,'')
            time = add_whitespace(str(hour)+':'+'{:02d}'.format(minutes),11,'')
            value = add_whitespace(interval_precip,10,'')
            timeseries.append(timeserie + date_ts + time + value + '\n')
            duration_precip += precip_interval
            if duration_precip >= 120:
                interval_precip=0
    lines = ['[TIMESERIES]',
';;Name           Date       Time       Value     ',
';;-------------- ---------- ---------- ----------',
timeseries]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')

def info_report(f):
    lines = ['[REPORT]',';;Reporting Options','SUBCATCHMENTS ALL',
    'NODES ALL','LINKS ALL']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')

def info_tag(f):
    lines = ['[TAGS]']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')  

def info_map(f):
    lines = ['[MAP]']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n')        

def info_coordinates(f,graph):
    coordinates = []
    for node in graph.nodes():
        name = str(node).replace(', ','_')
        try:
            xcoord, ycoord = graph.nodes[node].get('coordinates')
            node_coordinate = add_whitespace(name, 17, '') + add_whitespace(xcoord, 19, '') + \
                add_whitespace(ycoord, 19, '') + '\n'
            coordinates.append(node_coordinate)
        except TypeError:
            pass
    lines = ['[COORDINATES]',';;Node           X-Coord            Y-Coord',           
';;-------------- ------------------ ------------------',
coordinates]
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n') 

def info_vertices(f):
    lines = ['[VERTICES]',';;Link           X-Coord            Y-Coord',           
';;-------------- ------------------ ------------------']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n') 

def info_polygons(f):
    lines = ['[POLYGONS]',';;Subcatchment   X-Coord            Y-Coord',           
';;-------------- ------------------ ------------------']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n') 

def info_symbols(f):
    lines = ['[SYMBOLS]',';;Gage           X-Coord            Y-Coord',           
';;-------------- ------------------ ------------------']
    for line in lines:
        f.writelines(line)
        f.writelines('\n')
    f.writelines('\n') 

def info_backdrop(f):
    f.writelines('[BACKDROP]\n')
    f.writelines('\n')

def add_whitespace(phrase, white_space_count, value):
    if phrase is not str:
        phrase = str(phrase)
    if value is not str:
        str_value = str(value)
    else: 
        str_value = value   
    if len(phrase) >= white_space_count:
        white_space_to_add = 2
    else: 
        white_space_to_add = white_space_count-len(phrase)
    output = phrase + white_space_to_add*' ' + str_value
    return output

def rep_total_inflow_summary(rep_file_name):
    rep_file=open(rep_file_name,'r')
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "Runoff Quantity Continuity" in line:
            precip_number = line_number + 3
            evap_number = line_number + 4
            infil_number = line_number + 5
        if "Initial LID Storage" in line:
            precip_number +=1
            evap_number +=1
            infil_number +=1
        if "Flow Routing Continuity" in line:
            inflow_summary_number = line_number + 3
        line_number+=1
    rep_file.seek(0)
    lines = rep_file.read().splitlines()
    wet_inflow_vol_MG = float(lines[inflow_summary_number].split()[5]) 
    print(wet_inflow_vol_MG)
    total_precip_MG = float(lines[precip_number].split()[3]) / 3.06888785
    evap_loss_MG = float(lines[evap_number].split()[3]) / 3.06888785
    infil_loss_MG = float(lines[infil_number].split()[3]) / 3.06888785
    return wet_inflow_vol_MG, total_precip_MG, evap_loss_MG, infil_loss_MG
        
def rep_node_flooding_summary(rep_file_name, flood_hour_threshold = 0.3):
    rep_file=open(rep_file_name,'r')
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "  No nodes were flooded." in line:
            node_flooding_summary_number = 0
            end_number = 0
            break
        if " Flooding refers to all water that overflows a node, whether it ponds or not." in line:
            node_flooding_summary_number = line_number + 7
        try: 
            if (line_number > node_flooding_summary_number) and ("*****" in line):
                end_number = line_number - 2
                break
        except UnboundLocalError:
                pass
        line_number+=1
    rep_file.seek(0)

    lines = rep_file.read().splitlines()
    flood_nodes_list = []
    max_flood_nodes = 0
    node_hours_flooded = float(0)
    node_flood_vol_MG = float(0)
    if node_flooding_summary_number > 0:
        for i in range(node_flooding_summary_number, end_number):
            try: 
                if float(lines[i].split()[1]) < flood_hour_threshold:
                    pass
                else:
                    flood_nodes_list.append(int(lines[i].split()[0]))
                    node_hours_flooded += float(lines[i].split()[1])
                    node_flood_vol_MG += float(lines[i].split()[5])
                    max_flood_nodes += 1
            except IndexError or UnboundLocalError:
                pass
    return flood_nodes_list, max_flood_nodes, node_hours_flooded, node_flood_vol_MG

def rep_outflow_summary(rep_file_name):
    rep_file=open(rep_file_name,'r')
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "Outfall Loading Summary" in line:
            outflow_summary_number = line_number + 8
        try: 
            if (line_number > outflow_summary_number) and ("-----" in line):
                end_number = line_number
                break
        except UnboundLocalError:
            pass
        line_number+=1

    rep_file.seek(0)

    lines = rep_file.read().splitlines()
    avg_flow_cfs = 0
    max_flow_cfs = 0
    total_vol_MG = 0
    for i in range(outflow_summary_number, end_number):
        try: 
            avg_flow_cfs += float(lines[i].split()[2])
            max_flow_cfs += float(lines[i].split()[3])
            total_vol_MG += float(lines[i].split()[4])
        except IndexError or UnboundLocalError:
            pass
    return avg_flow_cfs, total_vol_MG

def max_out_flow_readout(output_file_name,outlet_node):
    opt_file = subprocess.run(['swmmtoolbox', 'extract', output_file_name, f'node,{outlet_node},Total_inflow'],capture_output=True, text=True)
    rep_inflow = []
    lines = opt_file.stdout.splitlines()[1:]
    for line in lines:
        rep_inflow.append(float(line.split(",")[1]))
    return max(rep_inflow)

def record_SWMM(input_file_name, net, antecedent_soil_moisture, mean_rainfall_inch, output_df, k, min_diam,changing_diam, mp, swmm_path = my_swmm_path):
    simulation_date = '06/01/2021'
    precip_name = 'hydrograph'
    flood_level = 10         # maximum depth allowable for stormwater to accumulate in MH
    simulation_date = '06/01/2021'
    precip_name = 'hydrograph'
    infiltration='HORTON'
    flowrouting='DYNWAVE' 
    new_file=open(input_file_name,'w')
    make_inp(f=new_file,outlet_node=net.outlet_node,soil_nodes=net.soil_nodes,simulation_date=simulation_date,infiltration=infiltration,pcntimp = 80, flowrouting=flowrouting,precip_name=precip_name,graph=net.gph,flood_level=flood_level,
    antecedent_soil_moisture = antecedent_soil_moisture, mean_rainfall_inch = mean_rainfall_inch)
    new_file.close()
    report_file_name='rep_'+input_file_name
    output_file_name='op_'+input_file_name
    subprocess.run([swmm_path, input_file_name, report_file_name, output_file_name])
    flood_nodes_list, max_flood_nodes, node_hours_flooded, node_flood_vol_MG = rep_node_flooding_summary(report_file_name)
    net.flood_nodes = tuple(flood_nodes_list)
    avg_flow_cfs, total_outflow_vol_MG = rep_outflow_summary(report_file_name)
    max_flow_cfs = max_out_flow_readout(output_file_name,net.outlet_node)
    wet_inflow_vol_MG, total_precip_MG, evap_loss_MG, infil_loss_MG = rep_total_inflow_summary(report_file_name)
    output_df.at[k,'soil_node_distance_list'] = net.calc_node_distance()
    output_df.at[k,'soil_clustering'] = net.calc_node_clustering()
    output_df.at[k,'soil_nodes_count'] = len(net.soil_nodes)/(net.nodes_num)*100
    output_df.at[k,'max_flood_nodes'] = max_flood_nodes
    output_df.at[k,'flood_duration_total_list'] = node_hours_flooded
    output_df.at[k,'total_flooded_vol_MG'] = node_flood_vol_MG
    output_df.at[k,'avg_flow_cfs'] = avg_flow_cfs
    output_df.at[k,'max_flow_cfs'] = max_flow_cfs
    output_df.at[k,'total_outflow_vol_MG'] = total_outflow_vol_MG
    output_df.at[k,'wet_inflow_vol_MG'] = wet_inflow_vol_MG
    output_df.at[k,'total_precip_MG'] = total_precip_MG
    output_df.at[k,'evap_loss_MG'] = evap_loss_MG
    output_df.at[k,'infil_loss_MG'] = infil_loss_MG
    output_df.at[k,'mean_rainfall'] = mean_rainfall_inch
    output_df.at[k,'antecedent_soil'] = antecedent_soil_moisture
    output_df.at[k,'soil_nodes_list'] = net.soil_nodes
    output_df.at[k,'flood_nodes_list'] = net.flood_nodes
    output_df.at[k,'beta'] = net.beta
    output_df.at[k,'changing_diam'] = changing_diam
    output_df.at[k,'min_diam'] = net.min_diam
    output_df.at[k,'max_diam'] = net.max_diam
    output_df.at[k,'pipe_cap'] = net.pipe_cap
    output_df.at[k,'path_diff'] = net.network.path_diff
    output_df.at[k,'path_diff_prime'] = net.network.path_diff_prime
    
    if mp: 
        subprocess.run(['rm',input_file_name, report_file_name, output_file_name])

def main(main_df,antecedent_soil_moisture,mean_rainfall_set,nodes_num,i,beta=0,changing_diam=True, min_diam = 1.5,count=0,soil_nodes=None,
         mp=True, fixing_graph = False, file_name = None, make_cluster = False):
    """
    main_df:        dataframe to append new data to
    antecedent_soil_moisture:       antecedent soil moisture (-), default 0.5
    mean_rainfall_set:  2-hour rainfall inputs
    nodes_num:      number of nodes in the graph
    i:              number of current dataframe row to append to
    beta:           Gibbs distribution parameter
    changing_diam:  determines whether pipes are sized
    min_diam:       minimum pipe diameter (ft), default 1
    count:          number of green infrastructure nodes to generate, default 0
    soil_nodes:     the names of the green infrastructure nodes, default None
    mp:             determines whether to use multiprocessing to parapllelly run simulations
    fixing_graph:   run simulation on one single graph, default False
    file_name:      file path of the graph, default None
    make_cluster:   making green infrastructure nodes in close clusters, default False
    """
    node_drainage_area = 2 
    outlet_level = 1
    outlet_elev = 85               
    outlet_node_drainage_area = node_drainage_area*10e5          # set the river area to very large
    init_level = 0.0
    output_df = pd.DataFrame(data={'soil_nodes_list':[()],'flood_nodes_list':[()]},dtype=object)#, columns=output_columns)
    k = 0

    net = hn.Storm_network(beta = beta, nodes_num = nodes_num, level = init_level, node_drainage_area = node_drainage_area, outlet_level = outlet_level, 
outlet_node_drainage_area = outlet_node_drainage_area, outlet_elev= outlet_elev, count = count, soil_nodes = soil_nodes, changing_diam = changing_diam, 
min_diam = min_diam, fixing_graph = fixing_graph, file_name = file_name, make_cluster = make_cluster)
    soil_node_in_set_check = prod([(node in net.gph.nodes) for node in net.soil_nodes])
    if soil_node_in_set_check == 0:
        print(net.soil_nodes)
        print(net.gph.nodes)

    for mean_rainfall_inch in mean_rainfall_set: 
        input_file_name = f'dataset_{mean_rainfall_inch}-inch_{count}-GI_{antecedent_soil_moisture}-sm_beta-{beta}_{i}_start-{make_cluster}_dist-{net.network.path_diff}.inp'
        record_SWMM(input_file_name=input_file_name, net=net, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch,
        output_df=output_df, k=k, changing_diam=changing_diam,min_diam=min_diam, mp=mp)
        k += 1
    
    if mp:
        pickle_file_name = f'{antecedent_soil_moisture}_{mean_rainfall_inch}-inch_beta-{beta}_{i}_{len(net.soil_nodes)}-GI_start-cluster-{make_cluster}.pickle'
        f = open(pickle_file_name,'wb')
        pickle.dump(output_df, f)
        f.close()
        return pickle_file_name
    else:
        main_df = pd.concat([main_df, output_df], ignore_index=True)
        return main_df

if __name__ == '__main__':
    mean_rainfall_set = [2,3]   # test rainfall
    counts = [10, 20, 30] # test soil moisture counts
    file_name = r'./example/10-grid_beta-0.5_dist-128_ID-0x10b3f48e0>.pickle' # example
    
    ## For regular processing (no parallel processing)
    # all raw inputs and outputs files are retained
    main_df = pd.DataFrame()
    for cnt in counts:
        main_df = main(main_df= None, antecedent_soil_moisture=0.5, mean_rainfall_set=mean_rainfall_set, count = cnt,
            beta = 0, i= 1,nodes_num=100, mp=False, file_name=file_name)

    ## For parallel processing 
    # all raw inputs and outputs files are deleted. Outputs are saved
    # in a Pickle file. 
    all_reports = []
    for cnt in counts:
        report_pickle_name = main(main_df= None, antecedent_soil_moisture=0.5, mean_rainfall_set=mean_rainfall_set, count = cnt,
            beta = 0, i= 1,nodes_num=100, mp=True, file_name=file_name)
        all_reports.append(report_pickle_name)
    
    f = open(report_pickle_name, 'rb')
    print(pickle.load(f).head())
