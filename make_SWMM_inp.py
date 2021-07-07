import hydro_network as hn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import datetime as date
import datetime
import pickle
import subprocess
import os

def make_inp(f,soil_nodes,simulation_date,infiltration,flowrouting,precip_name,graph,flood_level,antecedent_soil_moisture,mean_rainfall_inch):
    info_header(f=f,simulation_date=simulation_date,infiltration=infiltration,flowrouting=flowrouting)
    info_evaporations(f=f,et_rate=5)
    info_temperature(f=f)
    info_raingages(f=f,precip_name=precip_name,raingage=1)
    info_subcatchments(f=f,graph=graph,raingage=1,soil_nodes=soil_nodes)
    info_subareas(f=f,graph=graph)
    info_infiltration(f=f,infiltration=infiltration,graph=graph)
    info_lid_controls(f=f)
    info_lid_usage(f=f,graph=graph,soil_nodes=soil_nodes,initsat=antecedent_soil_moisture)
    info_snowpacks(f=f)
    info_junctions(f=f,graph=graph,flood_level=flood_level)
    info_outfalls(f=f,graph=graph)
    info_conduits(f=f,graph=graph)
    info_xsections(f=f,graph=graph)
    info_timeseries(f=f,simulation_date=simulation_date,precip_name=precip_name,two_hour_precip=mean_rainfall_inch,precip_interval=15)
    info_report(f=f)
    info_tag(f=f)
    info_map(f=f) 
    info_coordinates(f=f)
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
    'END_TIME             00:00:00',
    'SWEEP_START          01/01',
    'SWEEP_END            12/31',
    'DRY_DAYS             0',
    'REPORT_STEP          00:15:00',
    'WET_STEP             00:05:00',
    'DRY_STEP             00:15:00',
    'ROUTING_STEP         0:00:30 ',
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

def info_subcatchments(f,graph,raingage,soil_nodes):
    subcatchments = [] ## This is where the node information is entered.
    for node in graph.nodes():
        if node == 0:
            continue
        name = 'SC'+str(node)
        raingage = raingage
        outlet = str(node)
        totalarea = graph.nodes[node].get('node_drainage_area')
        pcntimp = 0 if node in soil_nodes else 100
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

def info_subareas(f,graph):
    subareas = []
    for node in graph.nodes():
        if node == 0:
            continue
        subcatchment = 'SC'+str(node)
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
    
def info_infiltration(f,infiltration,graph):
    if infiltration == 'HORTON':
        infiltration = []
        for node in graph.nodes():
            if node == 0:
                continue
            subcatchment = 'SC'+str(node)
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
            if node == 0:
                continue
            subcatchment = 'SC'+str(node)
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

def info_lid_controls(f, name = 'bioret_cell', type = 'BC', surf_height = 8, surf_veg = 0.1, surf_n = 0.1, 
surf_slope = 1.0, soil_height = 18, soil_n = 0.43, soil_fc = 0.4, soil_wp = 0.18,
soil_k = 1.3, soil_kslope = 40, soil_psi = 3.5, stor_height = 0, stor_voidratio = 0.25,
stor_seepage = 1.5, stor_clog = 0, drain_coef = 0, drain_exp = 0.5, drain_offset = 6,
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
    lid_usages = []
    for node in soil_nodes:
        sc_name = 'SC'+str(node)
        number = 1
        area = graph.nodes[node].get('node_drainage_area') * 43560
        from_imp = 100
        to_perv = 100
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

def info_junctions(f,graph,flood_level):
    junctions = [] ## This is where the nodes information is entered
    for node in graph.nodes():
        if node == 0:
            continue
        name = str(node)
        elevation = round(graph.nodes[node].get('elev'),5)
        maxdepth = flood_level
        initdepth = 0
        surdepth = flood_level
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

def info_outfalls(f,graph):
    outfalls = add_whitespace(0,17,'')+add_whitespace(graph.nodes[0].get('elev'),11,'')+ \
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
        from_node = edge[0]
        to_node = edge[1]
        name = str(from_node)+'_'+str(to_node)
        length = graph.edges[edge].get('length')
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
        from_node = edge[0]
        to_node = edge[1]
        diameter = graph.edges[edge].get('diam')
        name = str(from_node)+'_'+str(to_node)
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

def info_timeseries(f,simulation_date,precip_name,two_hour_precip,precip_interval):
    """ precip_interval: minutes for hydrograph """
    interval_precip=two_hour_precip/(2*60/precip_interval)
    duration_precip = 0
    timeseries = []
    for hour in range(2,5):
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

def info_coordinates(f):
    lines = ['[COORDINATES]',';;Node           X-Coord            Y-Coord',           
';;-------------- ------------------ ------------------']
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

def rep_node_flooding_summary(rep_file_name):
    rep_file=open(rep_file_name,'r')
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "  No nodes were flooded." in line:
            node_flooding_summary_number = 0
            end_number = 0
            break
        if " Flooding refers to all water that overflows a node, whether it ponds or not." in line:
            # print(line_number)
            node_flooding_summary_number = line_number + 7
        # if "  Node Flooding Summary" in line:
        #     # print(line_number)
        #     node_flooding_summary_number = line_number + 10
        try: 
            if (line_number > node_flooding_summary_number) and ("*****" in line):
                end_number = line_number
                # print(end_number)
                break
        except UnboundLocalError:
                # node_flooding_summary_number = 0
                pass
        line_number+=1
    print('node_flooding_summary_number', node_flooding_summary_number,'end_number',end_number)
    rep_file.seek(0)

    lines = rep_file.read().splitlines()
    max_flood_nodes = 0
    node_hours_flooded = float(0)
    node_flood_vol_MG = float(0)
    if node_flooding_summary_number > 0:
        for i in range(node_flooding_summary_number, end_number):
            try: 
                max_flood_nodes += 1
                # print('max_flood_nodes',max_flood_nodes)
                # print(lines[i])
                node_hours_flooded += float(lines[i].split()[1])
                # print('node_hours_flooded',node_hours_flooded)
                node_flood_vol_MG += float(lines[i].split()[5])
                # print('node_flood_vol_MG',node_flood_vol_MG)
            except IndexError or UnboundLocalError:
                pass
    return max_flood_nodes, node_hours_flooded, node_flood_vol_MG

def rep_outflow_sumary(rep_file_name):
    rep_file=open(rep_file_name,'r')
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "Outfall Loading Summary" in line:
            # print(line_number)
            outflow_summary_number = line_number + 8
        try: 
            if (line_number > outflow_summary_number) and ("-----" in line):
                end_number = line_number
                # print(end_number)
                break
        except UnboundLocalError:
            # outflow_summary_number = 0 
            pass
        line_number+=1

    rep_file.seek(0)

    lines = rep_file.read().splitlines()
    avg_flow_cfs = 0
    max_flow_cfs = 0
    total_vol_MG = 0
    for i in range(outflow_summary_number, end_number):
        try: 
            # avg_flow_cfs += (lines[i].split()[2])
            max_flow_cfs += float(lines[i].split()[3])
            # print('max_flow_cfs',max_flow_cfs)
            total_vol_MG += float(lines[i].split()[4])
            # print('total_outflow_vol_MG', total_vol_MG)
        except IndexError or UnboundLocalError:
            pass
    return max_flow_cfs, total_vol_MG

def main(main_df,antecedent_soil_moisture,mean_rainfall_inch,nodes_num,i):
    simulation_date = '06/01/2021'
    precip_name = 'hydrograph'
    node_drainage_area = 2           # acres
    outlet_level = {0: 1} 
    outlet_elev = {0: 85}                 
    outlet_node_drainage_area = {0: node_drainage_area*10e5}             # set the river area to very large
    soil_depth = 6
    init_level = 0.0
    flood_level = 10
    report_file_list = []

    soil_nodes_combo, soil_nodes_combo_count = hn.random_sample_soil_nodes(range_min = 0, range_max = 5, range_count = 5, nodes_num = nodes_num)
    output_df = pd.DataFrame()#, columns=output_columns)
    output_df.loc[:,'soil_nodes_list'] = soil_nodes_combo
    k = 0

    for soil_nodes in soil_nodes_combo:
        # kernel = lambda x: np.exp(-mu)*mu**x/factorial(x)
        H = hn.create_networks(g_type = 'gn', nodes_num = nodes_num, level = init_level, diam = 1, node_drainage_area = node_drainage_area, outlet_level = outlet_level, 
    outlet_node_drainage_area = outlet_node_drainage_area, outlet_elev= outlet_elev, kernel=None,seed = None)
        soil_node_degree = hn.calc_soil_node_degree(H,soil_nodes)
        soil_node_elev = hn.calc_soil_node_elev(H,soil_nodes)
        input_file_name = 'dataset_'+str(round(mean_rainfall_inch,1))+'-inch_'+str(len(soil_nodes))+'-soil-nodes_'+'soil_moisture-'+str(round(antecedent_soil_moisture,1))+'_'+str(i)+'_'+str(k)+'.inp'
        print('Permeable nodes count: ', len(soil_nodes), soil_nodes, 'Rainfall intensity: ', mean_rainfall_inch, 'Soil moisture: ', antecedent_soil_moisture)
        infiltration='HORTON'
        flowrouting='KINWAVE'
        new_file=open(input_file_name,'w')
        make_inp(f=new_file,soil_nodes=soil_nodes,simulation_date=simulation_date,infiltration=infiltration,flowrouting=flowrouting,precip_name=precip_name,graph=H,flood_level=flood_level,antecedent_soil_moisture = antecedent_soil_moisture, mean_rainfall_inch = mean_rainfall_inch)
        new_file.close()
        report_file_name='rep_'+input_file_name
        output_file_name='op_'+input_file_name
        report_file_list.append(report_file_name)
        # subprocess.run(['/Users/xchen/Applications/swmm5/build/runswmm5',input_file_name, report_file_name, output_file_name])
        subprocess.run(['../../swmm51015_engine/build/runswmm5',input_file_name, report_file_name, output_file_name])
        
        max_flood_nodes, node_hours_flooded, node_flood_vol_MG = rep_node_flooding_summary(report_file_name)
        max_flow_cfs, total_outflow_vol_MG = rep_outflow_sumary(report_file_name)
        output_df.loc[k,'soil_node_degree_list'] = soil_node_degree
        output_df.loc[k,'soil_node_elev_list'] = soil_node_elev
        output_df.loc[k,'soil_nodes_count'] = len(soil_nodes)/nodes_num*100
        output_df.loc[k,'max_flood_nodes'] = max_flood_nodes
        output_df.loc[k,'flood_duration_total_list'] = node_hours_flooded
        output_df.loc[k,'total_flooded_vol_MG'] = node_flood_vol_MG
        output_df.loc[k,'max_flow_cfs'] = max_flow_cfs
        output_df.loc[k,'total_outflow_vol_MG'] = total_outflow_vol_MG
        output_df.loc[k,'mean_rainfall'] = mean_rainfall_inch
        output_df.loc[k,'antecedent_soil'] = antecedent_soil_moisture
        # 'mean_flood_nodes_TI'
        # 'mean_var_path_length', 'mean_disp_kg', 'mean_disp_g'
        subprocess.run(['rm',input_file_name, report_file_name, output_file_name])
        k += 1
    print(output_df)

    main_df = pd.concat([main_df, output_df], ignore_index=True)
    return main_df
        # main_df = pd.concat([main_df, output_df], ignore_index=True)        
        # f = open(datafile_name,'wb')
        # pickle.dump(main_df, f)
        # f.close()

if __name__ == '__main__':
    soil_moisture_list = np.linspace(0.0, 1.0, 4)
    mean_rainfall_set = np.linspace(13, 3, 4, endpoint=False)
    nodes_num = 20

    today = date.datetime.today()
    dt_str = today.strftime("%Y%m%d-%H%M")
    folder_name='./SWMM_'+dt_str
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass    
    os.chdir(folder_name)
    datafile_name = dt_str + '_full_dataset_'+str(nodes_num)+'-nodes'+'.pickle'

    main_df = pd.DataFrame()
    for i in range(10):
        for antecedent_soil_moisture in soil_moisture_list:
            for mean_rainfall_inch in mean_rainfall_set:
                main_df = main(main_df, antecedent_soil_moisture=antecedent_soil_moisture, mean_rainfall_inch=mean_rainfall_inch, nodes_num=nodes_num,i=i)
    f = open(datafile_name,'wb')
    pickle.dump(main_df, f)
    f.close()
    print(main_df)
    print(datafile_name)
