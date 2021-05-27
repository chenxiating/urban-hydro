import pandas as pd
rep_file = open(r'./SWMM_20210524-1415/dataset_5.0-inch_100-nodes_soil_moisture-0.2_8_rep.inp','r')
# a = rep_file.read().splitlines()
# print(a)

def rep_node_flooding_summary(rep_file):
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if " Flooding refers to all water that overflows a node, whether it ponds or not." in line:
            # print(line_number)
            node_flooding_summary_number = line_number + 7
        try: 
            if (line_number > node_flooding_summary_number) and ("*****" in line):
                end_number = line_number
                print(end_number)
                break
        except NameError:
            pass
        line_number+=1

    rep_file.seek(0)

    lines = rep_file.read().splitlines()
    max_flood_nodes = 0
    node_hours_flooded = 0
    node_flood_vol_MG = 0
    for i in range(node_flooding_summary_number, end_number):
        try: 
            max_flood_nodes += 1
            node_hours_flooded += (lines[i].split()[1])
            node_flood_vol_MG += (lines[i].split()[4])
        except IndexError:
            pass
    return max_flood_nodes, node_hours_flooded, node_flood_vol_MG

def rep_outflow_sumary(rep_file):
    rep_file.seek(0)
    line_number = 0
    for line in rep_file.read().split("\n"):
        if "Outfall Loading Summary" in line:
            # print(line_number)
            outflow_summary_number = line_number + 8
        try: 
            if (line_number > outflow_summary_number) and ("-----" in line):
                end_number = line_number
                print(end_number)
                break
        except NameError:
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
            max_flow_cfs += (lines[i].split()[3])
            total_vol_MG += (lines[i].split()[4])
        except IndexError:
            pass
    return max_flow_cfs, total_vol_MG

def main():
    rep_node_flooding_summary(rep_file)
    rep_outflow_sumary(rep_file)