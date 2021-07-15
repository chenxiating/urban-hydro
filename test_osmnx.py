import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# city = ox.geocode_to_gdf("St Paul, Minnesota, USA")
# city_proj = ox.project_gdf(city)
# ax = city_proj.plot(fc='gray',ec='none')
# _ = ax.axis('off')
# location_point = (44.9,-93.1)
# # G = ox.graph_from_bbox(city_proj.bbox_north[0], city_proj.bbox_south[0], city_proj.bbox_east[0], 
# # city_proj.bbox_west[0], network_type = "drive_service")
# G = ox.graph_from_point(location_point,dist=750,dist_type = "bbox",network_type="all")
# print(city_proj.bbox_north[0], city_proj.bbox_south[0], city_proj.bbox_east[0], city_proj.bbox_west[0])
# fig, ax = ox.plot_graph(G,node_color="r")
# plt.show()

G=nx.read_shp(r'/Users/xchen/python_scripts/urban_stormwater_analysis/urban-hydro/shp/STP_STRMPIPE.shp') 
pos = {k: v for k,v in enumerate(G.nodes())}
X=nx.Graph() #Empty graph
X.add_nodes_from(pos.keys()) #Add nodes preserving coordinates
l=[set(x) for x in G.edges()] #To speed things up in case of large objects
edg=[tuple(k for k,v in pos.items() if v in sl) for sl in l] #Map the G.edges start and endpoints onto pos
nx.draw_networkx_nodes(X,pos,node_size=10,node_color='r')
X.add_edges_from(edg)
nx.draw_networkx_edges(X,pos)
xlim = [v[0] for v in pos.values()]
ylim = [v[1] for v in pos.values()]
# plt.xlim(min(xlim), (max(xlim)-min(xlim))/3+min(xlim)) #This changes and is problem specific
# plt.ylim(min(ylim), max(ylim)) #This changes and is problem specific
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.xticks(np.linspace(min(xlim), max(xlim),num=5))
plt.yticks(np.linspace(min(ylim), max(ylim),num=5))
plt.title('From shapefiles to NetworkX')
print(min(ylim), max(ylim),min(xlim), max(xlim))
plt.show()
