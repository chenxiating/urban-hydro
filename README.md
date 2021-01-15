# urban-hydro 
Jan 12, 2021
- Tested transparency, added fig_text in test_graph.py
- Cleaned up flood nodes and flood time calculations in efficiency_test_manning.py
- rainfall_nodes_func now outputs runoff, h_new instead of s, h_new, edge_h
- Added max highest runoff as a performance criteria. It doesn't quite work, because all the water levels remain the same.
This is because the runoff currently is only affecting other conveyance activities. The runoff does not go into downstream
soil. 
- Turned off attr_array_func. It's unnecessary.

Jan 9, 2021
Added test_box_plot.py to create box plots, but I'm not very happy with how they look. 

Jan 5, 2021
create_network function now can calculate diameter changes based on degrees away from the outlet.
