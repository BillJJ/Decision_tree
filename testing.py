import pygraphviz

g = pygraphviz.AGraph(directed=True)
g.node_attr['shape']='box'

g.add_node(1)
g.add_node(2)
g.add_edge(1,2)

g.get_node(2).attr['label'] = 10

g.layout('dot')
g.draw('tree_visual.png')

a = None
if a:
    print('sudgh')