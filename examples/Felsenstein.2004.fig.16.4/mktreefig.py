from ete3 import Tree, TreeStyle
from ete3 import add_face_to_node, TextFace, AttrFace

nw = '(0:0.01, 1:0.2, (2:0.3, (3:0.3, 4:0.02)5:0.05)6:0.15);'
t = Tree(nw, format=1)
ts = TreeStyle()
ts.show_branch_length = False
ts.show_leaf_name = False
#ts.show_internal = True
#t.render('out.svg', tree_style=ts, show_internal=True)

# ete3/treeview/qt4_render.py
"""
na_face = faces.AttrFace("name", fsize=10, ftype="Arial", fgcolor="black")
for n in root_node.traverse():
    faces.add_face_to_node(na_face, n, 0, position="branch-right")
    update_node_faces(n, n2f, img)
"""

def my_layout(node):
    #
    # add names to all nodes (not just to leaf nodes)
    # ete3/test/test_treeview/face_rotation.py
    F = TextFace(node.name, tight_text=True)
    add_face_to_node(F, node, column=0, position="branch-right")
    #
    # add branch lengths
    # ete3/treeview/qt4_render.py
    bl_face = AttrFace(
            "dist", fsize=8, ftype="Arial", fgcolor="black", formatter="%0.3g")
    #
    # This is a failed attempt to center the branch length text on the branch.
    #a = 1 # 0 left, 1 center, 2 right
    #bl_face.hz_align = a
    #bl_face.vt_align = a
    #
    add_face_to_node(bl_face, node, column=0, position="branch-top")

ts.layout_fn = my_layout

t.render('out.svg', tree_style=ts)
print(t.get_ascii(show_internal=True))
