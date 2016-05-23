from ete3 import PhyloTree, Tree, TreeStyle
from ete3 import add_face_to_node, TextFace, AttrFace, SequenceFace

nw = '(0:0.01, 1:0.2, (2:0.3, (3:0.3, 4:0.02)5:0.05)6:0.15)7;'
fa = """
>0
A
>1
C
>2
C
>3
C
>4
G
"""

t = PhyloTree(nw, alignment=fa, alg_format='fasta', format=1)
ts = TreeStyle()
ts.show_branch_length = False
ts.show_leaf_name = False
ts.draw_guiding_lines = True
ts.draw_aligned_faces_as_table = True
ts.show_scale = False

def my_layout(node):
    #
    # add names to all nodes (not just to leaf nodes)
    # ete3/test/test_treeview/face_rotation.py
    F = TextFace(node.name, tight_text=True)
    add_face_to_node(F, node, column=0, position="branch-right")
    #
    # add branch lengths
    # ete3/treeview/qt4_render.py
    if not node.is_root():
        bl_face = AttrFace("dist", fsize=8, ftype="Arial",
                fgcolor="black", formatter="%0.3g")
        #
        # This is a failed attempt to center the branch length text on the branch.
        #a = 1 # 0 left, 1 center, 2 right
        #bl_face.hz_align = a
        #bl_face.vt_align = a
        #add_face_to_node(bl_face, node, column=0, aligned=True, position="branch-top")
        add_face_to_node(bl_face, node, column=0, position="branch-top")
    #
    # I guess we also have to explicitly add the alignment column
    # if we are overriding the layout function.
    # ete3/treeview/layouts.py : phylogeny(node)
    if hasattr(node, 'sequence'):
        seq_face = SequenceFace(node.sequence, seqtype='nt', fsize=13)
        seq_face.margin_left = 4
        add_face_to_node(seq_face, node, column=1, aligned=True)


ts.layout_fn = my_layout

t.render('out.svg', tree_style=ts)
print(t.get_ascii(show_internal=True))
