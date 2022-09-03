from aluneth.mathematics.geometry.geoclidean.euclidean_primitives import *

img_path = "/Users/melkor/miniforge3/envs/Melkor/lib/python3.9/site-packages/aluneth/mathematics/geometry/geoclidean/images"


concept = [
    'l1 = line(p1(), p2())',
    'c1* = circle(p1(), p2())',
    'c2* = circle(p2(), p1())',
    'l2 = line(p1(), p3(c1, c2))',
    'l3 = line(p2(), p3()))'
]



for i in range(3):
    generate_concept(concept, mark_points=False, show_plots=True,path ="output")


concept = [
    'c1 = circle(p1(), p2())',
    'c2 = circle(p3(c1), p4())',
    'l3 = line(p5(c1), p6(c1, c2))'
]

for i in range(3):
    generate_concept(concept, mark_points=False, show_plots=True,path = "output")