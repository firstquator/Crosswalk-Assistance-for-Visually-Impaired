import numpy as np

pt1 = (100, 200)
pt2 = (300, 400)

dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]

print(np.degrees(np.arctan2(dy, dx)))
