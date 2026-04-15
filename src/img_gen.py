import cv2
import numpy as np

# canvas
size = 600
img = np.zeros((size, size, 3), dtype=np.uint8)
center_x = size // 2
center_y = size // 2
center = (size // 2, size // 2)

# base motif (a small circle cluster)
# def draw_motif(canvas, pos, scale=1.0):
#     x, y = pos
#     r = int(6 * scale)
#     cv2.circle(canvas, (x, y), r, (255, 0, 0), 1)
    # cv2.circle(canvas, (x + int(10*scale), y), r, (0, 0, 255), 1)
    # cv2.circle(canvas, (x, y + int(10*scale)), r, (0, 0, 255), 1)

def draw_motif(canvas, pos, angle=45, scale=1.0):
    x, y = pos

    x = int(x)
    y = int(y)

    w = int(10 * scale)
    h = int(18 * scale)

    rect = ((float(x), float(y)), (float(w), float(h)), float(angle))

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.polylines(canvas, [box], True, (255, 0, 0), 1)

def draw_circle_motif(canvas, pos, scale=1.0, color=(255, 255, 255), thickness=1):
    x, y = pos
    x = int(x)
    y = int(y)

    radius = max(1, int(6 * scale))

    cv2.circle(canvas, (x, y), radius, color, thickness)

# create radial center
num = 40
radius = 80

# coordinate grid in math space
xs = np.linspace(-1, 1, size)
ys = np.linspace(-1, 1, size)
X, Y = np.meshgrid(xs, ys)
eps = 0.03
F = X**2 + Y**2 + 1 - 2 * (np.abs(X) + np.abs(Y))
boundary = (np.abs(F) < eps).astype(np.uint8) * 255

contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)

# rectangle motif
x1, y1 = 149, 149
x2, y2 = 449, 449

# top edge
top_x = np.arange(x1, x2 + 1)
top_y = np.full_like(top_x, y1)

# bottom edge
bot_x = np.arange(x1, x2 + 1)
bot_y = np.full_like(bot_x, y2)

# left edge
left_y = np.arange(y1, y2 + 1)
left_x = np.full_like(left_y, x1)

# right edge
right_y = np.arange(y1, y2 + 1)
right_x = np.full_like(right_y, x2)

# combine everything
xs = np.concatenate([top_x, bot_x, left_x, right_x])
ys = np.concatenate([top_y, bot_y, left_y, right_y])

points = np.stack([xs, ys], axis=1)

theta = np.pi / 4  # 45 degrees

cx = (x1 + x2) / 2
cy = (y1 + y2) / 2

# convert to float for precision
pts = points.astype(np.float32)

# shift to origin
pts[:, 0] -= cx
pts[:, 1] -= cy

# rotation
cos_t = np.cos(theta)
sin_t = np.sin(theta)

x_new = pts[:, 0] * cos_t - pts[:, 1] * sin_t
y_new = pts[:, 0] * sin_t + pts[:, 1] * cos_t

# shift back
x_new += cx
y_new += cy

# convert back to int pixel coords
rotated_points = np.stack([x_new, y_new], axis=1).astype(np.int32)

step = 12

for i in range(0, len(points), step):
        px, py = points[i]
        draw_motif(img, (px, py), scale=0.5)
        draw_motif(img, (px, py), angle=-45, scale=0.5)
        
for i in range(0, len(rotated_points), step):
        px, py = rotated_points[i]
        draw_motif(img, (px, py), scale=0.5)
        draw_motif(img, (px, py), angle=-45, scale=0.5)

        
# optional: thin out the points so motifs are not insanely dense
ys_idx, xs_idx = np.where(boundary)

# for i in range(num):
#     angle = 2 * np.pi * i / num
#     x = int(center[0] + radius * np.cos(angle))
#     y = int(center[1] + radius * np.sin(angle))
#     draw_motif(img, (x, y), scale=1.0)

step = 8
for i in range(0, len(contour), step):
    px, py = contour[i][0]
    draw_circle_motif(img, (px, py), scale=0.5, color=(255, 0, 0), thickness=1)
    
# create one arm
arm = np.zeros_like(img)

for i in range(8):
    x = center[0]
    y = center[1] - 120 + i * 20
    draw_motif(arm, (x, y), scale=0.8)

# rotate arm 4 times
for i in range(4):
    M = cv2.getRotationMatrix2D(center, i * 90, 1.0)
    rotated = cv2.warpAffine(arm, M, (size, size))
    img = cv2.add(img, rotated)

# optional blur
# img = cv2.GaussianBlur(img, (3, 3), 0)

while True:
    cv2.imshow("pattern", img)
    
    key = cv2.waitKey(1) & 0xFF  # wait 1ms and get key

    if key == ord('q'):
        break

cv2.imwrite("assets/bg.png", img)
cv2.destroyAllWindows()
