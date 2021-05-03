# Detect Square from Static Background
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

# Function to find the intersection b/w two lines
def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1*math.pi/180), np.sin(theta1*math.pi/180)], [np.cos(theta2*math.pi/180), np.sin(theta2*math.pi/180)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

# Image Information
def main():

    # WIDTH and HEIGHT to Resize
    WIDTH = 300
    HEIGHT = 400

    # Load background image, resize, convert to grayscale, blur
    background = cv2.imread('ex_bg_4.jpg')
    background = cv2.resize(background,(WIDTH, HEIGHT))
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (5,5), 0)

    # Load foreground image, resize, convert to grayscale, blur
    foreground = cv2.imread('ex_fg_4.jpg')
    foreground = cv2.resize(foreground, (WIDTH, HEIGHT))
    per_tf_input = foreground.copy()
    fg_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    fg_gray = cv2.GaussianBlur(fg_gray, (5,5), 0)

    # Take the difference between the images and blur
    difference = cv2.absdiff(bg_gray, fg_gray)
    difference = cv2.GaussianBlur(difference, (5,5), 0)

    # Canny Edge Detector for edges
    edges = cv2.Canny(difference, 100, 200)

    # HOUGH TRANSFORM

    # Get number of rows and cols in images
    rows = edges.shape[0]
    cols = edges.shape[1]

    # Get maximum index values for accumulator matrix
    max_r = int(math.sqrt(math.pow(rows,2) + math.pow(cols,2)))
    max_theta = 180
    H = np.zeros((max_r, max_theta+1))

    # Get edge points to use in Hough Transform
    edge_points = []
    for r in range(rows):
        for c in range(cols):
            if edges[r,c] > 0:
                edge_points.append([r,c])

    # Loop through edge_points and vote on values of r and theta
    THETA_OFFSET = 5                                                        # hyperparameter
    for e in edge_points:
        x = e[1]
        y = e[0]
        for theta in range(0, 181, THETA_OFFSET):
            angle_rad = math.radians(theta)
            r = int(x * math.cos(angle_rad) + y * math.sin(angle_rad))
            H[r][theta] = H[r][theta] + 1

    # Voting
    ACCUMULATOR_THRESHOLD = 30                                              # hyperparameter
    r_vals = []
    theta_vals = []
    for i in range(max_r):
        for j in range(max_theta + 1):
            if(H[i][j] > ACCUMULATOR_THRESHOLD):
                r_vals.append(i)
                theta_vals.append(j)

    # Draw Lines Based on Hough Transform Values
    img = foreground.copy()
    pooled = img.copy()
    kmeans_img = img.copy()
    horz_lines = []
    vert_lines = []
    y_0_init = 0
    y_1_init = rows - 1
    count = 0
    for top_r in r_vals:
        top_theta = theta_vals[count]
        try:
            x0 = int((top_r - y_0_init * math.sin(math.radians(top_theta)))/ math.cos(math.radians(top_theta)))
            x1 = int((top_r - y_1_init * math.sin(math.radians(top_theta))) / math.cos(math.radians(top_theta)))
        except ZeroDivisionError:
            count = count + 1
            continue
        try:
            if top_theta >= 0 and top_theta < 45 or top_theta <= 180 and top_theta > 135:
                cv2.line(img, (x0, y_0_init), (x1, y_1_init), (0, 255, 0), 2)
                vert_lines.append([top_r, top_theta])
            else:
                cv2.line(img, (x0, y_0_init), (x1, y_1_init), (0, 255, 255), 2)
                horz_lines.append([top_r, top_theta])
        except OverflowError:
            count = count + 1
            continue
        count = count + 1

    # Get the Points of Intersection
    intersections = []
    for h in horz_lines:
        for v in vert_lines:
            inter = intersection(h,v)
            inter = inter[0]
            if(inter[0] > 0 and inter[0] < cols and inter[1] > 0 and inter[1] < rows):
                pooled[inter[1],inter[0]] = (255, 255, 0)
                intersections.append(inter)

    # Perform K-Means Clustering
    X = np.array(intersections)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    corners = kmeans.cluster_centers_
    for c in corners:
        kmeans_img = cv2.circle(kmeans_img, (int(c[0]), int(c[1])), 10, (0, 0, 255), -1)            # Draw circle at corners
    print('Corners Detected: ')
    print(corners)

    # Compute Perspective Transformation
    pts1 = np.float32(corners)
    pts2 = np.float32([[400, 400],[0, 0],[0, 400],[400, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(per_tf_input, M, (400, 400))

    # Make plots showing square detection results
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    plt.title("Foreground")
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title("Background")
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
    plt.title("Background Subtracted")
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.title("Edge Detection")
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Line Detection")
    plt.subplot(2,4,6)
    plt.imshow(cv2.cvtColor(pooled, cv2.COLOR_BGR2RGB))
    plt.title("Intersection Points")
    plt.subplot(2,4,7)
    plt.imshow(cv2.cvtColor(kmeans_img,cv2.COLOR_BGR2RGB))
    plt.title("K-Means Cluster Centers")
    plt.subplot(2,4,8)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title("Perspective Transform")
    plt.suptitle("Square Detection Results")
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    plt.savefig('square_detection_results.png')
    plt.show()
    cv2.waitKey(0)

if __name__ == '__main__':
    main()