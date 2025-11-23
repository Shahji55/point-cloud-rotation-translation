# point-cloud-rotation-translation
Predict rotation and translation parameters between 2D point clouds using a NN.

1) Generate synthetic data:

    - X = source
    - Y = target
    - R = [cos(angle), -sin(angle)]
          [sin(angle), cos(angle)]
    - Target Y is determined using following transformation: Y=R.X+t
      Where R = rotation matrix and t = translation vector

2) Neural Network:

    - Input: flattened X + flattened Y
    - Output: 4 values (cos(theta), sin(theta), tx, ty)
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Adam

3) Baseline:
    a) Rotation:
        - cX and cY are the centroids of the source and target points.
        - X[0] - cX = vector from the centroid to the first point in X.
        - Y[0] - cY = vector from the centroid to the first point in Y.
        - np.arctan2(x, y) = angle of vector (x,y) relative to the x-axis.
        - base_angle = angle of target - angle of source

    b) Translation:
        - base_R = [cos(base_angle), -sin(base_angle)]
                 [sin(base_angle), cos(base_angle)]

          Where base_R is the rotation matrix made using base_angle

        - base_t = cY - base_R.cX

          Where base_t is the translation needed to move the rotated centroid of X to the centroid of Y.

4) Libraries used: numpy, matplotlib, pytorch