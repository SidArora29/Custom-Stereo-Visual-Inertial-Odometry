import gtsam
from gtsam.symbol_shorthand import X
import numpy as np

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), prior_noise))

odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
relative_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0))

for i in range(4):
    graph.add(gtsam.BetweenFactorPose3(X(i), X(i+1), relative_pose, odom_noise))

# Deliberately bad initial guess
for i in range(5):
    initial.insert(X(i), gtsam.Pose3())

result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
result.print("Final result:\n")