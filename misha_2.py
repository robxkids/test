from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2

class LfChallengeNoCvTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)


    def solve(self):
        env = self.generated_task['env']
        while True:
            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            distance_to_road_center = lane_pose.dist
            angle_from_straight_in_rads = lane_pose.angle_rad
            # Требуется по положению робота в полосе определить линейную и угловые скорости
            speed = 0.2
            steering = 0
            print("distance", distance_to_road_center)
            print("angle", angle_from_straight_in_roads)
            k_p = 10
            k_d = 1
            steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads
            env.step([speed, steering])
            env.render()
