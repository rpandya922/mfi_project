#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "kinova_testing");
    ros::NodeHandle n;

    // try publishing a joint pose command
    // ros::Publisher joint_pub = n.advertise<std_msgs::Float64MultiArray>("/siemens_demo/joint_cmd", 1000);
    ros::Publisher pose_pub = n.advertise<geometry_msgs::PoseStamped>("/kinova_demo/pose_cmd", 1);
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok()) {
        // std_msgs::Float64MultiArray j_state;
        // float joint_arr[7] = {30.05087883047087, 0.40245806974509, 179.55441712788306, 273.4147947399282, 0.16227331516802898, 286.97403163872417, 110.58438965992093};
        // for (size_t i=0; i<7; ++i) {
        //     j_state.data.push_back(joint_arr[i]);
        // }

        // joint_pub.publish(j_state);
        geometry_msgs::PoseStamped pose_des;
        pose_des.pose.position.x = 0.496;
        pose_des.pose.position.y = 0.138;
        pose_des.pose.position.z = 0.5;
        pose_des.pose.orientation.x = 0.0;
        pose_des.pose.orientation.y = 0.0;
        pose_des.pose.orientation.z = 0.0;
        pose_des.pose.orientation.w = 1.0;

        pose_pub.publish(pose_des);

        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
}
// 0.496 0.138 0.128 -180 0 90