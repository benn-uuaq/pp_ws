from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.227.134",
        description="Elite Robot CS612 controller IP address",
    )

    robot_control_node = Node(
        package="elite_robot_controller",
        executable="robot_control_node",
        name="robot_control_node",
        output="screen",
        parameters=[{"robot_ip": LaunchConfiguration("robot_ip")}],
    )

    return LaunchDescription([robot_ip_arg, robot_control_node])
