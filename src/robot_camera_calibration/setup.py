import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_camera_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pp_ws',
    maintainer_email='hoonyoung323@gmail.com',
    description='Robot-camera (eye-to-hand) calibration for piece picking (Helios, ArUco)',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motion_tool = robot_camera_calibration.motion_tool:main',
            'calib_gui = robot_camera_calibration.calib_gui:main',
        ],
    },
)
